import os
import sys
import time
import random
import argparse

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
import paddle.nn.functional as F

paddle.empty([32, 1024, 1024, 1024], "uint8")
paddle.set_printoptions(linewidth=200)

import teramoe
import teramoe.fused_a2a as fused_a2a
import teramoe.placement as placement
from teramoe.fp8_utils import quant_weight
from teramoe.triton_ops import accumulate_wgrad_

# use develop paddlefleet
FLEET_PATH = os.path.realpath("../../erniebot_test_speed/third_party/PaddleFleet/src")
sys.path.insert(0, FLEET_PATH)
import paddlefleet
assert paddlefleet.__path__[0] == os.path.join(FLEET_PATH, "paddlefleet"), (
    f"Unexpected paddlefleet path: {paddlefleet.__path__[0]}")
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed

SEQLEN = 16384
NUM_COMM_SMS = 52

E = 64
H = 4096
I = 2048
TOP_K = 4
NUM_REPLICAS = 2


def initialize_fleet():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "sharding_degree": world_size,
        "pp_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "dp_degree": 1,
        "ep_degree": world_size,
        "mp_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    paddlefleet.training.initialize.initialize_fleet(strategy=strategy)
    model_parallel_cuda_manual_seed(rank)
    group = ProcessGroupCollection.use_mpu_process_groups()
    return group


def dispatch_forward(buffer, hidden_states, token_indices, token_probs, num_experts):
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = buffer.get_dispatch_layout(
        token_indices,
        num_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    )

    return buffer.dispatch(
        hidden_states,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        is_token_in_rank=is_token_in_rank,
        async_finish=False,
        allocate_on_comm_stream=False,
    )


def make_map(EP, LE, R, kind, seed=0):
    """Build a physical_to_logical_map, identical on all ranks."""
    rng = random.Random(seed)
    m = []
    for r in range(EP):
        m += list(range(r * LE, (r + 1) * LE))
        for j in range(R):
            if kind == "random":  # includes -1, own experts and duplicated experts
                m.append(rng.choice([-1] + list(range(EP * LE))))
            elif kind == "ring":  # every rank sends and receives exactly R replicas
                m.append(((r + 1 + j) % EP) * LE + j)
            elif kind == "hot":  # one expert replicated to every slot, including its own rank
                m.append(3)
            elif kind == "empty":
                m.append(-1)
    return m


def nccl_weight_sync(group, m, master, replica):
    tasks, _ = placement.weight_sync(group, m, master, replica)
    for t in tasks:
        t.wait()


def check_weight_sync(buffer, group, master, m, R, H, I, num_sms):
    EP, LE = group.nranks, master[0].shape[0]
    assert len(m) == EP * (LE + R)
    ref = [paddle.zeros([R, *w.shape[1:]], dtype=w.dtype) for w in master]
    tasks, _ = placement.weight_sync(group, m, master, ref)
    for t in tasks:
        t.wait()

    map_gpu = paddle.to_tensor(m, dtype="int32")
    is_async = num_sms % 2 == 1
    previous_event = buffer.capture() if is_async else None
    out, event = buffer.weight_sync(
        map_gpu, master, R, H, I, num_sms, previous_event=previous_event, async_finish=is_async)
    event.current_stream_wait() if is_async else ()

    assert len(out) == len(master)
    num_checked = 0
    for j in range(R):
        if m[group.rank * (LE + R) + LE + j] == -1:
            continue
        for o, r in zip(out, ref):
            assert o.shape == r.shape and o.dtype == r.dtype
            assert bool(paddle.equal_all(o.view("uint8")[j], r.view("uint8")[j])), (
                f"rank {group.rank} replica {j} mismatch")
        num_checked += 1

    dist.barrier(group)
    return num_checked


def nccl_grad_reduce(group, m, replica_grad, master_grad):
    """Reference: grad_sync + accumulate_wgrad_."""
    LE = master_grad[0].shape[0]
    tasks, grads = placement.grad_sync(group, m, replica_grad, LE)
    for t in tasks:
        t.wait()
    for mg, gl in zip(master_grad, grads):
        for e in range(LE):
            # accumulate_wgrad_ takes at most 7 grads, splitting keeps the reduction order
            for i in range(0, len(gl[e]), 7):
                accumulate_wgrad_(mg[e], gl[e][i:i + 7])


def check_grad_reduce(buffer, group, master_grad_init, m, R, H, I, num_sms):
    EP, LE = group.nranks, master_grad_init[0].shape[0]
    replica_grad = buffer.get_replica_grads(R, H, I)
    for g, mg in zip(replica_grad, master_grad_init):
        assert g.dtype == paddle.float32 and g.shape == [R, *mg.shape[1:]]
        paddle.assign(paddle.randn(g.shape, dtype="float32"), output=g)

    ref = [g.clone() for g in master_grad_init]
    nccl_grad_reduce(group, m, replica_grad, ref)

    out = [g.clone() for g in master_grad_init]
    map_gpu = paddle.to_tensor(m, dtype="int32")
    is_async = num_sms % 2 == 1
    previous_event = buffer.capture() if is_async else None
    event = buffer.grad_reduce(
        map_gpu, out, R, H, I, num_sms, previous_event=previous_event, async_finish=is_async)
    event.current_stream_wait() if is_async else ()
    for o, r in zip(out, ref):
        assert bool((o == r).all()), (
            f"rank {group.rank} grad_reduce mismatch, max diff {(o - r).abs().max().item()}")

    dist.barrier(group)
    num_sources = sum(x == group.rank * LE + e for x in m for e in range(LE)) - LE
    return num_sources


def fp8_master(w1, w2, transpose):
    """Quantized weights as used by forward (transpose) or backward, with scales in their physical layout."""
    w1_fp8, w1_scale = quant_weight(w1, transpose=transpose)
    w2_fp8, w2_scale = quant_weight(w2, transpose=transpose)
    return [w1_fp8, w1_scale.mT, w2_fp8, w2_scale.mT]


def bench(fn, name, num_iters=10, num_warmups=3):
    events = [paddle.cuda.Event(enable_timing=True) for _ in range(num_iters * 2)]

    for _ in range(num_warmups):
        fn()

    paddle.base.core.nvprof_nvtx_push(name)

    for i in range(num_iters):
        paddle.randn([1024, 1024, 1024])
        dist.all_reduce(paddle.empty([1]))
        events[2 * i].record()
        paddle.base.core.nvprof_nvtx_push("run")
        fn()
        paddle.base.core.nvprof_nvtx_pop()
        events[2 * i + 1].record()

    paddle.base.core.nvprof_nvtx_pop()
    dist.barrier()

    # For each run, take the time of the slowest rank
    times_per_iter = paddle.to_tensor(
        [events[2 * i].elapsed_time(events[2 * i + 1]) for i in range(num_iters)])
    dist.all_reduce(paddle.to_tensor(times_per_iter), op=dist.ReduceOp.MAX)
    avg_time = times_per_iter.mean().item() / 1e3
    return avg_time


def main():
    pg_collection = initialize_fleet()
    group = pg_collection.ep
    fused_a2a.configure_buffer(NUM_COMM_SMS)
    EP = group.nranks
    assert EP == 8, f"This is an intranode test, got {EP=}"

    hidden_states = paddle.randn([SEQLEN, H], dtype="bfloat16")
    hidden_bytes = fused_a2a.get_hidden_bytes(hidden_states)

    # Heterogeneous layers, the buffer grows to the largest one (the small layer comes first to test regrowth)
    layers = [(E, H // 2, I // 2, 1), (E // 2, H, I // 2, 3), (E, H, I, NUM_REPLICAS)]
    for _, h, i, r in layers:
        replica_bytes = fused_a2a.get_replica_bytes(r, h, i)
        buffer = fused_a2a.get_buffer(group, hidden_bytes, replica_bytes)
    print("replica buffer bytes:", buffer.num_replica_bytes)

    # Warmup the buffer by dispatch
    scores = paddle.randn([SEQLEN, E])
    token_probs, token_indices = scores.topk(TOP_K)
    dispatch_forward(buffer, hidden_states, token_indices, token_probs, E)

    ################################ VALIDATE #################################

    cases = [("ring", 0), ("hot", 0), ("empty", 0)] + [("random", s) for s in range(8)]
    for num_experts, h, i, R in layers:
        LE = num_experts // EP
        w1 = paddle.randn([LE, h, 2 * i], dtype="bfloat16")
        w2 = paddle.randn([LE, i, h], dtype="bfloat16")
        g1 = paddle.randn(w1.shape, dtype="float32")
        g2 = paddle.randn(w2.shape, dtype="float32")
        masters = {"bf16": [w1, w2], "fp8-fwd": fp8_master(w1, w2, True), "fp8-bwd": fp8_master(w1, w2, False)}
        layer = f"E={num_experts} H={h} I={i} R={R}"
        for kind, seed in cases:
            m = make_map(EP, LE, R, kind, seed)
            for name, master in masters.items():
                for num_sms in (1, 3, 16):
                    n = check_weight_sync(buffer, group, master, m, R, h, i, num_sms)
            for num_sms in (1, 3, 16):
                n_src = check_grad_reduce(buffer, group, [g1, g2], m, R, h, i, num_sms)
            print(f"[{layer}] check passed: {kind}-{seed}, {n} replicas synced, {n_src} grad sources reduced")

    ################################# PROFILE #################################

    paddle.base.core.nvprof_start()

    LE, R = E // EP, NUM_REPLICAS
    w1 = paddle.randn([LE, H, 2 * I], dtype="bfloat16")
    w2 = paddle.randn([LE, I, H], dtype="bfloat16")
    m = make_map(EP, LE, R, "ring")
    map_gpu = paddle.to_tensor(m, dtype="int32")
    for name, master in (("bf16", [w1, w2]), ("fp8", fp8_master(w1, w2, True))):
        num_bytes = R * sum(w[0].numel().item() * w.element_size() for w in master)
        replica = [paddle.empty([R, *w.shape[1:]], dtype=w.dtype) for w in master]
        t = bench(lambda: nccl_weight_sync(group, m, master, replica), "nccl_weight_sync")
        if group.rank == 0:
            print(f"weight_sync {name} nccl: {t * 1e6:.0f} us, {num_bytes / t / 1e9:.1f} GB/s per rank")
        for num_sms in (4, 8, 16, 24, 32, 48):
            t = bench(lambda: buffer.weight_sync(map_gpu, master, R, H, I, num_sms), f"nvl_weight_sync_{num_sms}")
            if group.rank == 0:
                print(f"weight_sync {name} num_sms {num_sms:2d}: {t * 1e6:.0f} us, {num_bytes / t / 1e9:.1f} GB/s per rank")

    g1 = paddle.randn(w1.shape, dtype="float32")
    g2 = paddle.randn(w2.shape, dtype="float32")
    replica_grad = buffer.get_replica_grads(R, H, I)
    num_bytes = sum(g.numel().item() * g.element_size() for g in replica_grad)
    t = bench(lambda: nccl_grad_reduce(group, m, replica_grad, [g1, g2]), "nccl_grad_reduce")
    if group.rank == 0:
        print(f"grad_reduce nccl: {t * 1e6:.0f} us, {num_bytes / t / 1e9:.1f} GB/s per rank")
    for num_sms in (4, 8, 16, 24, 32, 48):
        t = bench(lambda: buffer.grad_reduce(map_gpu, [g1, g2], R, H, I, num_sms), f"nvl_grad_reduce_{num_sms}")
        if group.rank == 0:
            print(f"grad_reduce num_sms {num_sms:2d}: {t * 1e6:.0f} us, {num_bytes / t / 1e9:.1f} GB/s per rank")

    dist.barrier()
    paddle.base.core.nvprof_stop()
    print("PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("args", nargs="*", help="Unused args")
    args = parser.parse_args()

    main()
