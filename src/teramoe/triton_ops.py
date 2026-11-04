import paddle
from paddle import Tensor

paddle.enable_compat(scope={"triton"}, silent=True)

import triton
import triton.language as tl


@triton.jit
def _accumulate_wgrad_kernel(
    master_grad_ptr,
    replica_grad_0_ptr,
    replica_grad_1_ptr,
    replica_grad_2_ptr,
    replica_grad_3_ptr,
    replica_grad_4_ptr,
    replica_grad_5_ptr,
    replica_grad_6_ptr,
    size,
    NUM_REPLICAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size

    acc = tl.load(master_grad_ptr + offs, mask=mask)
    if NUM_REPLICAS > 0:
        acc += tl.load(replica_grad_0_ptr + offs, mask=mask)
    if NUM_REPLICAS > 1:
        acc += tl.load(replica_grad_1_ptr + offs, mask=mask)
    if NUM_REPLICAS > 2:
        acc += tl.load(replica_grad_2_ptr + offs, mask=mask)
    if NUM_REPLICAS > 3:
        acc += tl.load(replica_grad_3_ptr + offs, mask=mask)
    if NUM_REPLICAS > 4:
        acc += tl.load(replica_grad_4_ptr + offs, mask=mask)
    if NUM_REPLICAS > 5:
        acc += tl.load(replica_grad_5_ptr + offs, mask=mask)
    if NUM_REPLICAS > 6:
        acc += tl.load(replica_grad_6_ptr + offs, mask=mask)
    tl.store(master_grad_ptr + offs, acc, mask=mask)


def accumulate_wgrad_(master_grad: Tensor, replica_grads: list[Tensor]):
    """Accumulate replica_grads to master_grad in-place."""
    if not replica_grads:
        return
    assert len(replica_grads) <= 7, (
        f"Only supports at most 7 replica grads, got {len(replica_grads)}")
    for grad in replica_grads:
        assert master_grad.shape == grad.shape and master_grad.dtype == grad.dtype

    size = master_grad.size
    BLOCK = 1024
    grid = (triton.cdiv(size, BLOCK),)

    _accumulate_wgrad_kernel[grid](
        master_grad,
        *replica_grads,
        *(None,) * (7 - len(replica_grads)),
        size,
        len(replica_grads),
        BLOCK,
    )


@triton.jit
def _topk_to_dense_kernel(
    topk_ids_ptr,
    probs_ptr,
    routing_map_ptr,
    partial_loads_ptr,
    seq_len,
    num_experts,
    TOP_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_e = tl.program_id(1)
    rows = pid_t.to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    row_mask = rows < seq_len
    col_mask = cols < num_experts

    # probs holds the topk slot index k at each selected expert, so the order of
    # topk_probs can be restored after rerouting. Padding ids (-1) never match.
    probs = tl.zeros([BLOCK_T, BLOCK_E], dtype=tl.float32)
    hit = tl.zeros([BLOCK_T, BLOCK_E], dtype=tl.int32)
    for k in tl.static_range(TOP_K):
        ids = tl.load(topk_ids_ptr + rows * TOP_K + k, mask=row_mask, other=-1)
        match = ids[:, None] == cols[None, :]
        probs = tl.where(match, float(k), probs)
        hit = tl.where(match, 1, hit)

    offs = rows[:, None] * num_experts + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    tl.store(probs_ptr + offs, probs, mask=mask)
    tl.store(routing_map_ptr + offs, hit.to(tl.uint8), mask=mask)
    tl.store(partial_loads_ptr + pid_t * num_experts + cols, tl.sum(hit, axis=0), mask=col_mask)


@triton.jit
def _reduce_loads_kernel(
    partial_loads_ptr,
    expert_loads_ptr,
    num_blocks,
    num_experts,
    BLOCK_E: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    col_mask = cols < num_experts
    acc = tl.zeros([BLOCK_E], dtype=tl.int32)
    for i in range(num_blocks):
        acc += tl.load(partial_loads_ptr + i * num_experts + cols, mask=col_mask, other=0)
    tl.store(expert_loads_ptr + cols, acc, mask=col_mask)


def topk_to_dense(topk_ids: Tensor, num_experts: int) -> tuple[Tensor, Tensor, Tensor]:
    """Convert sparse topk_ids to dense (probs, routing_map, expert_loads).

    probs[t, topk_ids[t, k]] = k (float32), routing_map is uint8 one-hot, and
    expert_loads (int32) is the column sum of routing_map. Rows of all -1 are padding.
    """
    assert topk_ids.dtype in (paddle.int32, paddle.int64)
    seq_len, top_k = topk_ids.shape

    BLOCK_T = 64
    BLOCK_E = 128
    num_blocks = triton.cdiv(seq_len, BLOCK_T)

    probs = paddle.empty([seq_len, num_experts], dtype="float32")
    routing_map = paddle.empty([seq_len, num_experts], dtype="uint8")
    partial_loads = paddle.empty([num_blocks, num_experts], dtype="int32")
    expert_loads = paddle.empty([num_experts], dtype="int32")

    if seq_len == 0:
        return probs, routing_map, paddle.zeros([num_experts], dtype="int32")

    grid = (num_blocks, triton.cdiv(num_experts, BLOCK_E))
    _topk_to_dense_kernel[grid](
        topk_ids, probs, routing_map, partial_loads,
        seq_len, num_experts, top_k, BLOCK_T, BLOCK_E,
    )
    grid = (triton.cdiv(num_experts, BLOCK_E),)
    _reduce_loads_kernel[grid](partial_loads, expert_loads, num_blocks, num_experts, BLOCK_E)
    return probs, routing_map, expert_loads


@triton.jit
def _dense_to_topk_kernel(
    expanded_probs_ptr,
    expanded_routing_map_ptr,
    topk_ids_ptr,
    num_physical,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_P)
    col_mask = cols < num_physical

    hit = tl.load(expanded_routing_map_ptr + row * num_physical + cols, mask=col_mask, other=0)
    slot = tl.load(expanded_probs_ptr + row * num_physical + cols, mask=col_mask, other=0.0)
    slot = tl.where(hit != 0, slot.to(tl.int32), -1)

    # Each selected physical expert carries its topk slot index k; padding rows have no hits.
    ks = tl.arange(0, BLOCK_K)
    match = slot[:, None] == ks[None, :]
    ids = tl.max(tl.where(match, cols[:, None], -1), axis=0)
    tl.store(topk_ids_ptr + row * TOP_K + ks, ids.to(topk_ids_ptr.dtype.element_ty), mask=ks < TOP_K)


def dense_to_topk(expanded_probs: Tensor, expanded_routing_map: Tensor, top_k: int,
                  dtype=paddle.int64) -> Tensor:
    """Recover topk_ids from the rerouted dense mapping, keeping the original topk order.

    expanded_probs[t, p] holds the topk slot index k of physical expert p (see topk_to_dense).
    Rows without any routed expert produce -1.
    """
    seq_len, num_physical = expanded_probs.shape
    assert expanded_routing_map.shape == expanded_probs.shape
    if expanded_routing_map.dtype == paddle.bool:
        expanded_routing_map = expanded_routing_map.view("uint8")

    topk_ids = paddle.empty([seq_len, top_k], dtype=dtype)
    if seq_len == 0:
        return topk_ids

    _dense_to_topk_kernel[(seq_len,)](
        expanded_probs, expanded_routing_map, topk_ids, num_physical,
        top_k, triton.next_power_of_2(top_k), triton.next_power_of_2(num_physical),
    )
    return topk_ids
