import paddle
from paddle import Tensor
import paddle.distributed as dist
from paddle.base.core import task
from paddle.distributed.communication.group import Group


def weight_sync(group: Group, physical_to_logical_map: list[int], master: list[Tensor],
                replica: list[Tensor]) -> tuple[list[task], int]:
    """Distribute master weights to replicas."""
    EP = group.world_size
    LE = len(master[0])        # local logical experts
    PE = LE + len(replica[0])  # local physical experts
    LOFF = LE * group.rank
    POFF = PE * group.rank

    ops = []
    peers = set()

    for rank in range(EP):
        for phys_id in range(PE * rank + LE, PE * (rank + 1)):
            logi_id = physical_to_logical_map[phys_id]

            # 本地专家需要发送给外部 replica
            if LOFF <= logi_id < LOFF + LE:
                master_id = logi_id - LOFF
                dst_rank = group.ranks[rank]
                peers.add(dst_rank)
                for w in master:
                    ops.append(dist.P2POp(dist.isend, w[master_id], dst_rank, group))

            # 本地 replica 需要接收外部专家
            if rank == group.rank and logi_id != -1:
                replica_id = phys_id - POFF - LE
                src_rank = group.ranks[logi_id // LE]
                peers.add(src_rank)
                for w in replica:
                    ops.append(dist.P2POp(dist.irecv, w[replica_id], src_rank, group))

    return (dist.batch_isend_irecv(ops) if ops else []), len(peers)


def grad_sync(group: Group, physical_to_logical_map: list[int], replica: list[Tensor],
              num_experts: int) -> tuple[list[task], list[list[list[Tensor]]]]:
    """Aggregate replica gradients to masters."""
    EP = group.world_size
    LE = num_experts           # local logical experts
    PE = LE + len(replica[0])  # local physical experts
    LOFF = LE * group.rank
    POFF = PE * group.rank

    ops = []
    grad = [[[] for _ in range(LE)] for _ in replica]

    for rank in range(EP):
        for phys_id in range(PE * rank + LE, PE * (rank + 1)):
            logi_id = physical_to_logical_map[phys_id]

            # 本地专家需要从外部 replica 接收 grad
            if LOFF <= logi_id < LOFF + LE:
                master_id = logi_id - LOFF
                src_rank = group.ranks[rank]
                for w, g in zip(replica, grad):
                    t = paddle.empty_like(w[0])
                    ops.append(dist.P2POp(dist.irecv, t, src_rank, group))
                    g[master_id].append(t)

            # 本地 replica grad 需要发送给外部专家
            if rank == group.rank and logi_id != -1:
                replica_id = phys_id - POFF - LE
                dst_rank = group.ranks[logi_id // LE]
                for w in replica:
                    ops.append(dist.P2POp(dist.isend, w[replica_id], dst_rank, group))

    return (dist.batch_isend_irecv(ops) if ops else []), grad


def update_placement(
    group: Group, topk_probs: Tensor, topk_ids: Tensor, num_experts: int, num_replicas: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Update expert placement for a single layer based on real-time load statistics."""
    import ultra_ep

    NVL_DOMAIN_SIZE = 8
    EP = group.world_size
    seq_len, top_k = topk_ids.shape
    padding_mask = topk_ids == -1
    assert (padding_mask.any(axis=1) == padding_mask.all(axis=1)).all()

    # preprocess: keep the order of topk_probs
    topk_probs_for_sort = paddle.arange(top_k, dtype="int32").float().broadcast_to(topk_probs.shape)
    probs = paddle.zeros([seq_len, num_experts]).put_along_axis(topk_ids, topk_probs_for_sort, axis=-1)
    routing_map = paddle.zeros_like(probs, dtype="uint8").put_along_axis(
        topk_ids, paddle.ones([1], dtype="uint8"), axis=-1)

    paddle.base.core.nvprof_nvtx_push("all_gather")
    expert_loads = routing_map.sum(axis=0, dtype="int32")
    expert_loads_per_rank = paddle.empty([EP, num_experts], dtype="int32")
    dist.all_gather(expert_loads_per_rank, expert_loads, group)
    expert_loads = expert_loads_per_rank.sum(axis=0, dtype="int32")
    paddle.base.core.nvprof_nvtx_pop()

    paddle.base.core.nvprof_nvtx_push("solve")
    (
        physical_to_logical_map,
        logical_to_physical_map,
        logical_replica_counts,
        logical_instance_quota,
        logical_instance_quota_prefix,
        rank_quota_prefix,
    ) = ultra_ep._C.solve_placement_for_test(
        expert_loads,
        expert_loads_per_rank,
        EP,
        num_experts // EP,
        num_replicas,
        NVL_DOMAIN_SIZE,
        min_tokens_per_replica=1024,
        rank_quota_source_rank=group.rank,
    )
    paddle.base.core.nvprof_nvtx_pop()

    paddle.base.core.nvprof_nvtx_push("reroute")
    expanded_probs, expanded_routing_map = ultra_ep._C.dense_reroute_for_test(
        routing_map.view("bool"),
        probs,
        logical_to_physical_map,
        logical_replica_counts,
        rank_quota_prefix,
        num_experts + num_replicas * EP,
    )
    paddle.base.core.nvprof_nvtx_pop()

    # post process: restore the order of topk_probs
    expected_routing_map = paddle.zeros_like(expanded_routing_map, dtype="uint8").put_along_axis(
        topk_ids + topk_ids // (num_experts // EP) * num_replicas, paddle.ones([1], dtype="uint8"),
        axis=-1)
    expanded_probs = paddle.where(
        expanded_routing_map, expanded_probs, paddle.full_like(expanded_probs, float("inf")))
    rerouted_topk_probs, rerouted_topk_ids = expanded_probs.topk(
        top_k, axis=1, sorted=True, largest=False)
    rerouted_topk_probs = paddle.where(
        padding_mask, paddle.zeros_like(rerouted_topk_probs), rerouted_topk_probs)
    rerouted_topk_ids = paddle.where(
        padding_mask, paddle.full_like(rerouted_topk_ids, -1), rerouted_topk_ids)
    rerouted_topk_probs = topk_probs.take_along_axis(rerouted_topk_probs.int32(), axis=1)

    assert (topk_probs == rerouted_topk_probs).all()

    return rerouted_topk_probs, rerouted_topk_ids, physical_to_logical_map


def prioritize_comm_stream():
    """Launch some dummy kernerls in calc stream to let comm kernels launch first."""
    for _ in range(3):
        paddle.zeros([1])
