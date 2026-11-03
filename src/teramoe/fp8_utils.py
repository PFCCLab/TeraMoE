from contextlib import contextmanager

import paddle
import paddle.distributed as dist
from paddle import Tensor
from paddle.distributed.communication.group import Group

from . import deep_ep, deep_gemm
from .async_utils import AsyncLoad, async_offload, get_profile_timers, started
from .task_utils import GroupedTaskLauncher, make_batch_task_queue
from .placement import weight_sync, grad_sync, update_placement, prioritize_comm_stream
from .triton_ops import accumulate_wgrad_

FP8_ALIGN = 128
QUANT_BLOCK_SIZE = 512


def quant_input(x: Tensor, transpose_scale: bool = False) -> tuple[Tensor, Tensor]:
    """对于 hidden_states, 在 hidden 维上使用 128 分块量化."""
    x_fp8, scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        x,
        quant_method="1x128",
        output_scale_transpose=transpose_scale,
        using_ue8m0_scale=True,
    )
    assert x_fp8.shape == x.shape
    scale_shape = [x.shape[0], x.shape[1] // QUANT_BLOCK_SIZE]
    assert scale.shape == (scale_shape[::-1] if transpose_scale else scale_shape)
    assert x_fp8.is_contiguous()
    assert scale.is_contiguous()
    return x_fp8, scale


def quant_weight(w: Tensor, transpose: bool = False) -> tuple[Tensor, Tensor]:
    """对于 weight, 在最后一维使用 128 分块量化."""
    import paddlefleet_ops
    expert_weight_list = list(w)  # quant 算子只接受 list 输入

    if transpose:
        w_fp8, scale = paddlefleet_ops.fuse_stack_transpose_fp8_quant(
            expert_weight_list,
            using_pow2_scaling=False,
            using_ue8m0_scale=True,
            output_scale_transpose=False,
        )
        assert w_fp8.shape == [w.shape[0] * w.shape[2], w.shape[1]]
        assert scale.shape == [w.shape[0] * w.shape[2], w.shape[1] // QUANT_BLOCK_SIZE]
        assert w_fp8.is_contiguous()
        assert scale.is_contiguous()
    else:
        w_fp8, scale = paddlefleet_ops.fuse_stack_fp8_quant(
            expert_weight_list,
            using_pow2_scaling=False,
            using_ue8m0_scale=True,
            output_scale_transpose=False,
        )
        assert w_fp8.shape == [w.shape[0] * w.shape[1], w.shape[2]]
        assert scale.shape == [w.shape[0] * w.shape[1], w.shape[2] // QUANT_BLOCK_SIZE]
        assert w_fp8.is_contiguous()
        assert scale.is_contiguous()

    # quant 算子输出把专家维铺平了，需要重新展开
    w_fp8 = w_fp8.reshape([w.shape[0], -1, w_fp8.shape[1]])
    scale = scale.reshape([w.shape[0], -1, scale.shape[1]])
    # ue8m0 要求 scale 最后两维 transpose
    scale = scale.transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])
    return w_fp8, scale


def get_quant_weight(w: Tensor, transpose: bool = False) -> tuple[Tensor, Tensor]:
    """从预处理的 fp8_cache 中获取 weight 的量化结果, 若不存在则现场量化."""
    if transpose:
        if getattr(w, "fp8_weight_stacked_transpose", None) is not None:
            w_fp8, scale = w.fp8_weight_stacked_transpose, w.fp8_scale_stacked_transpose
            assert w_fp8.shape[-1] == w.shape[1]
            assert scale.shape[-1] == w.shape[1] // QUANT_BLOCK_SIZE
            assert w_fp8.is_contiguous()
            assert scale.is_contiguous()
            w_fp8 = w_fp8.reshape([w.shape[0], w.shape[2], w_fp8.shape[-1]])
            scale = scale.reshape([w.shape[0], w.shape[2], scale.shape[-1]])
            # TODO: scale 应该提前 transpose 吗
            scale = scale.transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])
            return w_fp8, scale
    else:
        if hasattr(w, "fp8_weight_stacked"):
            w_fp8, scale = w.fp8_weight_stacked, w.fp8_scale_stacked
            assert w_fp8.shape[-1] == w.shape[2]
            assert scale.shape[-1] == w.shape[2] // QUANT_BLOCK_SIZE
            assert w_fp8.is_contiguous()
            assert scale.is_contiguous()
            w_fp8 = w_fp8.reshape([w.shape[0], w.shape[1], w_fp8.shape[-1]])
            scale = scale.reshape([w.shape[0], w.shape[1], scale.shape[-1]])
            scale = scale.transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])
            return w_fp8, scale
    return quant_weight(w, transpose)


def dequant_input(x: Tensor, scale: Tensor) -> Tensor:
    if len(x) == 0:
        return paddle.empty(x.shape, dtype="bfloat16")
    return paddle.incubate.nn.functional.fused_act_dequant(x, scale)


@contextmanager
def cached_notify_guard() -> deep_ep.EventOverlap:
    """Make the calc_stream after this guard wait for notify_dispatch to finish."""
    event = deep_ep.Buffer.capture()
    try:
        yield event
    finally:
        event.current_stream_wait()


class FP8Weight:
    def __init__(self, fp8: Tensor, scale: Tensor):
        self.fp8 = fp8
        self.scale = scale
        self.shape = fp8.shape

    def __iter__(self):
        yield self.fp8
        yield self.scale


class TeraMoENode:
    def __init__(
        self,
        buffer: deep_ep.Buffer,
        hidden_states: Tensor,
        token_probs: Tensor,
        token_indices: Tensor,
        w_gateup: Tensor,
        w_down: Tensor,
        group: Group,
        num_experts: int,
        fp8: str,
        fp8_wgrad: bool,
        chunk_size: int,
        num_calc_sms: int,
        combine_overlap_ratio: float,
        activation: dict[str, int | float],
        barrier_async: bool,
    ):
        self.buffer = buffer
        self.hidden_states = hidden_states
        self.token_probs = token_probs
        self.token_indices = token_indices
        self.w_gateup = w_gateup
        self.w_down = w_down
        self.group = group
        self.num_experts = num_experts
        self.fp8 = fp8
        self.fp8_wgrad = fp8_wgrad
        self.chunk_size = chunk_size
        self.num_calc_sms = num_calc_sms
        self.combine_overlap_ratio = combine_overlap_ratio
        self.activation = activation
        self.barrier_async = barrier_async

        self.avg_tokens_per_card = token_indices.size  # seqlen * topk
        self.avg_num_tasks = int(self.avg_tokens_per_card / chunk_size + len(w_gateup) / 2)
        self.enable_overlap = self.buffer.runtime.get_num_rdma_ranks() > 1

        timers = get_profile_timers()
        self.dispatch_timer = timers("dispatch", use_event=True)
        self.compute_timer = timers("fusion_mlp", use_event=True)
        self.combine_timer = timers("combine", use_event=True)

        # ultraep
        self.num_replicas = 2
        self.num_dispatch_experts = num_experts + self.num_replicas * group.world_size
        self.num_permute_experts = len(w_gateup) + self.num_replicas
        self.enable_overlap = False

    def _reduce_max_tokens_per_card(self, num_tokens_per_expert: Tensor):
        """归约当前 ep_group 内每个 rank 的展开 token 数的最大值, 用于统计均衡度.

        NOTES:
            1. 通信量为 ep_size 个 int, 成本极低;
            2. 同时起到了 barrier_ep 的作用, 可选 sync 或 async;
            3. offload 结果由 notify_dispatch 天然同步, 需要在调用完 dispatch 后才能使用.
        """
        global_num_tokens_per_card = num_tokens_per_expert.reshape(
            [self.group.nranks, -1]).sum(axis=1, dtype="int32")
        dist.all_reduce(global_num_tokens_per_card, group=self.group)
        value = global_num_tokens_per_card.max()
        self.max_tokens_per_card = (
            async_offload(value) if self.barrier_async else value.pin_memory())

    def _sync_replica_weights(self, fp8_weights=None):
        if isinstance(self.physical_to_logical_map, Tensor):
            self.physical_to_logical_map = self.physical_to_logical_map.tolist()

        if self.fp8:
            (w1_fp8, w1_scale), (w2_fp8, w2_scale) = fp8_weights
            src = [w1_fp8, w1_scale.mT, w2_fp8, w2_scale.mT]
            w1, w2 = self.w_gateup_replica, self.w_down_replica
            dst = [w1.fp8, w1.scale.mT, w2.fp8, w2.scale.mT]
            for t1, t2 in zip(src, dst):
                assert t1.is_contiguous() and t2.is_contiguous() and t1.shape[1:] == t2.shape[1:]
        else:
            src = [self.w_gateup, self.w_down]
            dst = [self.w_gateup_replica, self.w_down_replica]

        paddle.base.core.nvprof_nvtx_push("weight_sync")
        tasks, self.num_peers = weight_sync(self.group, self.physical_to_logical_map, src, dst)
        paddle.base.core.nvprof_nvtx_pop()

        return tasks

    def _clear_replica_grads(self):
        """Create main_grad for replica experts and set to zero."""
        w1, w2 = self.w_gateup_replica, self.w_down_replica
        paddle.base.core.nvprof_nvtx_push("clear_grad")
        w1.main_grad = paddle.zeros(w1.shape, dtype="float32")
        w2.main_grad = paddle.zeros(w2.shape, dtype="float32")
        paddle.base.core.nvprof_nvtx_pop()

    def _sync_replica_grads(self):
        paddle.base.core.nvprof_nvtx_push("grad_sync")
        out = grad_sync(
            self.group, self.physical_to_logical_map,
            [self.w_gateup_replica.main_grad, self.w_down_replica.main_grad], len(self.w_gateup))
        paddle.base.core.nvprof_nvtx_pop()

        prioritize_comm_stream()
        deep_gemm.set_num_sms(148 - self.num_peers * 4)

        return out

    def forward(self):
        """Forward dispatch, compute and combine.

        This function has no input/output, as the inputs are set at node initialization,
        and the output is hold util the user calls forward_wait.
        """
        paddle.base.core.nvprof_nvtx_push("teramoe_forward")
        self.dispatch_timer.start()

        if self.num_replicas > 0:
            _, I, H = self.w_down.shape
            N = self.num_replicas

            if self.fp8:
                # forward uses transposed weights
                self.w_gateup_replica = FP8Weight(
                    paddle.empty([N, 2 * I, H], dtype="float8_e4m3fn"),
                    paddle.empty([N, H // QUANT_BLOCK_SIZE, 2 * I], dtype="int32").mT)
                self.w_down_replica = FP8Weight(
                    paddle.empty([N, H, I], dtype="float8_e4m3fn"),
                    paddle.empty([N, I // QUANT_BLOCK_SIZE, H], dtype="int32").mT)
            else:
                self.w_gateup_replica = paddle.empty([N, H, 2 * I], dtype="bfloat16")
                self.w_down_replica = paddle.empty([N, I, H], dtype="bfloat16")

            paddle.base.core.nvprof_nvtx_push("update_placement")
            self.token_probs, self.token_indices, physical_to_logical_map = update_placement(
                self.group, self.token_probs, self.token_indices, self.num_experts, N)
            self.physical_to_logical_map = async_offload(physical_to_logical_map)
            paddle.base.core.nvprof_nvtx_pop()

        if self.enable_overlap:
            if self.fp8:
                self._forward_fp8()
            else:
                self._forward_bf16()
        else:
            self._forward_serial()

        paddle.base.core.nvprof_nvtx_pop()

    def _forward_bf16(self):
        ########################### DISPATCH FORWARD ###########################

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.buffer.get_dispatch_layout(
            self.token_indices,
            self.num_experts,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        self._reduce_max_tokens_per_card(num_tokens_per_expert)

        self.dispatch_layout = {
            "num_tokens_per_rank": num_tokens_per_rank,
            "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "is_token_in_rank": is_token_in_rank,
        }

        (
            recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle, event,
            unzipped_tokens, unzipped_probs, atomic_to_zip, zip_to_atomic,
            num_valid_topk, task_queue,
        ) = self.buffer.dispatch(
            self.hidden_states,
            topk_idx=self.token_indices,
            topk_weights=self.token_probs,
            **self.dispatch_layout,
            async_finish=True,
            allocate_on_comm_stream=False,
            unzip_alignment=FP8_ALIGN,
            unzip_chunk_size=self.chunk_size,
        )

        del self.hidden_states
        self.recv_x = recv_x
        self.tokens_per_expert = tokens_per_expert
        self.unzipped_probs = unzipped_probs
        self.zip_to_atomic = zip_to_atomic

        ############################# GEMM FORWARD #############################

        w_gateup, w_down = self.w_gateup, self.w_down
        H, I = w_gateup.shape[1], w_down.shape[1]
        num_recv_tokens = len(recv_x)
        num_unzipped_tokens = len(unzipped_tokens)

        o1 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="bfloat16")
        o2 = paddle.empty([num_unzipped_tokens, I], dtype="bfloat16")
        o3 = paddle.empty([num_unzipped_tokens, H], dtype="bfloat16")
        zipped_out = paddle.empty([num_recv_tokens, H], dtype="bfloat16")

        token_done = paddle.zeros([num_recv_tokens], dtype="int32")
        zip_done = paddle.zeros([num_recv_tokens], dtype="int32")

        funcs = [
            lambda task_idx: deep_gemm.bf16_chunk_gemm_nn(
                unzipped_tokens, w_gateup, o1, task_queue, task_idx),
            lambda task_idx: deep_gemm.chunk_weighted_swiglu(
                o1, unzipped_probs, o2, task_queue, task_idx, self.chunk_size, **self.activation),
            lambda task_idx: deep_gemm.bf16_chunk_gemm_nn(o2, w_down, o3, task_queue, task_idx),
            lambda task_idx: deep_gemm.chunk_zip(
                o3, zipped_out, atomic_to_zip, zip_to_atomic, recv_token_indices, num_valid_topk,
                token_done, zip_done, task_queue, task_idx, self.chunk_size)
        ]

        task_launcher = GroupedTaskLauncher(
            funcs, len(task_queue), event, self.combine_overlap_ratio, self.avg_num_tasks)

        deep_gemm.set_num_sms(self.num_calc_sms)
        task_launcher.run_dispatch_overlap()
        self.dispatch_timer.stop()

        deep_gemm.set_num_sms(0)
        with started(self.compute_timer):
            task_launcher.run_compute()

        ############################# COMBINE FORWARD ##############################

        out, _, event = self.buffer.combine(
            zipped_out, handle, async_finish=True, allocate_on_comm_stream=False,
            zip_done=zip_done)

        deep_gemm.set_num_sms(self.num_calc_sms)
        self.combine_timer.start()
        task_launcher.run_combine_overlap()

        self.o1 = o1
        self.out = out
        self.combine_done_event = event

    def _forward_fp8(self):
        ########################### DISPATCH FORWARD ###########################

        hidden_states = quant_input(self.hidden_states)
        del self.hidden_states

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.buffer.get_dispatch_layout(
            self.token_indices,
            self.num_experts,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        self._reduce_max_tokens_per_card(num_tokens_per_expert)

        self.dispatch_layout = {
            "num_tokens_per_rank": num_tokens_per_rank,
            "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "is_token_in_rank": is_token_in_rank,
        }

        (
            recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle, event,
            unzipped_tokens, unzipped_probs, atomic_to_zip, zip_to_atomic,
            num_valid_topk, task_queue,
        ) = self.buffer.dispatch(
            hidden_states,
            topk_idx=self.token_indices,
            topk_weights=self.token_probs,
            **self.dispatch_layout,
            async_finish=True,
            allocate_on_comm_stream=False,
            unzip_alignment=FP8_ALIGN,
            unzip_chunk_size=self.chunk_size,
        )

        unzipped_tokens, unzipped_scale = unzipped_tokens
        unzipped_x = (unzipped_tokens, unzipped_scale.T)

        del hidden_states
        self.recv_x = recv_x
        self.tokens_per_expert = tokens_per_expert
        self.unzipped_probs = unzipped_probs
        self.zip_to_atomic = zip_to_atomic
        if not self.fp8_wgrad:
            self.recv_token_indices = recv_token_indices
            self.handle = handle

        ############################# GEMM FORWARD #############################

        w_gateup, w_down = self.w_gateup, self.w_down
        H, I = w_gateup.shape[1], w_down.shape[1]
        num_recv_tokens = len(recv_x[0])
        num_unzipped_tokens = len(unzipped_tokens)

        w_gateup_t = get_quant_weight(w_gateup, transpose=True)
        w_down_t = get_quant_weight(w_down, transpose=True)

        o1 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="bfloat16")
        o2_fp8 = paddle.empty([num_unzipped_tokens, I], dtype="float8_e4m3fn")
        o2_scale = paddle.empty([I // QUANT_BLOCK_SIZE, num_unzipped_tokens], dtype="int32").T
        o3 = paddle.empty([num_unzipped_tokens, H], dtype="bfloat16")
        zipped_out = paddle.empty([num_recv_tokens, H], dtype="bfloat16")

        token_done = paddle.zeros([num_recv_tokens], dtype="int32")
        zip_done = paddle.zeros([num_recv_tokens], dtype="int32")

        funcs = [
            lambda task_idx: deep_gemm.fp8_chunk_gemm_nt(
                unzipped_x, w_gateup_t, o1, task_queue, task_idx),
            lambda task_idx: deep_gemm.chunk_weighted_swiglu(
                o1, unzipped_probs, o2_fp8, task_queue, task_idx, self.chunk_size,
                **self.activation, o2_scales=o2_scale),
            lambda task_idx: deep_gemm.fp8_chunk_gemm_nt(
                (o2_fp8, o2_scale), w_down_t, o3, task_queue, task_idx),
            lambda task_idx: deep_gemm.chunk_zip(
                o3, zipped_out, atomic_to_zip, zip_to_atomic, recv_token_indices, num_valid_topk,
                token_done, zip_done, task_queue, task_idx, self.chunk_size),
        ]

        task_launcher = GroupedTaskLauncher(
            funcs, len(task_queue), event, self.combine_overlap_ratio, self.avg_num_tasks)

        deep_gemm.set_num_sms(self.num_calc_sms)
        task_launcher.run_dispatch_overlap()
        self.dispatch_timer.stop()

        deep_gemm.set_num_sms(0)
        with started(self.compute_timer):
            task_launcher.run_compute()

        ########################### COMBINE FORWARD ############################

        with cached_notify_guard() as event:
            out, _, event = self.buffer.combine(
                zipped_out, handle, async_finish=True, previous_event=event,
                allocate_on_comm_stream=False, zip_done=zip_done)

            deep_gemm.set_num_sms(self.num_calc_sms)
            self.combine_timer.start()
            task_launcher.run_combine_overlap()

        self.o1 = o1
        self.out = out
        self.combine_done_event = event

        ################################ WGRAD #################################

        # NOTES: 将 wgrad 的一部分工作提前到前向, 降低反向负载

        # 各专家 token 数重新向 512 对齐
        ks_cpu, m_start, m_start_wgrad = [], [0], [0]
        wgrad_align = QUANT_BLOCK_SIZE if self.fp8_wgrad else FP8_ALIGN
        for n in tokens_per_expert:
            ks_cpu.append((n + wgrad_align - 1) // wgrad_align * wgrad_align)
            m_start.append(m_start[-1] + (n + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN)
            m_start_wgrad.append(m_start_wgrad[-1] + ks_cpu[-1])
        self.ks_cpu = ks_cpu
        self.m_start_wgrad = m_start_wgrad

        async_load = AsyncLoad()
        t = async_load(ks_cpu + m_start + m_start_wgrad + tokens_per_expert, dtype="int32")
        self.grouped_layout = t[:len(ks_cpu)]
        self.m_start_gpu = t[len(ks_cpu) : len(ks_cpu) + len(m_start)]
        self.m_start_wgrad_gpu = t[len(ks_cpu) + len(m_start) : len(ks_cpu) + len(m_start) * 2]
        self.tokens_per_expert_gpu = t[len(ks_cpu) + len(m_start) * 2:]

        self.ordered_to_zip = paddle.empty([m_start_wgrad[-1]], dtype="int32")
        ordered_to_atomic = paddle.empty_like(self.ordered_to_zip) if not self.fp8_wgrad else None
        deep_gemm.sort_map(zip_to_atomic, self.m_start_gpu, m_start_wgrad[-1],
                           self.ordered_to_zip, ordered_to_atomic, self.m_start_wgrad_gpu)

        # 提前为 bf16_wgrad 恢复顺序
        if not self.fp8_wgrad:
            self.o1 = deep_gemm.token_gather(self.o1, ordered_to_atomic)
            self.unzipped_probs = deep_gemm.token_gather(
                self.unzipped_probs.unsqueeze(1), ordered_to_atomic).squeeze(1)
            self.recv_token_indices = self.recv_token_indices.cast("int32")

    def _forward_serial(self):
        """Simple fallback for intranode."""
        ########################### DISPATCH FORWARD ###########################

        hidden_states = quant_input(self.hidden_states) if self.fp8 else self.hidden_states
        del self.hidden_states

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = self.buffer.get_dispatch_layout(
            self.token_indices,
            self.num_dispatch_experts,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        self._reduce_max_tokens_per_card(num_tokens_per_expert)

        (
            recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle, event
        ) = self.buffer.dispatch(
            hidden_states,
            topk_idx=self.token_indices,
            topk_weights=self.token_probs,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            is_token_in_rank=is_token_in_rank,
            async_finish=True,
            allocate_on_comm_stream=False,
        )

        w_gateup, w_down = self.w_gateup, self.w_down
        E, I, H = w_down.shape
        if self.fp8:
            w_gateup_t = get_quant_weight(w_gateup, transpose=True)
            w_down_t = get_quant_weight(w_down, transpose=True)

        ks_cpu = []
        for i, n in enumerate(tokens_per_expert):
            ks_cpu.append((n + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN)
        async_load = AsyncLoad()
        self.ks_cpu = ks_cpu
        t = async_load(ks_cpu + tokens_per_expert, dtype="int32")
        self.grouped_layout = t[:len(ks_cpu)]
        self.tokens_per_expert_gpu = t[len(ks_cpu):]
        self.m_indices = paddle.repeat_interleave(
            paddle.arange(len(tokens_per_expert), dtype="int32"), self.grouped_layout)

        M = sum(self.ks_cpu[:E])
        self.m_indices, self.m_indices_replica = self.m_indices[:M], self.m_indices[M:] - E

        task_queue_np = make_batch_task_queue(tokens_per_expert)
        async_load.wait()

        event.current_stream_wait()
        self.dispatch_timer.stop()
        self.compute_timer.start()

        del hidden_states
        self.recv_x = recv_x
        recv_token_indices = recv_token_indices.cast("int32")
        self.recv_token_indices = recv_token_indices
        self.tokens_per_expert = tokens_per_expert
        self.handle = handle

        ############################# GEMM FORWARD #############################

        tasks = self._sync_replica_weights((w_gateup_t, w_down_t) if self.fp8 else None)

        if self.fp8:
            recv_x, recv_scale = recv_x
        else:
            recv_scale = None

        with paddle.amp.auto_cast(False):
            x, rowmap, unzipped_probs, x_scale = paddle.nn.functional.moe_permute(
                recv_x, recv_scale, recv_token_indices, recv_token_probs,
                padding_alignment=FP8_ALIGN, num_experts=self.num_permute_experts,
                tokens_per_expert=tokens_per_expert, using_ue8m0_scale=True)

        for task in tasks:
            task.wait()

        num_unzipped_tokens = len(x)

        if self.fp8:
            o1 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="bfloat16")
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (x[:M], x_scale[:M].T.contiguous().T), w_gateup_t, o1[:M], self.m_indices)
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (x[M:], x_scale[M:].T.contiguous().T), tuple(self.w_gateup_replica), o1[M:],
                self.m_indices_replica)
            del x, x_scale

            o2_fp8 = paddle.empty([num_unzipped_tokens, I], dtype="float8_e4m3fn")
            o2_scale = paddle.empty([I // QUANT_BLOCK_SIZE, num_unzipped_tokens], dtype="int32").T
            deep_gemm.chunk_weighted_swiglu(
                o1, unzipped_probs, o2_fp8, async_load(task_queue_np), task_idx=-1,
                chunk_size=FP8_ALIGN, **self.activation, o2_scales=o2_scale)

            o3 = paddle.empty([num_unzipped_tokens, H], dtype="bfloat16")
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (o2_fp8[:M], o2_scale[:M].T.contiguous().T), w_down_t, o3[:M], self.m_indices)
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (o2_fp8[M:], o2_scale[M:].T.contiguous().T), tuple(self.w_down_replica), o3[M:],
                self.m_indices_replica)
            del o2_fp8, o2_scale
        else:
            o1 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="bfloat16")
            deep_gemm.m_grouped_bf16_gemm_nn_contiguous(x[:M], w_gateup, o1[:M], self.m_indices)
            deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
                x[M:], self.w_gateup_replica, o1[M:], self.m_indices_replica)
            del x

            o2 = paddle.empty([num_unzipped_tokens, I], dtype="bfloat16")
            deep_gemm.chunk_weighted_swiglu(
                o1, unzipped_probs, o2, async_load(task_queue_np), task_idx=-1,
                chunk_size=FP8_ALIGN, **self.activation)

            o3 = paddle.empty([num_unzipped_tokens, H], dtype="bfloat16")
            deep_gemm.m_grouped_bf16_gemm_nn_contiguous(o2[:M], w_down, o3[:M], self.m_indices)
            deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
                o2[M:], self.w_down_replica, o3[M:], self.m_indices_replica)
            del o2

        ########################### COMBINE FORWARD ###########################

        with paddle.amp.auto_cast(False):
            zipped_out, _ = paddle.nn.functional.moe_unpermute(
                o3, rowmap, recv_token_indices, unzipped_probs,
                total_zipped_tokens=len(recv_token_indices), num_experts=self.num_permute_experts)
        del o3

        self.compute_timer.stop()
        self.combine_timer.start()

        with cached_notify_guard() as event:
            out, _, event = self.buffer.combine(
                zipped_out, handle, async_finish=True, previous_event=event,
                allocate_on_comm_stream=False)

        self.unzipped_probs = unzipped_probs
        self.o1 = o1
        self.out = out
        self.combine_done_event = event

    def forward_wait(self) -> Tensor:
        """Wait for combine to finish and return the combined result."""
        out = self.out
        self.combine_done_event.current_stream_wait()
        self.combine_timer.stop()
        self.tokens_per_card_max_mean_ratio = (
            self.max_tokens_per_card.item() / self.avg_tokens_per_card)
        del self.out, self.combine_done_event, self.max_tokens_per_card
        del self.w_gateup_replica, self.w_down_replica
        return out

    def backward(self, dout: Tensor):
        """Backward dispatch, compute, combine and wgrad.

        This function accepts one input, and has no output in the same way as forward.
        """
        paddle.base.core.nvprof_nvtx_push("teramoe_backward")

        if self.num_replicas > 0:
            _, I, H = self.w_down.shape
            N = self.num_replicas

            if self.fp8:
                self.w_gateup_replica = FP8Weight(
                    paddle.empty([N, H, 2 * I], dtype="float8_e4m3fn"),
                    paddle.empty([N, 2 * I // QUANT_BLOCK_SIZE, H], dtype="int32").mT)
                self.w_down_replica = FP8Weight(
                    paddle.empty([N, I, H], dtype="float8_e4m3fn"),
                    paddle.empty([N, H // QUANT_BLOCK_SIZE, I], dtype="int32").mT)
            else:
                self.w_gateup_replica = paddle.empty([N, H, 2 * I], dtype="bfloat16")
                self.w_down_replica = paddle.empty([N, I, H], dtype="bfloat16")

        if self.enable_overlap:
            if self.fp8:
                if self.fp8_wgrad:
                    self._backward_fp8(dout)
                else:
                    self._backward_fp8_bf16_wgrad(dout)
            else:
                self._backward_bf16(dout)
        else:
            self._backward_serial(dout)
        # nvtx pop on wait

    def _backward_bf16(self, dout: Tensor):
        ############################# COMBINE BACKWARD #############################

        # 本来 dispatch 反向应该用 cache_mode, 但目前 cache_mode 无法计算 atomic_to_zip/zip_to_atomic,
        # 因此只能像前向一样再跑一遍, 会浪费一定的通信带宽
        (
            _, recv_token_indices, recv_token_probs, tokens_per_expert, handle, event,
            do3, _, atomic_to_zip_bwd, zip_to_atomic_bwd, num_valid_topk, task_queue,
        ) = self.buffer.dispatch(
            dout,
            topk_idx=self.token_indices,
            topk_weights=self.token_probs,  # 无用, 只是非 cache_mode 必须传入
            **self.dispatch_layout,
            async_finish=True,
            allocate_on_comm_stream=False,
            unzip_alignment=FP8_ALIGN,
            unzip_chunk_size=self.chunk_size,
        )

        del self.token_indices, self.token_probs, self.dispatch_layout

        ############################## GEMM BACKWARD ###############################

        w_gateup, w_down = self.w_gateup, self.w_down
        H, I = w_gateup.shape[1], w_down.shape[1]
        num_recv_tokens = len(recv_token_probs)
        num_unzipped_tokens = len(do3)

        dx = paddle.empty([num_unzipped_tokens, H], dtype="bfloat16")
        do1 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="bfloat16")
        do2 = paddle.empty([num_unzipped_tokens, I], dtype="bfloat16")
        o2_bwd = paddle.empty([num_unzipped_tokens, I], dtype="bfloat16")
        drecv_x = paddle.empty([num_recv_tokens, H], dtype="bfloat16")
        drecv_probs = paddle.zeros_like(recv_token_probs)  # 无效位预先填0

        token_done = paddle.zeros([num_recv_tokens], dtype="int32")
        zip_done = paddle.zeros([num_recv_tokens], dtype="int32")

        funcs = [
            lambda task_idx: deep_gemm.bf16_chunk_gemm_nt(do3, w_down, do2, task_queue, task_idx),
            lambda task_idx: deep_gemm.chunk_weighted_swiglu_grad(
                self.o1, self.unzipped_probs, do2, o2_bwd, do1, drecv_probs,
                atomic_to_zip_bwd, self.zip_to_atomic, recv_token_indices, task_queue, task_idx,
                self.chunk_size, **self.activation),
            lambda task_idx: deep_gemm.bf16_chunk_gemm_nt(do1, w_gateup, dx, task_queue, task_idx),
            lambda task_idx: deep_gemm.chunk_zip(
                dx, drecv_x, atomic_to_zip_bwd, zip_to_atomic_bwd, recv_token_indices,
                num_valid_topk, token_done, zip_done, task_queue, task_idx, self.chunk_size),
        ]

        task_launcher = GroupedTaskLauncher(
            funcs, len(task_queue), event, self.combine_overlap_ratio)

        deep_gemm.set_num_sms(self.num_calc_sms)
        task_launcher.run_dispatch_overlap()

        deep_gemm.set_num_sms(0)
        task_launcher.run_compute()

        ############################ DISPATCH BACKWARD #############################

        dhidden_states, dtoken_probs, event = self.buffer.combine(
            drecv_x, handle, drecv_probs, async_finish=True, allocate_on_comm_stream=False,
            zip_done=zip_done)

        deep_gemm.set_num_sms(self.num_calc_sms)
        task_launcher.run_combine_overlap()

        del self.o1, self.unzipped_probs, self.zip_to_atomic
        self.input_grads = (dhidden_states, dtoken_probs)
        self.combine_done_event = event

        ################################## WGRAD ###################################

        paddle.base.core.nvprof_nvtx_push("wgrad")

        ks_cpu, m_start = [], [0]
        for n in tokens_per_expert:
            ks_cpu.append((n + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN)
            m_start.append(m_start[-1] + ks_cpu[-1])

        async_load = AsyncLoad()
        t = async_load(ks_cpu + m_start, dtype="int32")
        grouped_layout, m_start = t[:len(ks_cpu)], t[len(ks_cpu):]

        ordered_to_zip = paddle.empty([num_unzipped_tokens], dtype="int32")
        ordered_to_atomic = paddle.empty([num_unzipped_tokens], dtype="int32")
        deep_gemm.sort_map(zip_to_atomic_bwd, m_start, num_unzipped_tokens,
                           ordered_to_zip, ordered_to_atomic)

        # x 从 recv_x 中解压, 这里使用 gather 并非最优性能, 因为重复读了 recv_x 的某些行
        x_wgrad = deep_gemm.token_gather(self.recv_x, ordered_to_zip)
        del self.recv_x

        # do1/o2_bwd/do3 从反向 atomic 序的输入重排序为标准 unzip 序
        do1_wgrad = deep_gemm.token_gather(do1, ordered_to_atomic)
        o2_wgrad = deep_gemm.token_gather(o2_bwd, ordered_to_atomic)
        do3_wgrad = deep_gemm.token_gather(do3, ordered_to_atomic)

        bf16_weight_grad(x_wgrad, do1_wgrad, w_gateup, ks_cpu, grouped_layout)
        bf16_weight_grad(o2_wgrad, do3_wgrad, w_down, ks_cpu, grouped_layout)

        paddle.base.core.nvprof_nvtx_pop()

    def _backward_fp8(self, dout: Tensor):
        ########################### COMBINE BACKWARD ###########################

        dout_quant = quant_input(dout)

        (
            _, recv_token_indices, recv_token_probs, _, handle, event,
            do3, _, atomic_to_zip_bwd, zip_to_atomic_bwd, num_valid_topk, task_queue,
        ) = self.buffer.dispatch(
            dout_quant,
            topk_idx=self.token_indices,
            topk_weights=self.token_probs,  # 无用
            **self.dispatch_layout,
            async_finish=True,
            allocate_on_comm_stream=False,
            unzip_alignment=FP8_ALIGN,
            unzip_chunk_size=self.chunk_size,
        )

        del self.token_indices, self.token_probs, self.dispatch_layout

        do3_fp8, do3_scale = do3
        do3 = (do3_fp8, do3_scale.T)

        # 将 wgrad 的 recv_x requant 提前到 dispatch overlap, 因为一般 gemm 刚开始都在空等
        x_w = deep_gemm.requant_wgrad_input(
            self.recv_x[0], self.recv_x[1].T.contiguous().T, self.ordered_to_zip)
        del self.recv_x

        ############################ GEMM BACKWARD #############################

        w_gateup, w_down = self.w_gateup, self.w_down
        H, I = w_gateup.shape[1], w_down.shape[1]
        num_recv_tokens = len(recv_token_probs)
        num_unzipped_tokens = len(do3_fp8)

        w_gateup_n = get_quant_weight(w_gateup)
        w_down_n = get_quant_weight(w_down)

        do2 = paddle.empty([num_unzipped_tokens, I], dtype="bfloat16")
        dx = paddle.empty([num_unzipped_tokens, H], dtype="bfloat16")
        do1_fp8 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="float8_e4m3fn")
        do1_scale = paddle.empty([2 * I // QUANT_BLOCK_SIZE, num_unzipped_tokens], dtype="int32").T
        do1 = (do1_fp8, do1_scale)
        o2_bwd_fp8 = paddle.empty([num_unzipped_tokens, I], dtype="float8_e4m3fn")
        o2_bwd_scale = paddle.empty([I // QUANT_BLOCK_SIZE, num_unzipped_tokens], dtype="int32").T
        o2_bwd = (o2_bwd_fp8, o2_bwd_scale)
        drecv_x = paddle.empty([num_recv_tokens, H], dtype="bfloat16")
        drecv_probs = paddle.zeros_like(recv_token_probs)  # 无效位预先填0

        token_done = paddle.zeros([num_recv_tokens], dtype="int32")
        zip_done = paddle.zeros([num_recv_tokens], dtype="int32")

        funcs = [
            lambda task_idx: deep_gemm.fp8_chunk_gemm_nt(do3, w_down_n, do2, task_queue, task_idx),
            lambda task_idx: deep_gemm.chunk_weighted_swiglu_grad(
                self.o1, self.unzipped_probs, do2, o2_bwd_fp8, do1_fp8, drecv_probs,
                atomic_to_zip_bwd, self.zip_to_atomic, recv_token_indices, task_queue, task_idx,
                self.chunk_size, **self.activation,
                o2_bwd_scales=o2_bwd_scale, do1_scales=do1_scale),
            lambda task_idx: deep_gemm.fp8_chunk_gemm_nt(
                do1, w_gateup_n, dx, task_queue, task_idx),
            lambda task_idx: deep_gemm.chunk_zip(
                dx, drecv_x, atomic_to_zip_bwd, zip_to_atomic_bwd, recv_token_indices,
                num_valid_topk, token_done, zip_done, task_queue, task_idx, self.chunk_size),
        ]

        task_launcher = GroupedTaskLauncher(funcs, len(task_queue), event)

        deep_gemm.set_num_sms(self.num_calc_sms)
        task_launcher.run_dispatch_overlap()

        deep_gemm.set_num_sms(0)
        task_launcher.run_compute()

        ########################## DISPATCH BACKWARD ###########################

        dhidden_states, dtoken_probs, event = self.buffer.combine(
            drecv_x, handle, drecv_probs, async_finish=True, allocate_on_comm_stream=False)

        deep_gemm.set_num_sms(self.num_calc_sms)
        # NOTES: 反向不使用 combine overlap, 因为已经有计算量较大的 wgrad, 足以填满甚至超出 combine

        del self.o1, self.unzipped_probs, self.zip_to_atomic
        self.input_grads = (dhidden_states, dtoken_probs)
        self.combine_done_event = event

        ################################ WGRAD #################################

        paddle.base.core.nvprof_nvtx_push("wgrad")

        ordered_to_atomic = self.ordered_to_zip  # reuse forward map
        deep_gemm.sort_map(zip_to_atomic_bwd, self.m_start_gpu, self.m_start_wgrad[-1], None,
                           ordered_to_atomic, self.m_start_wgrad_gpu)

        do1_w = deep_gemm.requant_wgrad_input(*do1, ordered_to_atomic)
        o2_w = deep_gemm.requant_wgrad_input(*o2_bwd, ordered_to_atomic)
        do3_w = deep_gemm.requant_wgrad_input(*do3, ordered_to_atomic)

        fp8_weight_grad(x_w, do1_w, w_gateup, self.ks_cpu, self.grouped_layout)
        fp8_weight_grad(o2_w, do3_w, w_down, self.ks_cpu, self.grouped_layout)

        paddle.base.core.nvprof_nvtx_pop()

    def _backward_fp8_bf16_wgrad(self, dout: Tensor):
        ########################### COMBINE BACKWARD ###########################

        # 非 chunk 模式, 可以使用 cache mode, 发送 BF16
        with cached_notify_guard() as event:
            recv_dout, _, _, _, _, event = self.buffer.dispatch(
                dout, handle=self.handle, async_finish=True, previous_event=event,
                allocate_on_comm_stream=False)

        self.recv_dout = recv_dout
        self.dispatch_done_event = event

        recv_indices = self.recv_token_indices
        recv_probs = paddle.empty(recv_indices.shape, dtype="float32")  # placeholder
        self.m_indices = paddle.concat([
            paddle.full([self.ks_cpu[i]], i, dtype="int32")
            for i, n in enumerate(self.tokens_per_expert)
        ])

        # x 在 wgrad 时才用, 这里提前到 dispatch 以进行 overlap
        recv_x_dequant = dequant_input(*self.recv_x)
        del self.recv_x

        self.x, self.rowmap, _, _ = paddle.nn.functional.moe_permute(
            recv_x_dequant, None, recv_indices, recv_probs,
            padding_alignment=FP8_ALIGN,
            num_experts=len(self.w_gateup),
            tokens_per_expert=self.tokens_per_expert,
        )

    def _backward_fp8_bf16_wgrad_wait(self):
        w_gateup, w_down = self.w_gateup, self.w_down
        H, I = w_gateup.shape[1], w_down.shape[1]
        w_gateup_n = get_quant_weight(w_gateup)
        w_down_n = get_quant_weight(w_down)

        self.dispatch_done_event.current_stream_wait()
        del self.dispatch_done_event

        ############################ GEMM BACKWARD #############################

        deep_gemm.set_num_sms(0)

        do3, do3_fp8, do3_scale = deep_gemm.token_scatter(
            self.recv_dout, self.rowmap, self.tokens_per_expert_gpu, len(self.o1), FP8_ALIGN,
            quant=True)
        del self.recv_dout

        ########## DOWN_GRAD ###########

        recv_indices = self.recv_token_indices
        num_zipped_tokens = len(recv_indices)
        num_unzipped_tokens = self.m_start_wgrad[-1]

        do2 = paddle.empty([num_unzipped_tokens, I], dtype="bfloat16")
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (do3_fp8, do3_scale.T), w_down_n, do2, self.m_indices)
        del do3_fp8, do3_scale

        ######### SWIGLU_GRAD ##########

        do1, dprobs, o2, do1_fp8, do1_scale = deep_gemm.weighted_swiglu_grad(
            self.o1, self.unzipped_probs, do2, **self.activation, inplace=True, quant=True)
        del self.o1, do2, self.unzipped_probs

        bf16_weight_grad(self.x, do1, w_gateup, self.ks_cpu, self.grouped_layout)
        del self.x, do1

        ######### GATEUP_GRAD ##########

        dx = paddle.empty([num_unzipped_tokens, H], dtype="bfloat16")
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (do1_fp8, do1_scale.T), w_gateup_n, dx, self.m_indices)
        del do1_fp8, do1_scale

        drecv_x, drecv_probs = paddle.nn.functional.moe_unpermute(
            dx, self.rowmap, recv_indices, dprobs, total_zipped_tokens=num_zipped_tokens,
            num_experts=len(w_gateup))
        del dx, self.rowmap, dprobs

        ########################## DISPATCH BACKWARD ###########################

        dhidden_states, dtoken_probs, event = self.buffer.combine(
            drecv_x, self.handle, drecv_probs, async_finish=True, allocate_on_comm_stream=False)

        self.input_grads = (dhidden_states, dtoken_probs)
        self.combine_done_event = event

        ################################ WGRAD #################################

        paddle.base.core.nvprof_nvtx_push("wgrad")
        deep_gemm.set_num_sms(self.num_calc_sms)

        bf16_weight_grad(o2, do3, w_down, self.ks_cpu, self.grouped_layout)

        paddle.base.core.nvprof_nvtx_pop()

    def _backward_serial(self, dout: Tensor):
        ########################### COMBINE BACKWARD ###########################

        with cached_notify_guard() as event:
            recv_dout, _, _, _, _, event = self.buffer.dispatch(
                dout, handle=self.handle, async_finish=True, previous_event=event,
                allocate_on_comm_stream=False)

        w_gateup, w_down = self.w_gateup, self.w_down
        E, I, H = w_down.shape
        if self.fp8:
            w_gateup_n = get_quant_weight(w_gateup)
            w_down_n = get_quant_weight(w_down)

        # 先 unzip 再 dequant, 与 baseline 保持一致
        x, rowmap, _, scale = paddle.nn.functional.moe_permute(
            self.recv_x[0] if self.fp8 else self.recv_x,  # recv_x
            self.recv_x[1] if self.fp8 else None,  # recv_scale
            self.recv_token_indices,
            paddle.empty(self.recv_token_indices.shape, dtype="float32"),  # placeholder
            padding_alignment=FP8_ALIGN, num_experts=self.num_permute_experts,
            tokens_per_expert=self.tokens_per_expert, using_ue8m0_scale=True)
        del self.recv_x

        self._clear_replica_grads()

        event.current_stream_wait()

        ############################ GEMM BACKWARD #############################

        tasks = self._sync_replica_weights((w_gateup_n, w_down_n) if self.fp8 else None)
        prioritize_comm_stream()

        do3, do3_fp8, do3_scale = deep_gemm.token_scatter(
            recv_dout, rowmap, self.tokens_per_expert_gpu, len(self.o1), FP8_ALIGN,
            quant=self.fp8 is not None)
        assert self.fp8 or (do3_fp8 is None and do3_scale is None)
        del recv_dout

        for task in tasks:
            task.wait()

        M = sum(self.ks_cpu[:E])

        if self.fp8:
            do2 = paddle.empty([len(do3), I], dtype="bfloat16")
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (do3_fp8[:M], do3_scale[:, :M].contiguous().T), w_down_n, do2[:M], self.m_indices)
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (do3_fp8[M:], do3_scale[:, M:].contiguous().T), tuple(self.w_down_replica),
                do2[M:], self.m_indices_replica)

            do1, dprobs, o2, do1_fp8, do1_scale = deep_gemm.weighted_swiglu_grad(
                self.o1, self.unzipped_probs, do2, **self.activation, inplace=True, quant=True)
            del self.o1, do2, self.unzipped_probs

            bf16_weight_grad(o2[:M], do3[:M], w_down, self.ks_cpu[:E], self.grouped_layout[:E])
            bf16_weight_grad(o2[M:], do3[M:], self.w_down_replica, self.ks_cpu[E:],
                             self.grouped_layout[E:])
            del o2, do3

            x_dequant = dequant_input(x, scale)
            del x, scale
            bf16_weight_grad(x_dequant[:M], do1[:M], w_gateup, self.ks_cpu[:E],
                             self.grouped_layout[:E])
            bf16_weight_grad(x_dequant[M:], do1[M:], self.w_gateup_replica, self.ks_cpu[E:],
                             self.grouped_layout[E:])
            del x_dequant, do1

            tasks, (w_gateup_replica_grads, w_down_replica_grads) = self._sync_replica_grads()

            dx = paddle.empty([len(do1_fp8), H], dtype="bfloat16")
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (do1_fp8[:M], do1_scale[:, :M].contiguous().T), w_gateup_n, dx[:M], self.m_indices)
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (do1_fp8[M:], do1_scale[:, M:].contiguous().T), tuple(self.w_gateup_replica),
                dx[M:], self.m_indices_replica)
            del do1_fp8, do1_scale
        else:
            do2 = paddle.empty([len(do3), I], dtype="bfloat16")
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(do3[:M], w_down, do2[:M], self.m_indices)
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                do3[M:], self.w_down_replica, do2[M:], self.m_indices_replica)

            do1, dprobs, o2, _, _ = deep_gemm.weighted_swiglu_grad(
                self.o1, self.unzipped_probs, do2, **self.activation, inplace=True)
            del self.o1, do2, self.unzipped_probs

            bf16_weight_grad(x[:M], do1[:M], w_gateup, self.ks_cpu[:E], self.grouped_layout[:E])
            bf16_weight_grad(x[M:], do1[M:], self.w_gateup_replica, self.ks_cpu[E:],
                             self.grouped_layout[E:])
            bf16_weight_grad(o2[:M], do3[:M], w_down, self.ks_cpu[:E], self.grouped_layout[:E])
            bf16_weight_grad(o2[M:], do3[M:], self.w_down_replica, self.ks_cpu[E:],
                             self.grouped_layout[E:])
            del x, o2, do3

            tasks, (w_gateup_replica_grads, w_down_replica_grads) = self._sync_replica_grads()

            dx = paddle.empty([len(do1), H], dtype="bfloat16")
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(do1[:M], w_gateup, dx[:M], self.m_indices)
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                do1[M:], self.w_gateup_replica, dx[M:], self.m_indices_replica)
            del do1

        deep_gemm.set_num_sms(0)

        ########################## DISPATCH BACKWARD ###########################

        drecv_x, drecv_probs = paddle.nn.functional.moe_unpermute(
            dx, rowmap, self.recv_token_indices, dprobs,
            total_zipped_tokens=len(self.recv_token_indices), num_experts=self.num_permute_experts)
        del dx, dprobs

        for task in tasks:
            task.wait()

        with cached_notify_guard() as event:
            dhidden_states, dtoken_probs, event = self.buffer.combine(
                drecv_x, self.handle, drecv_probs, async_finish=True, previous_event=event,
                allocate_on_comm_stream=False)

        paddle.base.core.nvprof_nvtx_push("accumulate_wgrad")
        w_gateup_grad = ensure_wgrad(w_gateup)
        w_down_grad = ensure_wgrad(w_down)
        for expert_id in range(E):
            accumulate_wgrad_(w_gateup_grad[expert_id], w_gateup_replica_grads[expert_id])
            accumulate_wgrad_(w_down_grad[expert_id], w_down_replica_grads[expert_id])
        paddle.base.core.nvprof_nvtx_pop()

        self.input_grads = (dhidden_states, dtoken_probs)
        self.combine_done_event = event

    def backward_wait(self) -> [Tensor, Tensor]:
        """Wait for combine to finish and return the combined result."""
        if self.enable_overlap:
            if self.fp8 and not self.fp8_wgrad:
                self._backward_fp8_bf16_wgrad_wait()
        paddle.base.core.nvprof_nvtx_pop()

        input_grads = self.input_grads
        self.combine_done_event.current_stream_wait()
        del self.input_grads, self.combine_done_event
        del self.w_gateup_replica, self.w_down_replica
        return input_grads


def ensure_wgrad(w: Tensor) -> Tensor:
    attr = "main_grad" if hasattr(w, "main_grad") else "grad"
    grad = getattr(w, attr)
    if grad is None:
        grad = paddle.zeros(w.shape, dtype="float32")
        setattr(w, attr, grad)
    return grad


def bf16_weight_grad(x, dy, weight, ks_cpu, grouped_layout):
    grad = ensure_wgrad(weight)
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(x, dy, grad, ks_cpu, grouped_layout, grad)


def fp8_weight_grad(x, dy, weight, ks_cpu, grouped_layout):
    grad = ensure_wgrad(weight)
    deep_gemm.k_grouped_fp8_gemm_tn_contiguous(x, dy, grad, ks_cpu, grouped_layout, grad)
