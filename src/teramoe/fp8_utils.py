import logging
import functools
from types import SimpleNamespace
from contextlib import contextmanager

import paddle
import paddle.distributed as dist
from paddle import Tensor
from paddle.distributed.communication.group import Group

from . import deep_ep, deep_gemm
from .async_utils import AsyncLoad, async_offload, get_profile_timers, started
from .task_utils import GroupedTaskLauncher, make_batch_task_queue
from .placement import update_placement, prioritize_comm_stream

FP8_ALIGN = 128
QUANT_BLOCK_SIZE = 512

logger = logging.getLogger(__name__)

_global_previous_event = None
_global_barrier_ep_event = None


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


@functools.lru_cache(None)
def warning_once(*args, **kwargs):
    logger.warning(*args, **kwargs)


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
        num_replicas: int,
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

        self.enable_overlap = chunk_size > 0 and self.buffer.runtime.get_num_rdma_ranks() > 1
        self.avg_tokens_per_card = token_indices.size  # seqlen * topk
        if self.enable_overlap:
            self.avg_num_tasks = int(self.avg_tokens_per_card / chunk_size + len(w_gateup) / 2)
        self.barrier_ep_event = None

        timers = get_profile_timers()
        self.dispatch_timer = timers("dispatch", use_event=True)
        self.compute_timer = timers("fusion_mlp", use_event=True)
        self.combine_timer = timers("combine", use_event=True)

        # UltraEP
        self.num_replicas = num_replicas
        self.num_dispatch_experts = num_experts + self.num_replicas * group.world_size
        self.num_permute_experts = len(w_gateup) + self.num_replicas

        # Fallback check
        if self.enable_overlap and self.num_permute_experts > 32:
            warning_once("TeraMoE: overlap mode doesn't support more than 32 experts per rank. "
                         "Fallback to serial mode.")
            self.enable_overlap = False
        if num_replicas > 0 and self.enable_overlap and not self.fp8:
            warning_once("TeraMoE: overlap mode doesn't support UltraEP with bf16. "
                         "Fallback to serial mode.")
            self.enable_overlap = False
        if num_replicas > 0 and self.fp8 and self.fp8_wgrad:
            warning_once("TeraMoE: UltraEP doesn't support fp8_wgrad. Fallback to bf16_wgrad.")
            self.fp8_wgrad = False

    def _prepare_dispatch(self):
        # CPU should wait at the point where all ranks finished all_reduce
        global _global_barrier_ep_event
        if _global_barrier_ep_event is None:
            _global_barrier_ep_event = paddle.cuda.Event()
        self.barrier_ep_event = _global_barrier_ep_event

        # Summarize token statics. Need global all_reduce if not done in update_placement
        if self.num_replicas > 0:
            self.barrier_ep_event.record()
            max_tokens_per_card = self.global_tokens_per_expert.reshape(
                [self.group.nranks, -1]).sum(axis=1, dtype="int32").max()

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

        self.dispatch_layout = dict(
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            is_token_in_rank=is_token_in_rank,
        )

        if not self.num_replicas > 0:
            global_tokens_per_card = num_tokens_per_expert.reshape(
                [self.group.nranks, -1]).sum(axis=1, dtype="int32")
            dist.all_reduce(global_tokens_per_card, group=self.group)
            self.barrier_ep_event.record()
            max_tokens_per_card = global_tokens_per_card.max()

        self.max_tokens_per_card = async_offload(max_tokens_per_card)

    def _dispatch(self, x: Tensor, chunk: bool = False):
        if not self.barrier_async and self.barrier_ep_event is not None:
            self.barrier_ep_event.synchronize()
            self.barrier_ep_event = None

        chunk_args = dict(
            unzip_alignment=FP8_ALIGN,
            unzip_chunk_size=self.chunk_size,
        ) if chunk else {}

        (
            recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle, event,
            *chunk_outs
        ) = self.buffer.dispatch(
            x,
            topk_idx=self.token_indices,
            topk_weights=self.token_probs,
            **self.dispatch_layout,
            async_finish=True,
            allocate_on_comm_stream=False,
            **chunk_args,
        )

        self.handle = handle
        self.tokens_per_expert = tokens_per_expert

        return recv_x, recv_token_indices, recv_token_probs, event, *chunk_outs

    def _weight_sync(self, weights, num_sms=16):
        if self.fp8:
            (w1, w1_scale), (w2, w2_scale) = weights
            weights = (w1, w1_scale.mT, w2, w2_scale.mT)
            _, H, I = w2.shape
        else:
            _, H, I = weights[1].shape

        paddle.base.core.nvprof_nvtx_push("weight_sync")
        out, event = self.buffer.weight_sync(
            self.physical_to_logical_map, weights, self.num_replicas, H, I, num_sms,
            async_finish=True)
        paddle.base.core.nvprof_nvtx_pop()

        if self.fp8:
            w1, w1_scale, w2, w2_scale = out
            out = ((w1, w1_scale.mT), (w2, w2_scale.mT))
        return out, event

    def _get_replica_grads(self):
        _, I, H = self.w_down.shape
        w1_grad, w2_grad = self.buffer.get_replica_grads(self.num_replicas, H, I)

        paddle.base.core.nvprof_nvtx_push("grad_reset")
        w1_grad.zero_()
        w2_grad.zero_()
        paddle.base.core.nvprof_nvtx_pop()

        w1 = SimpleNamespace(main_grad=w1_grad)
        w2 = SimpleNamespace(main_grad=w2_grad)
        return w1, w2

    def _grad_reduce(self, num_sms=8):
        _, I, H = self.w_down.shape
        out = (ensure_wgrad(self.w_gateup), ensure_wgrad(self.w_down))

        paddle.base.core.nvprof_nvtx_push("grad_reduce")
        event = self.buffer.grad_reduce(
            self.physical_to_logical_map, out, self.num_replicas, H, I, num_sms, async_finish=True)
        paddle.base.core.nvprof_nvtx_pop()

        deep_gemm.set_num_sms(deep_gemm.get_num_sms() - num_sms)

        return event

    def forward(self):
        """Forward dispatch, compute and combine.

        This function has no input/output, as the inputs are set at node initialization,
        and the output is hold util the user calls forward_wait.
        """
        # 本次 dispatch 必须在上一次 combine 结束之后才能调用, 避免 allocator 频繁查询 event
        if _global_previous_event is not None:
            _global_previous_event.synchronize()

        paddle.base.core.nvprof_nvtx_push("teramoe_forward")
        self.dispatch_timer.start()

        if self.num_replicas > 0:
            paddle.base.core.nvprof_nvtx_push("update_placement")
            self.token_probs, self.token_indices, self.physical_to_logical_map, expert_loads = (
                update_placement(self.group, self.token_probs, self.token_indices,
                                 self.num_experts, self.num_replicas))
            self.global_tokens_per_expert = expert_loads
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

        self._prepare_dispatch()

        (
            recv_x, recv_token_indices, recv_token_probs, event, unzipped_tokens, unzipped_probs,
            atomic_to_zip, zip_to_atomic, num_valid_topk, task_queue
        ) = self._dispatch(self.hidden_states, chunk=True)

        del self.hidden_states
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

        with cached_notify_guard() as event:
            out, _, event = self.buffer.combine(
                zipped_out, self.handle, async_finish=True, previous_event=event,
                allocate_on_comm_stream=False, zip_done=zip_done)

            deep_gemm.set_num_sms(self.num_calc_sms)
            self.combine_timer.start()
            task_launcher.run_combine_overlap()

        self.o1 = o1
        self.out = out
        self.combine_done_event = event

        ################################ WGRAD #################################

        ks_cpu, m_start = [], [0]
        for n in self.tokens_per_expert:
            ks_cpu.append((n + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN)
            m_start.append(m_start[-1] + ks_cpu[-1])

        async_load = AsyncLoad()
        t = async_load(ks_cpu + m_start, dtype="int32")
        self.grouped_layout, self.m_start = t[:len(ks_cpu)], t[len(ks_cpu):]
        self.ks_cpu = ks_cpu

        # 提前为 wgrad 恢复顺序
        ordered_to_atomic = paddle.empty([num_unzipped_tokens], dtype="int32")
        deep_gemm.sort_map(zip_to_atomic, self.m_start, num_unzipped_tokens, None,
                           ordered_to_atomic)
        self.x = deep_gemm.token_gather(unzipped_tokens, ordered_to_atomic)

    def _forward_fp8(self):
        ########################### DISPATCH FORWARD ###########################

        self._prepare_dispatch()

        w_gateup, w_down = self.w_gateup, self.w_down
        H, I = w_gateup.shape[1], w_down.shape[1]
        w_gateup_t = get_quant_weight(w_gateup, transpose=True)
        w_down_t = get_quant_weight(w_down, transpose=True)

        REPLICA = self.num_replicas > 0
        if REPLICA:
            # weight_sync is on the same stream with dispatch and has nothing to overlap.
            # Use more SMs to make it as fast as possible
            (w_gateup_replica, w_down_replica), _ = self._weight_sync(
                (w_gateup_t, w_down_t), self.buffer.num_sms)
            prioritize_comm_stream()
        else:
            w_gateup_replica, w_down_replica = None, None

        hidden_states = quant_input(self.hidden_states)
        del self.hidden_states

        ########## DISPATCH ###########

        (
            recv_x, recv_token_indices, recv_token_probs, event, unzipped_tokens, unzipped_probs,
            atomic_to_zip, zip_to_atomic, num_valid_topk, task_queue
        ) = self._dispatch(hidden_states, chunk=True)

        unzipped_tokens, unzipped_scale = unzipped_tokens
        unzipped_x = (unzipped_tokens, unzipped_scale.T)

        del hidden_states
        self.unzipped_x = unzipped_x
        self.unzipped_probs = unzipped_probs
        self.zip_to_atomic = zip_to_atomic
        if not self.fp8_wgrad:
            self.recv_token_indices = recv_token_indices

        ############################# GEMM FORWARD #############################

        num_recv_tokens = len(recv_x[0])
        num_unzipped_tokens = len(unzipped_tokens)

        o1 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="bfloat16")
        o2_fp8 = paddle.empty([num_unzipped_tokens, I], dtype="float8_e4m3fn")
        o2_scale = paddle.empty([I // QUANT_BLOCK_SIZE, num_unzipped_tokens], dtype="int32").T
        o3 = paddle.empty([num_unzipped_tokens, H], dtype="bfloat16")
        zipped_out = paddle.empty([num_recv_tokens, H], dtype="bfloat16")

        token_done = paddle.zeros([num_recv_tokens], dtype="int32")
        zip_done = paddle.zeros([num_recv_tokens], dtype="int32")

        funcs = [
            lambda task_idx: deep_gemm.fp8_chunk_gemm_nt(
                unzipped_x, w_gateup_t, o1, task_queue, task_idx, w_gateup_replica),
            lambda task_idx: deep_gemm.chunk_weighted_swiglu(
                o1, unzipped_probs, o2_fp8, task_queue, task_idx, self.chunk_size,
                **self.activation, o2_scales=o2_scale),
            lambda task_idx: deep_gemm.fp8_chunk_gemm_nt(
                (o2_fp8, o2_scale), w_down_t, o3, task_queue, task_idx, w_down_replica),
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
                zipped_out, self.handle, async_finish=True, previous_event=event,
                allocate_on_comm_stream=False, zip_done=zip_done)

            deep_gemm.set_num_sms(self.num_calc_sms)
            self.combine_timer.start()
            task_launcher.run_combine_overlap()

        self.o1 = o1
        self.out = out
        self.combine_done_event = event

        ################################ WGRAD #################################

        # 各专家 token 数重新向 512 对齐
        ks_cpu, m_start, m_start_wgrad = [], [0], [0]
        wgrad_align = QUANT_BLOCK_SIZE if self.fp8_wgrad else FP8_ALIGN
        for n in self.tokens_per_expert:
            ks_cpu.append((n + wgrad_align - 1) // wgrad_align * wgrad_align)
            m_start.append(m_start[-1] + (n + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN)
            m_start_wgrad.append(m_start_wgrad[-1] + ks_cpu[-1])
        self.ks_cpu = ks_cpu
        self.m_start_wgrad = m_start_wgrad

        async_load = AsyncLoad()
        t = async_load(ks_cpu + m_start + m_start_wgrad + self.tokens_per_expert, dtype="int32")
        self.grouped_layout = t[:len(ks_cpu)]
        self.m_start_gpu = t[len(ks_cpu) : len(ks_cpu) + len(m_start)]
        self.m_start_wgrad_gpu = t[len(ks_cpu) + len(m_start) : len(ks_cpu) + len(m_start) * 2]
        self.tokens_per_expert_gpu = t[len(ks_cpu) + len(m_start) * 2:]

        self.ordered_to_zip = paddle.empty([m_start_wgrad[-1]], dtype="int32")
        self.ordered_to_atomic = paddle.empty_like(self.ordered_to_zip)
        deep_gemm.sort_map(zip_to_atomic, self.m_start_gpu, m_start_wgrad[-1],
                           self.ordered_to_zip, self.ordered_to_atomic, self.m_start_wgrad_gpu)

        # 提前为 bf16_wgrad 恢复顺序
        if not self.fp8_wgrad:
            self.o1 = deep_gemm.token_gather(self.o1, self.ordered_to_atomic)
            self.unzipped_probs = deep_gemm.token_gather(
                self.unzipped_probs.unsqueeze(1), self.ordered_to_atomic).squeeze(1)
            self.recv_token_indices = self.recv_token_indices.cast("int32")

    def _forward_serial(self):
        """Simple fallback for intranode."""
        ########################### DISPATCH FORWARD ###########################

        self._prepare_dispatch()

        hidden_states = quant_input(self.hidden_states) if self.fp8 else self.hidden_states
        del self.hidden_states

        recv_x, recv_token_indices, recv_token_probs, event = self._dispatch(hidden_states)

        ########## METADATA ###########

        w_gateup, w_down = self.w_gateup, self.w_down
        E, I, H = w_down.shape
        if self.fp8:
            w_gateup_t = get_quant_weight(w_gateup, transpose=True)
            w_down_t = get_quant_weight(w_down, transpose=True)

        ks_cpu = []
        for i, n in enumerate(self.tokens_per_expert):
            ks_cpu.append((n + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN)
        async_load = AsyncLoad()
        self.ks_cpu = ks_cpu
        t = async_load(ks_cpu + self.tokens_per_expert, dtype="int32")
        self.grouped_layout = t[:len(ks_cpu)]
        self.tokens_per_expert_gpu = t[len(ks_cpu):]
        self.m_indices = deep_gemm.gen_m_indices(
            self.tokens_per_expert_gpu, sum(ks_cpu), FP8_ALIGN)

        M = sum(self.ks_cpu[:E])
        REPLICA = self.num_replicas > 0
        if REPLICA:
            self.m_indices, self.m_indices_replica = self.m_indices[:M], self.m_indices[M:] - E

        task_queue_np = make_batch_task_queue(self.tokens_per_expert)
        async_load.wait()

        event.current_stream_wait()
        self.dispatch_timer.stop()
        self.compute_timer.start()

        del hidden_states
        self.recv_x = recv_x
        recv_token_indices = recv_token_indices.cast("int32")
        self.recv_token_indices = recv_token_indices

        ############################# GEMM FORWARD #############################

        if REPLICA:
            (w_gateup_replica, w_down_replica), event = self._weight_sync(
                (w_gateup_t, w_down_t) if self.fp8 else (w_gateup, w_down))

        if self.fp8:
            recv_x, recv_scale = recv_x
        else:
            recv_scale = None

        with paddle.amp.auto_cast(False):
            x, rowmap, unzipped_probs, x_scale = paddle.nn.functional.moe_permute(
                recv_x, recv_scale, recv_token_indices, recv_token_probs,
                padding_alignment=FP8_ALIGN, num_experts=self.num_permute_experts,
                tokens_per_expert=self.tokens_per_expert, using_ue8m0_scale=True)

        event.current_stream_wait() if REPLICA else ()
        num_unzipped_tokens = len(x)

        if self.fp8:
            o1 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="bfloat16")
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (x[:M], x_scale[:M].T.contiguous().T), w_gateup_t, o1[:M], self.m_indices)
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (x[M:], x_scale[M:].T.contiguous().T), w_gateup_replica, o1[M:],
                self.m_indices_replica) if REPLICA else ()
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
                (o2_fp8[M:], o2_scale[M:].T.contiguous().T), w_down_replica, o3[M:],
                self.m_indices_replica) if REPLICA else ()
            del o2_fp8, o2_scale
        else:
            o1 = paddle.empty([num_unzipped_tokens, 2 * I], dtype="bfloat16")
            deep_gemm.m_grouped_bf16_gemm_nn_contiguous(x[:M], w_gateup, o1[:M], self.m_indices)
            deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
                x[M:], w_gateup_replica, o1[M:], self.m_indices_replica) if REPLICA else ()
            del x

            o2 = paddle.empty([num_unzipped_tokens, I], dtype="bfloat16")
            deep_gemm.chunk_weighted_swiglu(
                o1, unzipped_probs, o2, async_load(task_queue_np), task_idx=-1,
                chunk_size=FP8_ALIGN, **self.activation)

            o3 = paddle.empty([num_unzipped_tokens, H], dtype="bfloat16")
            deep_gemm.m_grouped_bf16_gemm_nn_contiguous(o2[:M], w_down, o3[:M], self.m_indices)
            deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
                o2[M:], w_down_replica, o3[M:], self.m_indices_replica) if REPLICA else ()
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
                zipped_out, self.handle, async_finish=True, previous_event=event,
                allocate_on_comm_stream=False)

        self.unzipped_probs = unzipped_probs
        self.o1 = o1
        self.out = out
        self.combine_done_event = event

    def forward_wait(self) -> Tensor:
        """Wait for combine to finish and return the combined result."""
        global _global_previous_event
        _global_previous_event = self.combine_done_event
        out = self.out
        self.combine_done_event.current_stream_wait()
        self.combine_timer.stop()
        self.tokens_per_card_max_mean_ratio = (
            self.max_tokens_per_card.item() / self.avg_tokens_per_card)
        del self.out, self.combine_done_event, self.max_tokens_per_card
        return out

    def backward(self, dout: Tensor):
        """Backward dispatch, compute, combine and wgrad.

        This function accepts one input, and has no output in the same way as forward.
        """
        paddle.base.core.nvprof_nvtx_push("teramoe_backward")
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
            _, recv_token_indices, recv_token_probs, event, do3, _, atomic_to_zip_bwd,
            zip_to_atomic_bwd, num_valid_topk, task_queue
        ) = self._dispatch(dout, chunk=True)

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
            drecv_x, self.handle, drecv_probs, async_finish=True, allocate_on_comm_stream=False,
            zip_done=zip_done)

        deep_gemm.set_num_sms(self.num_calc_sms)
        task_launcher.run_combine_overlap()

        del self.o1, self.unzipped_probs, self.zip_to_atomic
        self.input_grads = (dhidden_states, dtoken_probs)
        self.combine_done_event = event

        ################################## WGRAD ###################################

        paddle.base.core.nvprof_nvtx_push("wgrad")

        ordered_to_atomic = paddle.empty([num_unzipped_tokens], dtype="int32")
        deep_gemm.sort_map(zip_to_atomic_bwd, self.m_start, num_unzipped_tokens, None,
                           ordered_to_atomic)

        do1_wgrad = deep_gemm.token_gather(do1, ordered_to_atomic)
        o2_wgrad = deep_gemm.token_gather(o2_bwd, ordered_to_atomic)
        do3_wgrad = deep_gemm.token_gather(do3, ordered_to_atomic)

        bf16_weight_grad(self.x, do1_wgrad, w_gateup, self.ks_cpu, self.grouped_layout)
        bf16_weight_grad(o2_wgrad, do3_wgrad, w_down, self.ks_cpu, self.grouped_layout)

        paddle.base.core.nvprof_nvtx_pop()

    def _backward_fp8(self, dout: Tensor):
        ########################### COMBINE BACKWARD ###########################

        dout_quant = quant_input(dout)

        (
            _, recv_token_indices, recv_token_probs, event, do3, _, atomic_to_zip_bwd,
            zip_to_atomic_bwd, num_valid_topk, task_queue
        ) = self._dispatch(dout_quant, chunk=True)

        del self.token_indices, self.token_probs, self.dispatch_layout

        do3_fp8, do3_scale = do3
        do3 = (do3_fp8, do3_scale.T)

        # 将 wgrad 的 x requant 提前到 dispatch overlap, 因为一般 gemm 刚开始都在空等
        x_w = deep_gemm.requant_wgrad_input(*self.unzipped_x, self.ordered_to_atomic)
        del self.unzipped_x

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
            drecv_x, self.handle, drecv_probs, async_finish=True, allocate_on_comm_stream=False)

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

        self.m_indices = deep_gemm.gen_m_indices(
            self.tokens_per_expert_gpu, self.m_start_wgrad[-1], FP8_ALIGN)

        self.x = deep_gemm.token_gather(
            self.unzipped_x[0], self.ordered_to_atomic, self.unzipped_x[1])
        del self.unzipped_x

        self.rowmap = deep_gemm.gen_rowmap(self.recv_token_indices, self.tokens_per_expert_gpu)

    def _backward_fp8_bf16_wgrad_wait(self):
        w_gateup, w_down = self.w_gateup, self.w_down
        E, I, H = w_down.shape
        w_gateup_n = get_quant_weight(w_gateup)
        w_down_n = get_quant_weight(w_down)

        REPLICA = self.num_replicas > 0
        M = sum(self.ks_cpu[:E])
        if REPLICA:
            w_gateup_replica_grad, w_down_replica_grad = self._get_replica_grads()

        self.dispatch_done_event.current_stream_wait()
        del self.dispatch_done_event

        ############################ GEMM BACKWARD #############################

        if REPLICA:
            (w_gateup_replica, w_down_replica), event = self._weight_sync((w_gateup_n, w_down_n))
            self.m_indices, self.m_indices_replica = self.m_indices[:M], self.m_indices[M:] - E

        deep_gemm.set_num_sms(0)

        do3, do3_fp8, do3_scale = deep_gemm.token_scatter(
            self.recv_dout, self.rowmap, self.tokens_per_expert_gpu, len(self.o1), FP8_ALIGN,
            quant=True)
        del self.recv_dout

        event.current_stream_wait() if REPLICA else ()

        ########## DOWN_GRAD ###########

        recv_indices = self.recv_token_indices
        num_zipped_tokens = len(recv_indices)
        num_unzipped_tokens = self.m_start_wgrad[-1]

        do2 = paddle.empty([num_unzipped_tokens, I], dtype="bfloat16")
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (do3_fp8[:M], do3_scale[:, :M].contiguous().T), w_down_n, do2[:M], self.m_indices)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (do3_fp8[M:], do3_scale[:, M:].contiguous().T), w_down_replica, do2[M:],
            self.m_indices_replica) if REPLICA else ()
        del do3_fp8, do3_scale

        ######### SWIGLU_GRAD ##########

        do1, dprobs, o2, do1_fp8, do1_scale = deep_gemm.weighted_swiglu_grad(
            self.o1, self.unzipped_probs, do2, **self.activation, inplace=True, quant=True)
        del self.o1, do2, self.unzipped_probs

        if not REPLICA:
            bf16_weight_grad(self.x, do1, w_gateup, self.ks_cpu, self.grouped_layout)
            bf16_weight_grad(o2, do3, w_down, self.ks_cpu, self.grouped_layout)
            del self.x, do1, o2, do3

        ######### GATEUP_GRAD ##########

        dx = paddle.empty([num_unzipped_tokens, H], dtype="bfloat16")
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (do1_fp8[:M], do1_scale[:, :M].contiguous().T), w_gateup_n, dx[:M], self.m_indices)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (do1_fp8[M:], do1_scale[:, M:].contiguous().T), w_gateup_replica, dx[M:],
            self.m_indices_replica) if REPLICA else ()
        del do1_fp8, do1_scale

        drecv_x, drecv_probs = paddle.nn.functional.moe_unpermute(
            dx, self.rowmap, recv_indices, dprobs, total_zipped_tokens=num_zipped_tokens,
            num_experts=self.num_permute_experts)
        del dx, self.rowmap, dprobs

        ########################## DISPATCH BACKWARD ###########################

        dhidden_states, dtoken_probs, event = self.buffer.combine(
            drecv_x, self.handle, drecv_probs, async_finish=True, allocate_on_comm_stream=False)

        self.input_grads = (dhidden_states, dtoken_probs)
        self.combine_done_event = event

        ################################ WGRAD #################################

        # Use wgrad overlap only when UltraEP balancing is enabled. This is because wgrad is
        # usually too slow to overlap and increases much peak memory in unbalanced cases
        if REPLICA:
            paddle.base.core.nvprof_nvtx_push("wgrad")
            deep_gemm.set_num_sms(self.num_calc_sms)
            bf16_weight_grad(self.x[:M], do1[:M], w_gateup, self.ks_cpu[:E],
                             self.grouped_layout[:E])
            bf16_weight_grad(self.x[M:], do1[M:], w_gateup_replica_grad, self.ks_cpu[E:],
                             self.grouped_layout[E:]) if REPLICA else ()
            bf16_weight_grad(o2[:M], do3[:M], w_down, self.ks_cpu[:E], self.grouped_layout[:E])
            bf16_weight_grad(o2[M:], do3[M:], w_down_replica_grad, self.ks_cpu[E:],
                             self.grouped_layout[E:]) if REPLICA else ()
            paddle.base.core.nvprof_nvtx_pop()

            self.combine_done_event = self._grad_reduce(self.buffer.num_sms)

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

        REPLICA = self.num_replicas > 0
        if REPLICA:
            w_gateup_replica_grad, w_down_replica_grad = self._get_replica_grads()

        event.current_stream_wait()

        ############################ GEMM BACKWARD #############################

        if REPLICA:
            (w_gateup_replica, w_down_replica), event = self._weight_sync(
                (w_gateup_n, w_down_n) if self.fp8 else (w_gateup, w_down))
            prioritize_comm_stream()

        do3, do3_fp8, do3_scale = deep_gemm.token_scatter(
            recv_dout, rowmap, self.tokens_per_expert_gpu, len(self.o1), FP8_ALIGN,
            quant=self.fp8 is not None)
        assert self.fp8 or (do3_fp8 is None and do3_scale is None)
        del recv_dout

        event.current_stream_wait() if REPLICA else ()
        M = sum(self.ks_cpu[:E])

        if self.fp8:
            do2 = paddle.empty([len(do3), I], dtype="bfloat16")
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (do3_fp8[:M], do3_scale[:, :M].contiguous().T), w_down_n, do2[:M], self.m_indices)
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (do3_fp8[M:], do3_scale[:, M:].contiguous().T), w_down_replica, do2[M:],
                self.m_indices_replica) if REPLICA else ()
            del do3_fp8, do3_scale

            do1, dprobs, o2, do1_fp8, do1_scale = deep_gemm.weighted_swiglu_grad(
                self.o1, self.unzipped_probs, do2, **self.activation, inplace=True, quant=True)
            del self.o1, do2, self.unzipped_probs

            bf16_weight_grad(o2[:M], do3[:M], w_down, self.ks_cpu[:E], self.grouped_layout[:E])
            bf16_weight_grad(o2[M:], do3[M:], w_down_replica_grad, self.ks_cpu[E:],
                             self.grouped_layout[E:]) if REPLICA else ()
            del o2, do3

            x_dequant = dequant_input(x, scale)
            del x, scale
            bf16_weight_grad(x_dequant[:M], do1[:M], w_gateup, self.ks_cpu[:E],
                             self.grouped_layout[:E])
            bf16_weight_grad(x_dequant[M:], do1[M:], w_gateup_replica_grad, self.ks_cpu[E:],
                             self.grouped_layout[E:]) if REPLICA else ()
            del x_dequant, do1

            event = self._grad_reduce() if REPLICA else None

            dx = paddle.empty([len(do1_fp8), H], dtype="bfloat16")
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (do1_fp8[:M], do1_scale[:, :M].contiguous().T), w_gateup_n, dx[:M], self.m_indices)
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (do1_fp8[M:], do1_scale[:, M:].contiguous().T), w_gateup_replica, dx[M:],
                self.m_indices_replica) if REPLICA else ()
            del do1_fp8, do1_scale
        else:
            do2 = paddle.empty([len(do3), I], dtype="bfloat16")
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(do3[:M], w_down, do2[:M], self.m_indices)
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                do3[M:], w_down_replica, do2[M:], self.m_indices_replica) if REPLICA else ()

            do1, dprobs, o2, _, _ = deep_gemm.weighted_swiglu_grad(
                self.o1, self.unzipped_probs, do2, **self.activation, inplace=True)
            del self.o1, do2, self.unzipped_probs

            bf16_weight_grad(x[:M], do1[:M], w_gateup, self.ks_cpu[:E], self.grouped_layout[:E])
            bf16_weight_grad(x[M:], do1[M:], w_gateup_replica_grad, self.ks_cpu[E:],
                             self.grouped_layout[E:]) if REPLICA else ()
            bf16_weight_grad(o2[:M], do3[:M], w_down, self.ks_cpu[:E], self.grouped_layout[:E])
            bf16_weight_grad(o2[M:], do3[M:], w_down_replica_grad, self.ks_cpu[E:],
                             self.grouped_layout[E:]) if REPLICA else ()
            del x, o2, do3

            event = self._grad_reduce() if REPLICA else None

            dx = paddle.empty([len(do1), H], dtype="bfloat16")
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(do1[:M], w_gateup, dx[:M], self.m_indices)
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                do1[M:], w_gateup_replica, dx[M:], self.m_indices_replica) if REPLICA else ()
            del do1

        ########################## DISPATCH BACKWARD ###########################

        drecv_x, drecv_probs = paddle.nn.functional.moe_unpermute(
            dx, rowmap, self.recv_token_indices, dprobs,
            total_zipped_tokens=len(self.recv_token_indices), num_experts=self.num_permute_experts)
        del dx, dprobs

        event.current_stream_wait() if REPLICA else ()
        deep_gemm.set_num_sms(0)

        with cached_notify_guard() as event:
            dhidden_states, dtoken_probs, event = self.buffer.combine(
                drecv_x, self.handle, drecv_probs, async_finish=True, previous_event=event,
                allocate_on_comm_stream=False)

        self.input_grads = (dhidden_states, dtoken_probs)
        self.combine_done_event = event

    def backward_wait(self) -> [Tensor, Tensor]:
        """Wait for combine to finish and return the combined result."""
        if self.enable_overlap:
            if self.fp8 and not self.fp8_wgrad:
                self._backward_fp8_bf16_wgrad_wait()
        paddle.base.core.nvprof_nvtx_pop()

        global _global_previous_event
        _global_previous_event = self.combine_done_event
        input_grads = self.input_grads
        self.combine_done_event.current_stream_wait()
        del self.input_grads, self.combine_done_event
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
