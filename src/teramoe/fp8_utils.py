from typing import Callable

import paddle
import paddle.distributed as dist
from paddle import Tensor
from paddle.distributed.communication.group import Group

from teramoe import deep_ep, deep_gemm

FP8_ALIGN = 128
QUANT_BLOCK_SIZE = 512

_grouped_launch_stream = None
_task_done_event = None
_sort_map_stream = None
_sort_map_done_event = None


class GroupedTaskLauncher:
    def __init__(self,
        funcs: list[Callable[[int], None]],
        num_tasks: int,
        dispatch_done_event: deep_ep.EventOverlap | None = None,
        combine_overlap_ratio: float = 0.0,
        avg_num_tasks: int = None,
    ):
        self._funcs = funcs
        self._num_tasks = num_tasks
        self._dispatch_done_event = dispatch_done_event
        self._combine_overlap_ratio = combine_overlap_ratio
        self._avg_num_tasks = num_tasks if avg_num_tasks is None else avg_num_tasks
        self._next_task_idx = 0
        self._calc_stream = paddle.cuda.current_stream()

        global _grouped_launch_stream, _task_done_event
        if _grouped_launch_stream is None:
            _grouped_launch_stream = paddle.cuda.Stream()
        if _task_done_event is None:
            _task_done_event = paddle.cuda.Event()

    def run_dispatch_overlap(self, dual_stream: bool = False):
        """Stage A: compute overlaps with dispatch util dispatch finishes."""
        assert self._dispatch_done_event is not None, (
            "dispatch_overlap requires dispatch_done_event be given")
        if dual_stream:
            return self._run_dispatch_overlap_dual_stream()

        for task_idx in range(self._num_tasks):
            paddle.base.core.nvprof_nvtx_push(f"A{task_idx}")
            for func in self._funcs:
                func(task_idx)
            paddle.base.core.nvprof_nvtx_pop()

            # 只提前发射一个 task, 从而及时根据 dispatch 完成状态切换 SM 数
            if task_idx > 0:
                _task_done_event.synchronize()
            _task_done_event.record()
            self._next_task_idx = task_idx + 1

            # 当 dispatch 恰好完成时, 切换至纯计算模式
            if self._dispatch_done_event.query():
                break

        return self._next_task_idx

    def _run_dispatch_overlap_dual_stream(self):
        assert len(self._funcs) == 4, "dual stream is for 4-stage task"
        gemm0, act, gemm1, zip = self._funcs
        streams = [self._calc_stream, _grouped_launch_stream]
        stream_bases = [stream.stream_base for stream in streams]

        for task_idx in range(self._num_tasks):
            i = task_idx % 2
            paddle.base.core._set_current_stream(stream_bases[i])

            paddle.base.core.nvprof_nvtx_push(f"A{task_idx}")
            gemm0(task_idx)

            # 上一个 task 的 zip
            if task_idx > 0:
                paddle.base.core._set_current_stream(stream_bases[1 - i])
                zip(task_idx - 1)
                paddle.base.core._set_current_stream(stream_bases[i])

            act(task_idx)
            _task_done_event.record()

            gemm1(task_idx)
            paddle.base.core.nvprof_nvtx_pop()

            # 等当前 task 的 act 完成才开始发射下一个 task
            _task_done_event.synchronize()
            self._next_task_idx = task_idx + 1

            # 当 dispatch 恰好完成时, 切换至纯计算模式
            if self._dispatch_done_event.query():
                break

        # 结束 dispatch overlap 阶段时, 在最后一个 task 所在的 stream 发射其 zip
        if self._num_tasks > 0:
            zip(task_idx)
            paddle.base.core._set_current_stream(stream_bases[0])

        return self._next_task_idx

    def _grouped_launch(self, begin, end):
        """
        轮流在 calc/comm 两个 stream 上对 funcs 进行分组发射.

        使用两个 stream 可以让前后两个 kernel 重叠, 让下一个 kernel 充分利用上一个 kernel
        的尾部空出来的 SM, 达到类似 group_gemm 的效果.
        """
        # 对于每组 func，总是从主 stream 开始发射，这样同一个 task_idx 的前后 func
        # 必定在同一个 stream 上，可以天然保证同步
        stream_bases = [self._calc_stream.stream_base, _grouped_launch_stream.stream_base]
        i = 0

        for n, func in enumerate(self._funcs):
            # 每组 func 开始前让副 stream 等待一次主 stream, 保持组的边界;
            # 其实从正确性上没有必要, 只是让 timeline 更整齐, 对性能影响未知
            _task_done_event.record()
            _grouped_launch_stream.wait_event(_task_done_event)

            paddle.base.core.nvprof_nvtx_push(f"G{n}")
            for i, task_idx in enumerate(range(begin, end)):
                # 直接调用 core 比用 stream_guard 开销更低, 这里的发射速度非常关键
                if i != 0:
                    paddle.base.core._set_current_stream(stream_bases[i % 2])
                func(task_idx)
            paddle.base.core.nvprof_nvtx_pop()

            # 每组 func 调用结束时恢复到默认计算流
            if i % 2 != 0:
                paddle.base.core._set_current_stream(stream_bases[0])

        # 仅结束时主 stream 需要等待一次副 stream
        _task_done_event.record(_grouped_launch_stream)
        self._calc_stream.wait_event(_task_done_event)

    def run_compute(self):
        """Stage B: compute only."""
        begin = self._next_task_idx
        # 将 combine_overlap_ratio 的 task 留给 combine 进行 overlap, 但是 task 基数由均衡
        # 情况确定, 以此保持留出的 task 数量恒定 (而不是比例恒定)
        num_overlap_tasks = self._avg_num_tasks * self._combine_overlap_ratio
        end = max(int(self._num_tasks - num_overlap_tasks), begin)  # excluded

        if begin < end:
            paddle.base.core.nvprof_nvtx_push(f"B{begin}_{end - 1}")
            self._grouped_launch(begin, end)
            paddle.base.core.nvprof_nvtx_pop()

        self._next_task_idx = end
        return end

    def run_combine_overlap(self):
        """Stage C: compute overlaps with combine."""
        for task_idx in range(self._next_task_idx, self._num_tasks):
            paddle.base.core.nvprof_nvtx_push(f"C{task_idx}")
            for func in self._funcs:
                func(task_idx)
            paddle.base.core.nvprof_nvtx_pop()
            self._next_task_idx = task_idx + 1

    def __del__(self):
        assert self._next_task_idx == self._num_tasks, (
            f"tasks not correctly launched: next={self._next_task_idx} total={self._num_tasks}")


class AsyncLoad:
    def __init__(self):
        self._pin = None
        self._event = None

    def __call__(self, x, dtype=None) -> paddle.Tensor:
        """Copy x to GPU asynchronously."""
        assert self._pin is None, (
            "The previous copy is not finished, call wait() first before reuse for another copy.")

        cudart = paddle.cuda.cudart()
        self._pin = paddle.to_tensor(x, dtype=dtype, place=paddle.CUDAPinnedPlace())

        out = paddle.empty_like(self._pin)
        stream = paddle.cuda.current_stream()

        err = cudart.cudaMemcpyAsync(
            out.data_ptr(),
            self._pin.data_ptr(),
            out.size * out.itemsize,
            cudart.cudaMemcpyHostToDevice,
            stream.stream_base.cuda_stream,
        )
        assert err == cudart.cudaError.success, f"cudaMemcpyAsync failed: {err}"

        # the pinned tensor cannot be freed before the copy event is done
        if self._event is None:
            self._event = paddle.cuda.Event()
        self._event.record()

        return out

    def wait(self):
        """Wait the current copy to finish and free the pinned tensor."""
        if self._pin is not None:
            self._event.synchronize()
            self._pin = None

    def __del__(self):
        self.wait()


def async_offload(x: Tensor) -> Tensor:
    """Copy x to pin_memory asynchronously."""
    assert x.dtype == paddle.int32 and x.place.is_gpu_place()

    pin = paddle.core.DenseTensor()
    pin._set_dims(x.shape)
    pin._alloc_int(paddle.CUDAPinnedPlace())

    cudart = paddle.cuda.cudart()
    stream = paddle.cuda.current_stream()

    err = cudart.cudaMemcpyAsync(
        pin._ptr(),
        x.data_ptr(),
        x.size * x.itemsize,
        cudart.cudaMemcpyDeviceToHost,
        stream.stream_base.cuda_stream,
    )
    assert err == cudart.cudaError.success, f"cudaMemcpyAsync failed: {err}"

    out = paddle.Tensor()
    out.get_tensor()._share_data_nocheck_with(pin)
    return out


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
    assert scale.shape == scale_shape[::-1] if transpose_scale else scale_shape
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
        if hasattr(w, "fp8_weight_stacked_transpose"):
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
        self.tokens_per_expert = None
        self.tokens_per_card_max_mean_ratio = None

    def _reduce_max_tokens_per_card(self, num_tokens_per_expert: Tensor) -> Tensor:
        """归约当前 ep_group 内每个 rank 的展开 token 数的最大值, 用于统计均衡度.

        NOTES:
            1. 通信量为 ep_size 个 int, 成本极低;
            2. 同时起到了 barrier_ep 的作用;
            3. 返回值由 notify_dispatch 天然同步, 需要在调用完 dispatch 后才能使用.
        """
        global_num_tokens_per_card = num_tokens_per_expert.reshape(
            [self.group.nranks, -1]).sum(axis=1, dtype="int32")
        dist.all_reduce(global_num_tokens_per_card, group=self.group)
        return async_offload(global_num_tokens_per_card.max())

    def forward(self):
        """Forward dispatch, compute and combine.

        This function has no input/output, as the inputs are set at node initialization,
        and the output is hold util the user calls forward_wait.
        """
        if self.fp8:
            paddle.base.core.nvprof_nvtx_push("teramoe_fp8_fwd")
            self._forward_fp8()
            paddle.base.core.nvprof_nvtx_pop()
        else:
            paddle.base.core.nvprof_nvtx_push("teramoe_bf16_fwd")
            self._forward_bf16()
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

        max_tokens_per_card = self._reduce_max_tokens_per_card(num_tokens_per_expert)

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

        self.tokens_per_expert = tokens_per_expert
        avg_tokens_per_card = self.token_indices.size  # seqlen * topk
        self.tokens_per_card_max_mean_ratio = max_tokens_per_card.item() / avg_tokens_per_card

        del self.hidden_states
        self.recv_x = recv_x
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

        avg_num_tasks = int(avg_tokens_per_card / self.chunk_size + len(w_gateup) / 2)

        funcs = [
            lambda task_idx: deep_gemm.bf16_chunk_gemm_nn(
                unzipped_tokens, w_gateup, o1, task_queue, task_idx),
            lambda task_idx: deep_gemm.chunk_weighted_swiglu(
                o1, unzipped_probs, o2, task_queue, task_idx, self.chunk_size, precise=True),
            lambda task_idx: deep_gemm.bf16_chunk_gemm_nn(o2, w_down, o3, task_queue, task_idx),
            lambda task_idx: deep_gemm.chunk_zip(
                o3, zipped_out, atomic_to_zip, zip_to_atomic, recv_token_indices, num_valid_topk,
                token_done, zip_done, task_queue, task_idx, self.chunk_size)
        ]

        task_launcher = GroupedTaskLauncher(
            funcs, len(task_queue), event, self.combine_overlap_ratio, avg_num_tasks)

        deep_gemm.set_num_sms(self.num_calc_sms)
        task_launcher.run_dispatch_overlap()

        deep_gemm.set_num_sms(0)
        task_launcher.run_compute()

        ############################# COMBINE FORWARD ##############################

        out, _, event = self.buffer.combine(
            zipped_out, handle, async_finish=True, previous_event=deep_ep.Buffer.capture(),
            allocate_on_comm_stream=False, zip_done=zip_done)

        deep_gemm.set_num_sms(self.num_calc_sms)
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

        max_tokens_per_card = self._reduce_max_tokens_per_card(num_tokens_per_expert)

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

        self.tokens_per_expert = tokens_per_expert
        avg_tokens_per_card = self.token_indices.size
        self.tokens_per_card_max_mean_ratio = max_tokens_per_card.item() / avg_tokens_per_card

        del hidden_states
        self.recv_x = recv_x
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

        avg_num_tasks = int(avg_tokens_per_card / self.chunk_size + len(w_gateup) / 2)

        funcs = [
            lambda task_idx: deep_gemm.fp8_chunk_gemm_nt(
                unzipped_x, w_gateup_t, o1, task_queue, task_idx),
            lambda task_idx: deep_gemm.chunk_weighted_swiglu(
                o1, unzipped_probs, o2_fp8, task_queue, task_idx, self.chunk_size,
                o2_scales=o2_scale),
            lambda task_idx: deep_gemm.fp8_chunk_gemm_nt(
                (o2_fp8, o2_scale), w_down_t, o3, task_queue, task_idx),
            lambda task_idx: deep_gemm.chunk_zip(
                o3, zipped_out, atomic_to_zip, zip_to_atomic, recv_token_indices, num_valid_topk,
                token_done, zip_done, task_queue, task_idx, self.chunk_size),
        ]

        task_launcher = GroupedTaskLauncher(
            funcs, len(task_queue), event, self.combine_overlap_ratio, avg_num_tasks)

        deep_gemm.set_num_sms(self.num_calc_sms)
        task_launcher.run_dispatch_overlap()

        deep_gemm.set_num_sms(0)
        task_launcher.run_compute()

        ########################### COMBINE FORWARD ############################

        out, _, event = self.buffer.combine(
            zipped_out, handle, async_finish=True, allocate_on_comm_stream=False,
            zip_done=zip_done)

        deep_gemm.set_num_sms(self.num_calc_sms)
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
        t = async_load(ks_cpu + m_start + m_start_wgrad, dtype="int32")
        self.grouped_layout = t[:len(ks_cpu)]
        self.m_start_gpu = t[len(ks_cpu):-len(m_start_wgrad)]
        self.m_start_wgrad_gpu = t[-len(m_start_wgrad):]

        self.ordered_to_zip = paddle.empty([m_start_wgrad[-1]], dtype="int32")
        ordered_to_atomic = paddle.empty_like(self.ordered_to_zip) if not self.fp8_wgrad else None
        deep_gemm.sort_map(zip_to_atomic, self.m_start_gpu, m_start_wgrad[-1],
                           self.ordered_to_zip, ordered_to_atomic, self.m_start_wgrad_gpu)

        # 提前为 bf16_wgrad 恢复顺序
        if not self.fp8_wgrad:
            self.o1 = deep_gemm.token_gather(self.o1, ordered_to_atomic)
            self.unzipped_probs = deep_gemm.token_gather(
                self.unzipped_probs.unsqueeze(-1), ordered_to_atomic)
            self.recv_token_indices = self.recv_token_indices.cast("int32")

    def forward_wait(self) -> Tensor:
        """Wait for combine to finish and return the combined result."""
        out = self.out
        self.combine_done_event.current_stream_wait()
        del self.out, self.combine_done_event
        return out

    def backward(self, dout: Tensor):
        """Backward dispatch, compute, combine and wgrad.

        This function accepts one input, but has no output in the same way as forward.
        """
        if self.fp8:
            if self.fp8_wgrad:
                paddle.base.core.nvprof_nvtx_push("teramoe_fp8_bwd")
                self._backward_fp8(dout)
                paddle.base.core.nvprof_nvtx_pop()
            else:
                paddle.base.core.nvprof_nvtx_push("teramoe_fp8_bwd_bf16_wgrad")
                self._backward_fp8_bf16_wgrad(dout)
                # nvtx pop on wait
        else:
            paddle.base.core.nvprof_nvtx_push("teramoe_bf16_bwd")
            self._backward_bf16(dout)
            paddle.base.core.nvprof_nvtx_pop()

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
                self.chunk_size, precise=True),
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
            drecv_x, handle, drecv_probs, async_finish=True,
            previous_event=deep_ep.Buffer.capture(), allocate_on_comm_stream=False,
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
        global _sort_map_stream, _sort_map_done_event
        if _sort_map_stream is None:
            _sort_map_stream = paddle.cuda.Stream()
        if _sort_map_done_event is None:
            _sort_map_done_event = paddle.cuda.Event()

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

        # 将 wgrad 的 recv_x requant 提前到 dispatch overlap, 因为一般 gemm 刚开始都在空等;
        # 该 sort_map 使用的都是前向的数据, 所以与反向 dispatch 没有数据依赖
        x_w = deep_gemm.requant_wgrad_input(
            self.recv_x[0], self.recv_x[1].T.contiguous().T, self.ordered_to_zip)

        # 复用 ordered_to_zip 给 sort_map, 这要求 sort_map 必须在 requant 完成后才执行
        ordered_to_atomic = self.ordered_to_zip
        del self.recv_x, self.ordered_to_zip
        _task_done_event.record()
        _sort_map_stream.wait_event(_task_done_event)

        # 将反向 sort_map 用异步流紧跟在反向 dispatch 之后, 因为 dispatch 刚结束时 gemm 还没有立即切换到
        # compute 阶段, 此时 SM 有空余, 可以充分利用起来
        with paddle.device.stream_guard(_sort_map_stream):
            event.current_stream_wait()
            deep_gemm.sort_map(zip_to_atomic_bwd, self.m_start_gpu, self.m_start_wgrad[-1],
                               None, ordered_to_atomic, self.m_start_wgrad_gpu)
            _sort_map_done_event.record()

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
                self.chunk_size, o2_bwd_scales=o2_bwd_scale, do1_scales=do1_scale),
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

        paddle.cuda.current_stream().wait_event(_sort_map_done_event)
        do1_w = deep_gemm.requant_wgrad_input(*do1, ordered_to_atomic)
        o2_w = deep_gemm.requant_wgrad_input(*o2_bwd, ordered_to_atomic)
        do3_w = deep_gemm.requant_wgrad_input(*do3, ordered_to_atomic)

        fp8_weight_grad(x_w, do1_w, w_gateup, self.ks_cpu, self.grouped_layout)
        fp8_weight_grad(o2_w, do3_w, w_down, self.ks_cpu, self.grouped_layout)

        paddle.base.core.nvprof_nvtx_pop()

    def _backward_fp8_bf16_wgrad(self, dout: Tensor):
        ########################### COMBINE BACKWARD ###########################

        # 非 chunk 模式, 可以使用 cache mode, 发送 BF16
        recv_dout, _, _, _, _, event = self.buffer.dispatch(
            dout, handle=self.handle, async_finish=True, allocate_on_comm_stream=False)

        self.recv_dout = recv_dout
        self.dispatch_done_event = event

        recv_indices = self.recv_token_indices
        recv_probs = paddle.empty(recv_indices.shape, dtype="float32")  # placeholder
        self.m_indices = paddle.concat([
            paddle.full([self.ks_cpu[i]], i, dtype="int32")
            for i, n in enumerate(self.tokens_per_expert)
        ])

        # x 在 wgrad 时才用, 这里提前到 dispatch 以进行 overlap
        if self.recv_x[0].shape[0] > 0:
            recv_x_dequant = paddle.incubate.nn.functional.fused_act_dequant(*self.recv_x)
        else:
            recv_x_dequant = paddle.empty(self.recv_x[0].shape, dtype="bfloat16")
        del self.recv_x

        self.x, self.rowmap, _, _ = paddle.nn.functional.moe_permute(
            recv_x_dequant, None, recv_indices, recv_probs,
            padding_alignment=FP8_ALIGN,
            num_experts=len(self.w_gateup),
            tokens_per_expert=self.tokens_per_expert,
        )

    def _backward_fp8_bf16_wgrad_wait(self):
        import paddlefleet_ops

        w_gateup, w_down = self.w_gateup, self.w_down
        H, I = w_gateup.shape[1], w_down.shape[1]
        w_gateup_n = get_quant_weight(w_gateup)
        w_down_n = get_quant_weight(w_down)

        self.dispatch_done_event.current_stream_wait()
        del self.dispatch_done_event

        ############################ GEMM BACKWARD #############################

        deep_gemm.set_num_sms(0)

        # NOTES: paddle unzip 性能较差, 这里用重复读的 gather 性能都比它好, 而且实际运行时大部分 token
        #        都只有一个专家, 重复读影响很小; 前面用 paddle 只是为了构造 rowmap 给后面的 zip 用
        # NOTES: 先 unzip 再 quant, 因为 paddle quant 算子在非对齐情况性能较差
        do3 = deep_gemm.token_gather(self.recv_dout, self.ordered_to_zip)
        del self.recv_dout
        do3_fp8, do3_scale = quant_input(do3, transpose_scale=True)

        ########## DOWN_GRAD ###########

        recv_indices = self.recv_token_indices
        num_zipped_tokens = len(recv_indices)
        num_unzipped_tokens = self.m_start_wgrad[-1]

        do2 = paddle.empty([num_unzipped_tokens, I], dtype="bfloat16")
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (do3_fp8, do3_scale.T), w_down_n, do2, self.m_indices)
        del do3_fp8, do3_scale

        ######### SWIGLU_GRAD ##########

        do1, dprobs, o2 = paddlefleet_ops.fused_swiglu_probs_bwd(
            self.o1, do2, self.unzipped_probs, inplace=True)
        del self.o1, do2, self.unzipped_probs

        do1_fp8, do1_scale = quant_input(do1, transpose_scale=True)

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

        bf16_weight_grad(self.x, do1, w_gateup, self.ks_cpu, self.grouped_layout)
        del self.x, do1

        bf16_weight_grad(o2, do3, w_down, self.ks_cpu, self.grouped_layout)

        paddle.base.core.nvprof_nvtx_pop()

    def backward_wait(self) -> [Tensor, Tensor]:
        """Wait for combine to finish and return the combined result."""
        if self.fp8 and not self.fp8_wgrad:
            self._backward_fp8_bf16_wgrad_wait()
            paddle.base.core.nvprof_nvtx_pop()

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
