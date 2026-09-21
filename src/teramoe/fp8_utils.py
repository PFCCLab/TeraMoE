from typing import Callable

import paddle
from paddle import Tensor
from paddle.distributed.communication.group import Group

from teramoe import deep_ep, deep_gemm

FP8_ALIGN = 128

_grouped_launch_stream = None
_task_done_event = None


class GroupedTaskLauncher:
    def __init__(self,
        funcs: list[Callable[[int], None]],
        num_tasks: int,
        dispatch_done_event: deep_ep.EventOverlap,
        combine_overlap_ratio: float = 0.3,
    ):
        self._funcs = funcs
        self._num_tasks = num_tasks
        self._dispatch_done_event = dispatch_done_event
        self._combine_overlap_ratio = combine_overlap_ratio
        self._next_task_idx = 0
        self._calc_stream = paddle.cuda.current_stream()

        global _grouped_launch_stream, _task_done_event
        if _grouped_launch_stream is None:
            _grouped_launch_stream = paddle.cuda.Stream()
        if _task_done_event is None:
            _task_done_event = paddle.cuda.Event()

    def run_dispatch_overlap(self, dual_stream: bool = False):
        """Stage A: compute overlaps with dispatch util dispatch finishes."""
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
        # 对于每组 func，总是从 calc_stream 开始发射，这样同一个 task_idx 的前后 func
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

    def run_compute(self):
        """Stage B: compute only."""
        begin = self._next_task_idx
        end = max(int(self._num_tasks * (1 - self._combine_overlap_ratio)), begin)  # excluded

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


class TeraMoENode:
    def __init__(
        self,
        buffer: deep_ep.Buffer,
        hidden_states: Tensor,
        token_probs: Tensor,
        token_indices: Tensor,
        w_gateup: Tensor,
        w_down: Tensor,
        num_experts: int,
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
        self.num_experts = num_experts
        self.chunk_size = chunk_size
        self.num_calc_sms = num_calc_sms
        self.combine_overlap_ratio = combine_overlap_ratio

    def forward(self):
        """Forward dispatch, compute and combine.

        This function has no input/output, as the inputs are set at node initialization,
        and the output is hold util the user calls forward_wait.
        """
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

        self.dispatch_layout = {
            "num_tokens_per_rank": num_tokens_per_rank,
            "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "is_token_in_rank": is_token_in_rank,
        }

        (
            recv_x, recv_token_indices, recv_token_probs,
            num_recv_tokens_per_expert_list, handle, event,
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
                o1, unzipped_probs, o2, task_queue, task_idx, self.chunk_size, precise=True),
            lambda task_idx: deep_gemm.bf16_chunk_gemm_nn(o2, w_down, o3, task_queue, task_idx),
            lambda task_idx: deep_gemm.chunk_zip(
                o3, zipped_out, atomic_to_zip, zip_to_atomic, recv_token_indices, num_valid_topk,
                token_done, zip_done, task_queue, task_idx, self.chunk_size)
        ]

        task_launcher = GroupedTaskLauncher(
            funcs, len(task_queue), event, self.combine_overlap_ratio)

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

        ordered_to_zip, ordered_to_atomic = deep_gemm.sort_map(
            zip_to_atomic_bwd, m_start, num_unzipped_tokens)

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

    def backward_wait(self) -> [Tensor, Tensor]:
        """Wait for combine to finish and return the combined result."""
        input_grads = self.input_grads
        self.combine_done_event.current_stream_wait()
        del self.input_grads, self.combine_done_event
        return input_grads


def bf16_weight_grad(x, dy, weight, ks_cpu, grouped_layout):
    attr = "main_grad" if hasattr(weight, "main_grad") else "grad"
    grad = getattr(weight, attr)
    if grad is None:
        grad = paddle.zeros(weight.shape, dtype="float32")
        setattr(weight, attr, grad)
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(x, dy, grad, ks_cpu, grouped_layout, grad)
