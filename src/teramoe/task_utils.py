import numpy as np
from typing import Callable

import paddle

from . import deep_ep

FP8_ALIGN = 128

_grouped_launch_stream = None
_task_done_event = None


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

    def run_dispatch_overlap(self):
        """Stage A: compute overlaps with dispatch util dispatch finishes."""
        assert self._dispatch_done_event is not None, (
            "dispatch_overlap requires dispatch_done_event be given")

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
            paddle.base.core.nvprof_nvtx_push(f"Bx{end - begin}")
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


def make_batch_task_queue(tokens_per_expert: list[int]) -> np.ndarray:
    """Make a fine-grained task queue with chunk size 128 for batched launch."""
    tokens_per_expert = np.array(tokens_per_expert, np.int32)
    tokens_per_expert = tokens_per_expert[tokens_per_expert != 0]
    tasks_per_expert = (tokens_per_expert + FP8_ALIGN - 1) // FP8_ALIGN
    task_queue = np.empty((tasks_per_expert.sum(), 4), np.int32)

    # expert_id can be omitted
    task_queue[:, 1] = np.arange(0, len(task_queue) * FP8_ALIGN, FP8_ALIGN, np.int32)
    task_queue[:, 2] = FP8_ALIGN
    task_queue[:, 3] = 1

    last_task_per_expert = tasks_per_expert.cumsum() - 1
    last_size_per_expert = tokens_per_expert % FP8_ALIGN
    task_queue[last_task_per_expert, 2] = np.where(
        last_size_per_expert, last_size_per_expert, FP8_ALIGN)
    return task_queue
