import sys
import numpy as np
from contextlib import contextmanager
from typing import Any

import paddle
from paddle import Tensor


class AsyncLoad:
    """Copy data H2D asynchronously with pin memory life-cycle control."""

    def __init__(self):
        self._pin = None
        self._event = None

    def __call__(self, x: Any, dtype=None) -> Tensor:
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
    """Copy data D2H asynchronously."""
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


class DummyTimer:
    def start(self):
        pass

    def stop(self):
        pass


def dummy_timers(name: str, use_event: bool = False) -> DummyTimer:
    return DummyTimer()


def get_profile_timers():
    """Get paddlefleet profile timers if imported."""
    if "paddlefleet.training.global_vars" in sys.modules:
        from paddlefleet.training.global_vars import get_profile_timers
        timers = get_profile_timers()
        if timers is not None:
            return timers
    return dummy_timers


@contextmanager
def started(timer: Any):
    timer.start()
    try:
        yield
    finally:
        timer.stop()
