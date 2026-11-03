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


if __name__ == "__main__":
    paddle.empty([32, 1024, 1024, 1024], "uint8")

    master_grad = paddle.randn([4096, 4096])
    master_grad_ref = master_grad.clone()
    replica_grads = [paddle.randn_like(master_grad) for _ in range(6)]

    for grad in replica_grads:
        master_grad_ref.add_(grad)

    accumulate_wgrad_(master_grad, replica_grads)

    print("master_grad:", master_grad)
    print("master_grad_ref:", master_grad_ref)

    assert (master_grad == master_grad_ref).all()

    e0 = paddle.cuda.Event(enable_timing=True)
    e1 = paddle.cuda.Event(enable_timing=True)

    for i in range(10):
        paddle.randn([1024, 1024, 1024])
        e0.record()
        for grad in replica_grads:
            master_grad_ref.add_(grad)
        e1.record()
        paddle.device.synchronize()
        print("paddle:", e0.elapsed_time(e1)) if i else ()

    for i in range(10):
        paddle.randn([1024, 1024, 1024])
        e0.record()
        accumulate_wgrad_(master_grad, replica_grads)
        e1.record()
        paddle.device.synchronize()
        print(f"triton:", e0.elapsed_time(e1)) if i else ()
