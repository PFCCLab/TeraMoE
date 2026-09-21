from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import paddle
from paddle import Tensor
from paddle.autograd import PyLayer
from paddle.distributed.communication.group import Group

from .fp8_utils import FP8_ALIGN, TeraMoENode
from .fused_a2a import get_buffer, get_hidden_bytes


if TYPE_CHECKING:
    from typing import Any, Callable, NotRequired, Protocol, TypedDict

    class _GroupedGemmExpert(Protocol):
        weight1: Tensor
        weight2: Tensor

    class _CombineOverlapHandle(TypedDict):
        fn: Callable[[Any], Any]
        fn_args: tuple[Any]
        fn_out: NotRequired[Any]


def _fake_clone(s: Tensor) -> Tensor:
    """Create a cloned-like tensor with zero copy.

    The returned tensor breaks the gradient and inplace_version from the source tensor,
    but still shares the same GPU buffer.
    """
    t = paddle.Tensor()
    t.get_tensor()._share_data_nocheck_with(s.get_tensor())
    return t


class PreFnNode(PyLayer):
    """Applied before fn.

    During forward, does dispatch-compute-combine but not wait for combine, so that fn can overlap
    with combine. During backward, its role switches to PostFnNode."""

    @staticmethod
    def forward(ctx, hs: Tensor, probs: Tensor, node: TeraMoENode, *fn_args):
        node.forward()
        ctx.node = node
        # NOTES: 对 fn_args 进行假 clone, 既防止 fn_args 为叶节点时 PyLayer 报错,
        #        又防止 PyLayer 改变其 inplace_version 导致反向报错
        fn_args = [_fake_clone(t) for t in fn_args]
        paddle.base.core.nvprof_nvtx_push("shared_expert_fwd")
        return paddle.empty([0]), *fn_args

    @staticmethod
    def backward(ctx, _, *fn_args_grads):
        paddle.base.core.nvprof_nvtx_pop()
        hs_grad, probs_grad = ctx.node.backward_wait()
        return hs_grad, probs_grad, *fn_args_grads


class PostFnNode(PyLayer):
    """Applied after fn.

    During foward, waits for combine and gives the combined result. During backward, its role
    switches to PreFnNode."""

    @staticmethod
    def forward(ctx, _, node: TeraMoENode, *fn_out):
        paddle.base.core.nvprof_nvtx_pop()
        out = node.forward_wait()
        ctx.node = node
        return out, *fn_out

    @staticmethod
    def backward(ctx, out_grad, *fn_out_grads):
        ctx.node.backward(out_grad)
        paddle.base.core.nvprof_nvtx_push("shared_expert_bwd")
        return paddle.empty([0]), *fn_out_grads


def forward_autograd(
    hidden_states: Tensor,
    topk_weights: Tensor,
    topk_indices: Tensor,
    grouped_gemm_experts: _GroupedGemmExpert,
    num_experts: int,
    group: Group,
    combine_overlap_handle: _CombineOverlapHandle | None = None,
    hidden_act: Literal["silu"] = "silu",
    fp8: Literal["e4m3"] | None = None,
    fp8_wgrad: bool = True,
    use_ue8m0: bool = True,
    chunk_size: int = 4096,
    num_calc_sms: int = 100,
    combine_overlap_ratio: float = 0.3,
):
    # Check params
    assert hidden_act == "silu", f"Only supports silu activation, got {hidden_act}."
    assert fp8 is None or fp8 == "e4m3", f"Only supports fp8 e4m3, got {fp8}."
    assert fp8 is None or use_ue8m0, f"Only supports ue8m0 scale."
    assert isinstance(chunk_size, int) and chunk_size > 0 and chunk_size % FP8_ALIGN == 0, (
        f"The chunk_size must be multiple of {FP8_ALIGN}, got {chunk_size}.")
    assert isinstance(num_calc_sms, int) and num_calc_sms > 0, (
        f"The num_calc_sms must be positive integer, got {num_calc_sms}.")
    assert isinstance(combine_overlap_ratio, (int, float)) and 0 <= combine_overlap_ratio <= 1, (
        f"The combine_overlap_ratio must be in range [0, 1], got {combine_overlap_ratio}.")

    # Check weights
    w_gateup, w_down = grouped_gemm_experts.weight1, grouped_gemm_experts.weight2
    assert (
        w_gateup.ndim == 3 and w_down.ndim == 3 and w_gateup.shape[0] == w_down.shape[0] and
        w_gateup.shape[1] == w_down.shape[2] and w_gateup.shape[2] == w_down.shape[1] * 2
    ), (
        "Requires weight1 have shape [E, H, 2I], weight2 have shape [E, I, H], got "
        f"weight1.shape={w_gateup.shape}, weight2.shape={w_down.shape}."
    )
    assert w_gateup.is_contiguous() and w_down.is_contiguous(), "Requires weights be contiguous."

    # Check hidden
    assert hidden_states.shape[-1] == w_gateup.shape[1], (
        f"Requires hidden_states have shape [..., H], got {hidden_states.shape} "
        f"while H={w_gateup.shape[1]}.")
    assert hidden_states.is_contiguous(), "Requires hidden_states be contiguous."

    buffer = get_buffer(group, get_hidden_bytes(hidden_states))
    hs_2d = hidden_states.view([-1, hidden_states.shape[-1]])

    # 通过两个 node 将 teramoe 的流程拆成 dispatch-compute-combine 和 wait 两部分:
    # 1. 前向进行完第一部分之后不等待 combine, 而是继续执行 combine_overlap_handle, 从而实现
    #    它与 combine 的 overlap
    # 2. 由于前向我们用 pre/post 两个节点夹住了 combine_overlap_handle 的计算, 因此能够保证
    #    其反向恰好在 post 与 pre 之间执行, 从而也能控制它与 combine 进行 overlap
    node = TeraMoENode(
        buffer, hs_2d, topk_weights, topk_indices, w_gateup, w_down, num_experts,
        fp8, fp8_wgrad, chunk_size, num_calc_sms, combine_overlap_ratio)

    fn_args, fn_out = (), ()
    if combine_overlap_handle is not None:
        fn_args = combine_overlap_handle["fn_args"]

    # fake_state 只是为了在 pre/post 之间传递计算图依赖, 本身不承载数据, 中间数据都存在 node 里
    fake_state, *fn_args = PreFnNode.apply(hs_2d, topk_weights, node, *fn_args)
    if combine_overlap_handle is not None:
        fn_out = combine_overlap_handle["fn"](*fn_args)

    out, *fn_out = PostFnNode.apply(fake_state, node, *fn_out)
    if combine_overlap_handle is not None:
        combine_overlap_handle["fn_out"] = fn_out

    return out
