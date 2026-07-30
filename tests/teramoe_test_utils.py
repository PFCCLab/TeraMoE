import contextlib
import math
import os
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
import torch.nn.functional as F
from packaging import version

import teramoe

os.environ.setdefault('NVTE_CUTEDSL_FUSED_GROUPED_MLP', '1')
os.environ.setdefault('NVTE_GROUPED_LINEAR_SINGLE_PARAM', '1')

import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common import recipe as te_recipe
from transformer_engine.pytorch import fp8_autocast, moe_permute_with_probs, moe_unpermute
import triton
import triton.language as tl

@triton.jit
def _indices_to_multihot_kernel(
    indices_ptr,
    probs_in_indices_ptr,
    multihot_indices_ptr,
    probs_in_multihot_ptr,
    position_map_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    row_idx = tl.program_id(0)
    indices_row = tl.load(indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)
    indices_row = tl.where(topk_row_mask, indices_row, -1)
    probs_row = tl.load(probs_in_indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)

    position_row = tl.where(indices_row != -1, topk_row, -1)
    mask = (indices_row != -1) & (indices_row < num_of_local_experts)

    row_idx_offset = row_idx * num_of_local_experts
    tl.store(multihot_indices_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask)
    tl.store(position_map_ptr + row_idx_offset + num_exp_row, -1, mask=num_exp_row_mask)
    tl.debug_barrier()
    tl.store(multihot_indices_ptr + row_idx_offset + indices_row, 1, mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + indices_row, probs_row, mask)
    tl.store(position_map_ptr + row_idx_offset + indices_row, position_row, mask)


@triton.jit
def _multihot_to_indices_kernel(
    probs_in_multihot_ptr,
    position_map_ptr,
    probs_indices_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    topk_row = tl.arange(0, topk_next_power_of_2)
    topk_row = tl.where(topk_row < topk, topk_row, -1)
    topk_row_mask = topk_row != -1
    num_exp_row = tl.arange(0, num_of_local_experts_next_power_of_2)
    num_exp_row = tl.where(num_exp_row < num_of_local_experts, num_exp_row, -1)
    num_exp_row_mask = num_exp_row != -1

    row_idx = tl.program_id(0)
    ptr_offset = row_idx * num_of_local_experts + num_exp_row
    probs_in_multihot_row = tl.load(probs_in_multihot_ptr + ptr_offset, mask=num_exp_row_mask)
    position_map_row = tl.load(position_map_ptr + ptr_offset, mask=num_exp_row_mask)
    position_map_row = tl.where(num_exp_row_mask, position_map_row, -1)
    mask = position_map_row != -1

    tl.store(probs_indices_ptr + row_idx * topk + topk_row, 0, mask=topk_row_mask)
    tl.debug_barrier()
    tl.store(probs_indices_ptr + row_idx * topk + position_map_row, probs_in_multihot_row, mask)


class IndicesToMultihot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, probs_indices, num_of_local_experts):
        num_of_tokens = indices.shape[0]
        assert indices.shape == probs_indices.shape
        topk = indices.shape[1]
        multihot_indices = torch.empty((num_of_tokens, num_of_local_experts), dtype=torch.bool, device="cuda")
        probs_in_multihot = torch.empty((num_of_tokens, num_of_local_experts), dtype=probs_indices.dtype, device="cuda")
        position_map = torch.empty((num_of_tokens, num_of_local_experts), dtype=torch.int32, device="cuda")
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(num_of_local_experts)))
        _indices_to_multihot_kernel[(num_of_tokens,)](
            indices, probs_indices, multihot_indices, probs_in_multihot, position_map,
            num_of_local_experts, num_of_local_experts_next_power_of_2, topk,
            topk_next_power_of_2, BLOCK_SIZE=32, num_warps=1)
        ctx.save_for_backward(position_map)
        ctx.num_of_tokens = num_of_tokens
        ctx.num_of_local_experts = num_of_local_experts
        ctx.topk = topk
        return multihot_indices, probs_in_multihot

    @staticmethod
    def backward(ctx, grad_multihot_indices, grad_probs_in_multihot):
        position_map = ctx.saved_tensors[0]
        grad_probs_indices = torch.empty((ctx.num_of_tokens, ctx.topk), dtype=grad_probs_in_multihot.dtype, device="cuda")
        topk_next_power_of_2 = 2 ** int(math.ceil(math.log2(ctx.topk)))
        num_of_local_experts_next_power_of_2 = 2 ** int(math.ceil(math.log2(ctx.num_of_local_experts)))
        _multihot_to_indices_kernel[(ctx.num_of_tokens,)](
            grad_probs_in_multihot.contiguous(), position_map, grad_probs_indices,
            ctx.num_of_local_experts, num_of_local_experts_next_power_of_2, ctx.topk,
            topk_next_power_of_2, BLOCK_SIZE=32, num_warps=1)
        return None, grad_probs_indices, None


def fused_indices_to_multihot(indices, probs_indices, num_of_local_experts):
    return IndicesToMultihot.apply(indices, probs_indices, num_of_local_experts)


def _tokens_per_expert_tensor(tokens_per_expert, device):
    if isinstance(tokens_per_expert, torch.Tensor):
        return tokens_per_expert.to(device=device, dtype=torch.int32)
    return torch.tensor(tokens_per_expert, device=device, dtype=torch.int32)


def build_te_grouped_experts(W_gate, W_up, W_down, experts_per_rank):
    """Build TE baseline using raw te_ops.Sequential(GroupedLinear, ScaledSwiGLU, GroupedLinear)."""
    if te_ops is None:
        raise RuntimeError('Transformer Engine ops are required for the TE baseline')

    hidden = W_gate.shape[-1]
    intermediate = W_gate.shape[1]
    device = W_gate.device
    dtype = W_gate.dtype

    fc1 = te_ops.GroupedLinear(
        experts_per_rank, hidden, 2 * intermediate, bias=False,
        device=device, dtype=dtype,
        single_grouped_weight=True, single_grouped_bias=False,
        accumulate_into_main_grad=False, delay_wgrad_compute=False,
    )
    act = te_ops.ScaledSwiGLU(glu_interleave_size=32)
    fc2 = te_ops.GroupedLinear(
        experts_per_rank, intermediate, hidden, bias=False,
        device=device, dtype=dtype,
        single_grouped_weight=True, single_grouped_bias=False,
        accumulate_into_main_grad=False, delay_wgrad_compute=False,
    )

    # Pack weights: interleave gate/up in blocks of 32
    fc1_chunks = []
    for start in range(0, intermediate, 32):
        fc1_chunks.append(W_gate[:, start:start + 32, :])
        fc1_chunks.append(W_up[:, start:start + 32, :])
    fc1_weight = torch.cat(fc1_chunks, dim=1).contiguous()

    with torch.no_grad():
        if hasattr(fc1, 'weight'):
            fc1.weight.copy_(fc1_weight)
            fc2.weight.copy_(W_down.contiguous())
        else:
            for e in range(experts_per_rank):
                getattr(fc1, f'weight{e}').copy_(fc1_weight[e])
                getattr(fc2, f'weight{e}').copy_(W_down[e].contiguous())

    return te_ops.Sequential(fc1, act, fc2)


class MegatronFusedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, token_indices, token_probs, num_experts, buffer):
        layout = buffer.get_dispatch_layout(token_indices, num_experts)
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, _ = layout
        recv_x, recv_indices, recv_probs, tokens_per_expert, handle, _ = buffer.deepep_dispatch(
            x=x,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=token_indices,
            topk_weights=token_probs,
        )
        ctx.buffer = buffer
        ctx.handle = handle
        return recv_x, recv_indices, recv_probs, torch.tensor(tokens_per_expert), handle

    @staticmethod
    def backward(ctx, grad_x, _grad_indices, grad_probs, _grad_tokens_per_expert, _grad_handle):
        combined_x, combined_probs, _ = ctx.buffer.deepep_combine(
            grad_x.contiguous(), ctx.handle,
            topk_weights=None if grad_probs is None else grad_probs.float(),
        )
        return combined_x, None, combined_probs, None, None


class MegatronFusedCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, buffer, handle):
        output, _, _ = buffer.deepep_combine(x=x, handle=handle)
        ctx.buffer = buffer
        ctx.handle = handle
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_x, _, _, _, _, _ = ctx.buffer.deepep_dispatch(
            grad_output.contiguous(), handle=ctx.handle)
        return grad_x, None, None


def run_megatron_fused_baseline(x, topk_idx, topk_weights, num_experts,
                                 experts_per_rank, buffer, te_experts):
    recv_x, recv_idx, recv_probs, tokens_per_expert, handle = MegatronFusedDispatch.apply(
        x, topk_idx, topk_weights, num_experts, buffer)
    local_output = moe_compute_on_recv_te(
        recv_x, recv_idx, recv_probs, tokens_per_expert,
        te_experts, {}, experts_per_rank, use_fp8=False)
    return MegatronFusedCombine.apply(local_output, buffer, handle)


def start_memory_measurement():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def record_forward_memory(start_allocated):
    torch.cuda.synchronize()
    return max(0, torch.cuda.memory_allocated() - start_allocated)


def finish_memory_measurement(start_allocated, activation_retained, phase):
    torch.cuda.synchronize()
    peak_allocated = max(0, torch.cuda.max_memory_allocated() - start_allocated)
    peak_reserved = torch.cuda.max_memory_reserved()
    print(
        f'  [{phase} memory] forward activation retained: '
        f'{activation_retained / 1024 ** 2:.2f} MiB, '
        f'peak allocated increment (fwd+bwd): {peak_allocated / 1024 ** 2:.2f} MiB, '
        f'peak reserved (absolute): {peak_reserved / 1024 ** 2:.2f} MiB, '
        f'allocated before forward: {start_allocated / 1024 ** 2:.2f} MiB',
        flush=True,
    )
    return activation_retained, peak_allocated, peak_reserved


def report_memory_comparison(baseline_memory, teramoe_memory, baseline_name):
    names = ('forward activation retained', 'peak allocated increment (fwd+bwd)', 'peak reserved')
    print(f'  [Memory comparison] TeraMOE - {baseline_name}:', flush=True)
    for name, baseline_value, teramoe_value in zip(names, baseline_memory, teramoe_memory):
        delta = teramoe_value - baseline_value
        ratio = 100.0 * delta / baseline_value if baseline_value else float('nan')
        print(f'    {name}: {delta / 1024 ** 2:+.2f} MiB ({ratio:+.2f}%)', flush=True)


def clear_parameter_grads(module):
    for parameter in module.parameters():
        parameter.grad = None


def benchmark_cuda_events(fn, warmup_iters, repeat_iters, group=None):
    if warmup_iters < 0:
        raise ValueError(f'warmup_iters must be >= 0, got {warmup_iters}')
    if repeat_iters <= 0:
        raise ValueError(f'repeat_iters must be > 0, got {repeat_iters}')

    if group is not None:
        dist.barrier(group=group)
    torch.cuda.synchronize()

    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    if group is not None:
        dist.barrier(group=group)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(repeat_iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / repeat_iters


def get_te_grouped_expert_weight_grads(te_experts, experts_per_rank):
    """Extract fc1 and fc2 weight gradients from a te_ops.Sequential baseline."""
    fc1_op = te_experts[0]
    fc2_op = te_experts[2]

    if hasattr(fc1_op, 'weight'):
        # single_grouped_weight=True: one tensor per layer
        fc1_grad = fc1_op.weight.grad.detach()
        fc2_grad = fc2_op.weight.grad.detach()
    else:
        # single_grouped_weight=False: per-expert weight0, weight1, ...
        fc1_grads = [getattr(fc1_op, f'weight{e}').grad.detach() for e in range(experts_per_rank)]
        fc2_grads = [getattr(fc2_op, f'weight{e}').grad.detach() for e in range(experts_per_rank)]
        fc1_grad = torch.stack(fc1_grads, dim=0)
        fc2_grad = torch.stack(fc2_grads, dim=0)

    return fc1_grad, fc2_grad


def te_fp8_context(enabled):
    if not enabled:
        return contextlib.nullcontext()
    if fp8_autocast is None or te_recipe is None:
        raise RuntimeError('Transformer Engine FP8 support is required for the Megatron baseline')
    recipe = te_recipe.DelayedScaling(fp8_format=te_recipe.Format.HYBRID,
                                     amax_history_len=16, amax_compute_algo='max')
    return fp8_autocast(enabled=True, fp8_recipe=recipe)


def moe_compute_on_recv_te(recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list,
                           te_experts, workspace, experts_per_rank, use_fp8=False):
    if moe_permute_with_probs is None or moe_unpermute is None:
        raise RuntimeError('Transformer Engine MoE operators are required for the Megatron baseline')
    assert recv_topk_weights.dtype == torch.float32
    routing_map, probs_map = fused_indices_to_multihot(recv_topk_idx, recv_topk_weights, experts_per_rank)
    tokens_per_expert = _tokens_per_expert_tensor(recv_num_tokens_per_expert_list, recv_x.device)
    permuted_x, permuted_probs, row_map = moe_permute_with_probs(
        recv_x, probs_map, routing_map, num_out_tokens=tokens_per_expert.sum().item())
    with te_fp8_context(use_fp8):
        permuted_output = te_experts(permuted_x, tokens_per_expert, permuted_probs, tokens_per_expert)
    return moe_unpermute(permuted_output, row_map, restore_shape=recv_x.shape)


def megatron_group_limited_topk(scores, topk, num_groups, group_topk):
    num_tokens, num_experts = scores.shape
    if num_groups <= 0 or num_experts % num_groups != 0:
        raise ValueError(f'num_groups must divide num_experts, got num_groups={num_groups}, num_experts={num_experts}')
    if group_topk <= 0 or group_topk > num_groups:
        raise ValueError(f'group_topk must be in [1, num_groups], got group_topk={group_topk}, num_groups={num_groups}')
    if topk % group_topk != 0:
        raise ValueError(f'Megatron group_limited_topk requires topk % group_topk == 0, got topk={topk}, group_topk={group_topk}')
    group_scores = scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(
        num_tokens, num_groups, num_experts // num_groups).reshape(num_tokens, num_experts)
    return torch.topk(scores.masked_fill(~score_mask.bool(), float('-inf')), k=topk, dim=-1)


def make_megatron_router_inputs(num_tokens, num_experts, topk, num_groups, group_topk, score_function, device,
                                hotspot_expert_fraction=0.0, hotspot_expert_start=0, hotspot_logit_bias=0.0):
    if not 0.0 <= hotspot_expert_fraction <= 1.0:
        raise ValueError(
            f'hotspot_expert_fraction must be in [0, 1], got {hotspot_expert_fraction}')
    if hotspot_expert_fraction == 0.0:
        if hotspot_logit_bias != 0.0:
            raise ValueError('hotspot_logit_bias requires hotspot_expert_fraction > 0')
    elif hotspot_logit_bias == 0.0:
        raise ValueError('hotspot_expert_fraction requires a non-zero hotspot_logit_bias')

    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    routing_logits = logits
    if hotspot_expert_fraction > 0.0:
        hotspot_expert_count = math.ceil(num_experts * hotspot_expert_fraction)
        hotspot_expert_end = hotspot_expert_start + hotspot_expert_count
        if hotspot_expert_start < 0 or hotspot_expert_end > num_experts:
            raise ValueError(
                f'hotspot expert range [{hotspot_expert_start}, {hotspot_expert_end}) is outside '
                f'[0, {num_experts})')
        routing_logits = logits.clone()
        routing_logits[:, hotspot_expert_start:hotspot_expert_end] += hotspot_logit_bias

    if score_function == 'softmax':
        _, topk_idx = megatron_group_limited_topk(routing_logits, topk, num_groups, group_topk)
        topk_logits = logits.gather(1, topk_idx)
        topk_weights = torch.softmax(topk_logits, dim=-1, dtype=torch.float32)
    elif score_function in ('sigmoid', 'sqrtsoftplus'):
        scores = torch.sigmoid(logits) if score_function == 'sigmoid' else F.softplus(logits).sqrt()
        routing_scores = torch.sigmoid(routing_logits) if score_function == 'sigmoid' else F.softplus(routing_logits).sqrt()
        _, topk_idx = megatron_group_limited_topk(routing_scores, topk, num_groups, group_topk)
        topk_weights = scores.gather(1, topk_idx)
        if topk > 1:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f'Unsupported router score function: {score_function}')
    return topk_weights.contiguous(), topk_idx.to(teramoe.topk_idx_t).contiguous()
