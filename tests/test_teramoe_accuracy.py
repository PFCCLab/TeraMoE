"""TeraMOE precision test against the DeepEP + TE baseline."""

import argparse
import os
from dataclasses import dataclass
from typing import Optional

# Under mpirun each process must bind its own GPU BEFORE anything creates a CUDA
# context. Module-level imports below (torch, deep_ep, and teramoe_test_utils
# which calls torch.cuda.is_available() at import) can initialize CUDA on
# physical device 0 for every local rank, which later makes NCCL raise
# "Duplicate GPU detected". Pin CUDA_VISIBLE_DEVICES from the MPI local-rank
# here, before those imports run.
# NOTE: use unconditional assignment (not setdefault) because mpirun may
# propagate an empty-string CUDA_VISIBLE_DEVICES from the launch shell,
# and setdefault treats any existing key (even '') as already set.
if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
    _local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = _local_rank

import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils import calc_diff, init_dist

from teramoe_test_utils import (
    build_te_grouped_experts,
    clear_parameter_grads,
    get_te_grouped_expert_weight_grads,
    make_megatron_router_inputs,
    run_megatron_fused_baseline,
)

import teramoe


@dataclass(frozen=True)
class TestCase:
    num_tokens: int = 4096
    hidden: int = 2048
    intermediate: int = 2048
    experts_per_rank: int = 16
    num_topk: int = 8
    num_topk_groups: Optional[int] = None


def test(**kwargs):
    return TestCase(**kwargs)


TEST_CASES = [
    test(num_tokens=4096, hidden=2048, intermediate=2048, experts_per_rank=16, num_topk=8),
    # test(num_tokens=4096, hidden=2048, intermediate=3072, experts_per_rank=8, num_topk=6, num_topk_groups=3),
    # test(num_tokens=8192, hidden=4096, intermediate=4096, experts_per_rank=16, num_topk=8),
]


def compare_tensor(name, baseline, actual, rank, max_abs_tol, calc_diff_tol, cos_tol):
    baseline = baseline.to(device=actual.device)
    diff = calc_diff(baseline, actual)
    max_abs_diff = (baseline.float() - actual.float()).abs().max().item()
    cos_sim = F.cosine_similarity(
        baseline.float().flatten().unsqueeze(0),
        actual.float().flatten().unsqueeze(0),
    ).item()
    passed = max_abs_diff <= max_abs_tol and diff <= calc_diff_tol and cos_sim >= cos_tol

    if not passed:

        print(f'[Rank {rank}] === TeraMOE {name} vs DeepEP + TE Precision Alignment ===', flush=True)
        print(f'  calc_diff (lower=better): {diff:.6e} (tol={calc_diff_tol:.1e})', flush=True)
        print(f'  max_abs_diff: {max_abs_diff:.6e} (tol={max_abs_tol:.1e})', flush=True)
        print(f'  cosine_similarity: {cos_sim:.6f} (tol={cos_tol:.6f})', flush=True)
        print(f'  baseline norm: {baseline.float().norm().item():.6e}', flush=True)
        print(f'  teramoe norm: {actual.float().norm().item():.6e}', flush=True)

        raise AssertionError(
            f'{name} mismatch on rank {rank}: calc_diff={diff}, '
            f'max_abs_diff={max_abs_diff}, cosine_similarity={cos_sim}'
        )


def pack_te_grouped_gateup_gradients(fc1_grad, gate_template, up_template):
    gate_grad = torch.empty_like(gate_template)
    up_grad = torch.empty_like(up_template)
    chunk_base = 0
    intermediate = gate_template.shape[1]

    for start in range(0, intermediate, 32):
        rows = min(32, intermediate - start)
        gate_grad[:, start:start + rows, :] = fc1_grad[:, chunk_base:chunk_base + rows, :]
        up_grad[:, start:start + rows, :] = fc1_grad[:, chunk_base + rows:chunk_base + 2 * rows, :]
        chunk_base += 2 * rows

    gateup_grad = torch.empty(
        gate_template.shape[0], 2 * intermediate, gate_template.shape[2],
        dtype=gate_template.dtype, device=gate_template.device,
    )
    gateup_grad[:, 0::2, :] = gate_grad
    gateup_grad[:, 1::2, :] = up_grad
    return gateup_grad.contiguous()


def run_te_baseline(x, topk_idx, topk_weights, grad_output, num_experts, experts_per_rank, buffer, W_gate, W_up, W_down):
    te_experts = build_te_grouped_experts(W_gate.detach(), W_up.detach(), W_down.detach(), experts_per_rank)
    for parameter in te_experts.parameters():
        parameter.requires_grad_(True)

    baseline_x = x.detach().clone().requires_grad_(True)
    baseline_topk_weights = topk_weights.detach().clone().requires_grad_(True)
    baseline_output = run_megatron_fused_baseline(
        baseline_x, topk_idx, baseline_topk_weights, num_experts,
        experts_per_rank, buffer, te_experts,
    )
    if not baseline_output.requires_grad:
        raise AssertionError('DeepEP + TE output is not connected to autograd')
    baseline_output.backward(grad_output)

    if baseline_x.grad is None:
        raise AssertionError('DeepEP + TE backward did not return dX')
    if baseline_topk_weights.grad is None:
        raise AssertionError('DeepEP + TE backward did not return dTopKWeights')

    baseline_fc1_grad, baseline_fc2_grad = get_te_grouped_expert_weight_grads(te_experts, experts_per_rank)
    baseline_grad_w_gateup = pack_te_grouped_gateup_gradients(baseline_fc1_grad, W_gate, W_up)
    baseline_grad_x = baseline_x.grad.detach()
    baseline_grad_topk_weights = baseline_topk_weights.grad.detach()

    reference = {
        'output': baseline_output.detach(),
        'grad_x': baseline_grad_x,
        'grad_topk_weights': baseline_grad_topk_weights,
        'grad_w_gateup': baseline_grad_w_gateup,
        'grad_w_down': baseline_fc2_grad.detach(),
    }

    clear_parameter_grads(te_experts)
    return te_experts, reference


def run_teramoe_once(buffer, x, topk_idx, topk_weights, grad_output, num_experts,
                        args, W_gateup, W_down, num_sms):
    mk_x = x.detach().clone().requires_grad_(True)
    mk_topk_weights = topk_weights.detach().clone().requires_grad_(True)
    mk_w_gateup = W_gateup.detach().clone().requires_grad_(True)
    mk_w_down = W_down.detach().clone().requires_grad_(True)

    mk_output = buffer.teramoe_autograd(
        mk_x, topk_idx, mk_topk_weights, mk_w_gateup, mk_w_down, num_experts,
        num_dispatch_sms=args.teramoe_comm_sms,
        num_combine_sms=args.teramoe_comm_sms,
        total_sms=num_sms,
        stage=args.stage,
        compute_batch_size=args.compute_batch_size,
        combine_start_head_percent=args.combine_start_head_percent,
    )
    if not mk_output.requires_grad:
        raise AssertionError('TeraMOE output is not connected to autograd')
    mk_output.backward(grad_output)

    if mk_x.grad is None:
        raise AssertionError('TeraMOE backward did not return dX')
    if mk_w_gateup.grad is None or mk_w_down.grad is None:
        raise AssertionError('TeraMOE backward did not return expert weight gradients')
    if mk_topk_weights.grad is None:
        raise AssertionError('TeraMOE backward did not return dTopKWeights')

    return {
        'output': mk_output.detach(),
        'grad_x': mk_x.grad.detach(),
        'grad_topk_weights': mk_topk_weights.grad.detach(),
        'grad_w_gateup': mk_w_gateup.grad.detach(),
        'grad_w_down': mk_w_down.grad.detach(),
    }


def run_case(local_rank, num_local_ranks, rank, num_ranks, buffer, group, args, case, case_idx):
    if args.stage not in (1, 2):
        raise ValueError('teramoe debug backward currently supports stage 1 or 2')
    if args.repeat <= 0:
        raise ValueError(f'--repeat must be positive, got {args.repeat}')

    torch.manual_seed(42 + rank + case_idx * 1000003)
    num_nodes = max(1, num_ranks // num_local_ranks)
    num_experts = num_ranks * case.experts_per_rank
    router_num_groups = args.router_num_groups or num_nodes
    router_group_topk = args.router_group_topk or (
        num_nodes if case.num_topk_groups is None else case.num_topk_groups
    )

    x = torch.randn(case.num_tokens, case.hidden, dtype=torch.bfloat16, device='cuda') * 0.1
    topk_weights, topk_idx = make_megatron_router_inputs(
        case.num_tokens,
        num_experts,
        case.num_topk,
        router_num_groups,
        router_group_topk,
        args.router_score_function,
        'cuda',
        hotspot_expert_fraction=args.router_hotspot_expert_fraction,
        hotspot_expert_start=args.router_hotspot_expert_start,
        hotspot_logit_bias=args.router_hotspot_logit_bias,
    )

    torch.manual_seed(1000 + rank)
    W_gate = torch.randn(
        case.experts_per_rank, case.intermediate, case.hidden,
        dtype=torch.bfloat16, device='cuda') * 0.02
    W_up = torch.randn_like(W_gate) * 0.02
    W_down = torch.randn(
        case.experts_per_rank, case.hidden, case.intermediate,
        dtype=torch.bfloat16, device='cuda') * 0.02
    W_gateup = torch.empty(
        case.experts_per_rank, 2 * case.intermediate, case.hidden,
        dtype=torch.bfloat16, device='cuda')
    W_gateup[:, 0::2, :] = W_gate
    W_gateup[:, 1::2, :] = W_up
    W_gateup = W_gateup.contiguous()

    if rank == 0:
        print('', flush=True)
        print(f'=== Precision case {case_idx + 1} ===', flush=True)
        print(
            f'  tokens={case.num_tokens}, hidden={case.hidden}, intermediate={case.intermediate}, '
            f'experts_per_rank={case.experts_per_rank}, topk={case.num_topk}, ranks={num_ranks}',
            flush=True,
        )
        print(
            f'  warmup={args.warmup}, repeat={args.repeat}, stage={args.stage}, '
            f'baseline_sms={args.baseline_sms}, teramoe_comm_sms={args.teramoe_comm_sms}',
            flush=True,
        )
        print(
            f'  router hotspot: expert_fraction={args.router_hotspot_expert_fraction}, '
            f'expert_start={args.router_hotspot_expert_start}, '
            f'logit_bias={args.router_hotspot_logit_bias}',
            flush=True,
        )

    torch.manual_seed(2000 + rank + case_idx * 1000003)
    grad_output = torch.randn_like(x)

    buffer.set_num_sms(args.baseline_sms)
    baseline_reference = None
    if not args.skip_baseline:
        te_experts, baseline_reference = run_te_baseline(
            x, topk_idx, topk_weights, grad_output,
            num_experts, case.experts_per_rank, buffer, W_gate, W_up, W_down,
        )
        clear_parameter_grads(te_experts)
    else:
        te_experts = None

    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    def compare_against_baseline(tag, result):
        compare_tensor(
            f'{tag} forward', baseline_reference['output'], result['output'], rank,
            args.forward_max_abs_tol, args.forward_calc_diff_tol, args.forward_cos_tol,
        )
        compare_tensor(
            f'{tag} Backward dX', baseline_reference['grad_x'], result['grad_x'], rank,
            args.backward_max_abs_tol, args.backward_calc_diff_tol, args.backward_cos_tol,
        )
        compare_tensor(
            f'{tag} Backward dW_gateup', baseline_reference['grad_w_gateup'], result['grad_w_gateup'], rank,
            args.backward_max_abs_tol, args.backward_calc_diff_tol, args.backward_cos_tol,
        )
        compare_tensor(
            f'{tag} Backward dW_down', baseline_reference['grad_w_down'], result['grad_w_down'], rank,
            args.backward_max_abs_tol, args.backward_calc_diff_tol, args.backward_cos_tol,
        )
        compare_tensor(
            f'{tag} Backward dTopKWeights', baseline_reference['grad_topk_weights'], result['grad_topk_weights'], rank,
            args.backward_max_abs_tol, args.backward_calc_diff_tol, args.backward_cos_tol,
        )

    for w in range(args.warmup):
        if local_rank == 0:
            print(f'[Rank {rank}] TeraMOE warmup {w + 1}/{args.warmup}', flush=True)
        warmup_result = run_teramoe_once(
            buffer, x, topk_idx, topk_weights, grad_output, num_experts,
            args, W_gateup, W_down, num_sms,
        )
        if args.check_warmup_precision:
            compare_against_baseline(f'warmup[{w + 1}]', warmup_result)
        dist.barrier(group=group)
        torch.cuda.synchronize()

    if args.warmup > 0:
        dist.barrier(group=group)
        torch.cuda.synchronize()

    for repeat_idx in range(args.repeat):
        if local_rank == 0:
            print(f'[Rank {rank}] TeraMOE repeat {repeat_idx + 1}/{args.repeat}', flush=True)
        result = run_teramoe_once(
            buffer, x, topk_idx, topk_weights, grad_output, num_experts,
            args, W_gateup, W_down, num_sms,
        )
        compare_against_baseline(f'repeat[{repeat_idx + 1}]', result)
        dist.barrier(group=group)
        torch.cuda.synchronize()

    if baseline_reference is not None:
        print(f'[Rank {rank}] Precision checks passed for case {case_idx + 1}', flush=True)


def run_worker(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    buffer = teramoe.Buffer(
        group, int(2e9), int(1e9), low_latency_mode=False,
        num_qps_per_rank=num_sms, explicitly_destroy=True,
    )
    try:
        for case_idx, case in enumerate(TEST_CASES):
            run_case(
                local_rank, num_local_ranks, rank, num_ranks,
                buffer, group, args, case, case_idx,
            )
    finally:
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def run_mpirun(args):
    global_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    local_world_size = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', 8))

    # CUDA_VISIBLE_DEVICES is already pinned to this local rank at module import
    # (see top of file) so the CUDA context binds to the correct physical GPU.
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29501')
    os.environ['RANK'] = str(global_rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(backend='nccl')
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(0)
    group = dist.new_group(list(range(world_size)))
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    buffer = teramoe.Buffer(
        group, int(2e9), int(1e9), low_latency_mode=False,
        num_qps_per_rank=num_sms, explicitly_destroy=True,
    )
    try:
        for case_idx, case in enumerate(TEST_CASES):
            run_case(
                local_rank, local_world_size, global_rank, world_size,
                buffer, group, args, case, case_idx,
            )
    finally:
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='Compare teramoe forward/backward with DeepEP + TE')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=100,
                        help='Number of warmup iterations for the teramoe before the measured repeats')
    parser.add_argument('--repeat', type=int, default=10000,
                        help='Number of measured teramoe repeats to compare against the baseline')
    parser.add_argument('--skip-baseline', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--check-warmup-precision', action='store_true',
                        help='Compare the teramoe warmup iterations against the baseline reference')
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--baseline-sms', type=int, default=48)
    parser.add_argument('--teramoe-comm-sms', type=int, default=48)
    parser.add_argument(
        '--router-score-function', choices=['sigmoid', 'softmax', 'sqrtsoftplus'],
        default='sigmoid')
    parser.add_argument('--router-num-groups', type=int, default=0)
    parser.add_argument('--router-group-topk', type=int, default=0)
    parser.add_argument('--router-hotspot-expert-fraction', type=float, default=0.0,
                        help='Fraction of consecutive experts biased as a communication hotspot')
    parser.add_argument('--router-hotspot-expert-start', type=int, default=0,
                        help='First expert id in the hotspot range')
    parser.add_argument('--router-hotspot-logit-bias', type=float, default=0.0,
                        help='Logit bias added to hotspot experts before top-k selection')
    parser.add_argument('--forward-max-abs-tol', type=float, default=1e-2)
    parser.add_argument('--forward-calc-diff-tol', type=float, default=1e-5)
    parser.add_argument('--forward-cos-tol', type=float, default=0.99)
    parser.add_argument('--backward-max-abs-tol', type=float, default=1e-2)
    parser.add_argument('--backward-calc-diff-tol', type=float, default=1e-5)
    parser.add_argument('--backward-cos-tol', type=float, default=0.99)
    parser.add_argument('--compute-batch-size', type=int, default=4096,
                        choices=[1024, 2048, 4096],
                        help='TeraMOE compute batch size per expert')
    parser.add_argument('--combine-start-head-percent', type=int, default=70,
                        help='TeraMOE combine SM start threshold percentage')
    parser.add_argument('--mpirun', action='store_true')
    args = parser.parse_args()

    if args.warmup < 0:
        raise ValueError(f'--warmup must be >= 0, got {args.warmup}')
    if args.repeat <= 0:
        raise ValueError(f'--repeat must be > 0, got {args.repeat}')
    if args.stage not in (1, 2):
        raise ValueError(f'--stage must be 1 or 2, got {args.stage}')
    return args


if __name__ == '__main__':
    parsed_args = parse_args()
    if parsed_args.mpirun:
        run_mpirun(parsed_args)
    else:
        torch.multiprocessing.spawn(
            run_worker,
            args=(parsed_args.num_processes, parsed_args),
            nprocs=parsed_args.num_processes,
            join=True,
        )
