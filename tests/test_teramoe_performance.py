"""TeraMOE performance and memory test harness.

The baseline is DeepEP + TE. The TeraMOE section autotunes
compute_batch_size x combine_start_head_percent and reports the fastest
measured forward/backward configuration. Memory is measured separately with
the default TeraMOE config.
"""

import argparse
import itertools
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Under mpirun: pin CUDA_VISIBLE_DEVICES before any CUDA context is created.
if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
    _local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ['CUDA_VISIBLE_DEVICES'] = _local_rank

import torch
import torch.distributed as dist

from utils import init_dist

import teramoe
from teramoe.autotune import (
    AutotuneResult,
    COMPUTE_BATCH_SIZES,
    COMBINE_START_HEAD_PERCENTS,
)

from teramoe_test_utils import (
    build_te_grouped_experts,
    clear_parameter_grads,
    make_megatron_router_inputs,
    record_forward_memory,
    report_memory_comparison,
    run_megatron_fused_baseline,
    start_memory_measurement,
    finish_memory_measurement,
)


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
    # test(num_tokens=4096, hidden=2048, intermediate=2048, experts_per_rank=16, num_topk=8),
    # test(num_tokens=32768, hidden=2048, intermediate=3072, experts_per_rank=4, num_topk=2),
    test(num_tokens=8192, hidden=2048, intermediate=3072, experts_per_rank=16, num_topk=6)
    # test(num_tokens=32768, hidden=2048, intermediate=3072, experts_per_rank=4, num_topk=6, num_topk_groups=3),
    # test(num_tokens=16384, hidden=2048, intermediate=3072, experts_per_rank=4, num_topk=6, num_topk_groups=3),
    # test(num_tokens=8192, hidden=2048, intermediate=3072, experts_per_rank=4, num_topk=6, num_topk_groups=3),
    # test(num_tokens=8192, hidden=4096, intermediate=4096, experts_per_rank=16, num_topk=8),
]


def prepare_case_inputs(case, rank, num_ranks, num_local_ranks, args, case_idx):
    seed_offset = case_idx * 1000003 + args.case_seed_offset
    torch.manual_seed(42 + rank + seed_offset)
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

    torch.manual_seed(1000 + rank + seed_offset)
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
    return x, topk_idx, topk_weights, W_gate, W_up, W_down, W_gateup, num_experts


def report_topk_peak_to_average(topk_idx, num_experts, rank, group):
    expert_loads = torch.bincount(topk_idx.reshape(-1).to(torch.long), minlength=num_experts)
    expert_loads = expert_loads.to(dtype=torch.float32)
    dist.all_reduce(expert_loads, group=group)

    if rank == 0:
        peak_load = expert_loads.max().item()
        average_load = expert_loads.mean().item()
        peak_to_average = peak_load / average_load if average_load else float('nan')
        print(
            f'  router top-k load: peak={peak_load:.0f}, average={average_load:.2f}, '
            f'peak_to_average={peak_to_average:.4f}',
            flush=True,
        )


def benchmark_forward_backward(forward, backward, warmup_iters, repeat_iters, group=None):
    if warmup_iters < 0:
        raise ValueError(f'warmup_iters must be >= 0, got {warmup_iters}')
    if repeat_iters <= 0:
        raise ValueError(f'repeat_iters must be > 0, got {repeat_iters}')

    if group is not None:
        dist.barrier(group=group)
    torch.cuda.synchronize()

    for _ in range(warmup_iters):
        backward(forward())
    torch.cuda.synchronize()

    if group is not None:
        dist.barrier(group=group)

    forward_events = []
    backward_events = []
    for _ in range(repeat_iters):
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)

        forward_start.record()
        output = forward()
        forward_end.record()
        backward_start.record()
        backward(output)
        backward_end.record()
        forward_events.append((forward_start, forward_end))
        backward_events.append((backward_start, backward_end))

    torch.cuda.synchronize()
    forward_ms = sum(start.elapsed_time(end) for start, end in forward_events) / repeat_iters
    backward_ms = sum(start.elapsed_time(end) for start, end in backward_events) / repeat_iters
    return forward_ms, backward_ms, forward_ms + backward_ms


def make_baseline_functions(x, topk_idx, topk_weights, grad_output, num_experts, experts_per_rank, buffer,
                            te_experts):
    def forward():
        clear_parameter_grads(te_experts)
        baseline_x = x.detach().clone().requires_grad_(True)
        baseline_topk_weights = topk_weights.detach().clone().requires_grad_(True)
        baseline_output = run_megatron_fused_baseline(
            baseline_x, topk_idx, baseline_topk_weights, num_experts,
            experts_per_rank, buffer, te_experts,
        )
        if not baseline_output.requires_grad:
            raise AssertionError('DeepEP + TE output is not connected to autograd')
        return baseline_output

    def backward(output):
        output.backward(grad_output)

    return forward, backward


def make_teramoe_functions(buffer, x, topk_idx, topk_weights, grad_output, num_experts, args,
                               W_gateup, W_down, num_sms, compute_batch_size, combine_start_head_percent):
    # Pre-allocate weight tensors outside the timed loop to avoid measuring
    # clone bandwidth that has no baseline equivalent.
    mk_w_gateup = W_gateup.detach().clone().requires_grad_(True)
    mk_w_down = W_down.detach().clone().requires_grad_(True)

    def forward():
        mk_x = x.detach().clone().requires_grad_(True)
        mk_topk_weights = topk_weights.detach().clone().requires_grad_(True)
        mk_w_gateup.grad = None
        mk_w_down.grad = None
        mk_output = buffer.teramoe_autograd(
            mk_x, topk_idx, mk_topk_weights, mk_w_gateup, mk_w_down, num_experts,
            num_dispatch_sms=args.teramoe_comm_sms,
            num_combine_sms=args.teramoe_comm_sms,
            total_sms=num_sms,
            stage=args.stage,
            compute_batch_size=compute_batch_size,
            combine_start_head_percent=combine_start_head_percent,
        )
        if not mk_output.requires_grad:
            raise AssertionError('TeraMOE output is not connected to autograd')
        return mk_output

    def backward(output):
        output.backward(grad_output)

    return forward, backward


def autotune_teramoe(buffer, x, topk_idx, topk_weights, W_gateup, W_down,
                        num_experts, args, num_sms, grad_output, rank,
                        group=None, verbose=True):
    best_time = float('inf')
    best_forward_ms = 0.0
    best_backward_ms = 0.0
    best_config = (COMPUTE_BATCH_SIZES[0], COMBINE_START_HEAD_PERCENTS[0])
    results: List[Tuple[int, int, float]] = []

    configs = list(itertools.product(COMPUTE_BATCH_SIZES, COMBINE_START_HEAD_PERCENTS))
    total_configs = len(configs)

    for config_idx, (batch_size, percent) in enumerate(configs):
        if verbose and rank == 0:
            print(f'  [autotune] [{config_idx + 1}/{total_configs}] '
                  f'Starting: compute_batch_size={batch_size}, combine_start_head_percent={percent}%',
                  flush=True)

        forward, backward = make_teramoe_functions(
            buffer, x, topk_idx, topk_weights, grad_output,
            num_experts, args, W_gateup, W_down, num_sms, batch_size, percent,
        )
        forward_ms, backward_ms, total_ms = benchmark_forward_backward(
            forward, backward, args.warmup, args.repeat, group=group,
        )

        results.append((batch_size, percent, total_ms))

        if verbose and rank == 0:
            print(f'  [autotune] [{config_idx + 1}/{total_configs}] '
                  f'compute_batch_size={batch_size:4d}, '
                  f'combine_start_head_percent={percent:2d}% -> '
                  f'forward={forward_ms:.4f} ms, backward={backward_ms:.4f} ms, '
                  f'total={total_ms:.4f} ms/iter', flush=True)

        if total_ms < best_time:
            best_time = total_ms
            best_forward_ms = forward_ms
            best_backward_ms = backward_ms
            best_config = (batch_size, percent)

        if group is not None:
            dist.barrier(group=group)

    if not results:
        raise RuntimeError('Autotune did not run any configurations')

    if verbose and rank == 0:
        print(f'  [autotune] BEST: compute_batch_size={best_config[0]}, '
              f'combine_start_head_percent={best_config[1]}%, time={best_time:.4f} ms/iter',
              flush=True)

    best_result = AutotuneResult(
        compute_batch_size=best_config[0],
        combine_start_head_percent=best_config[1],
        time_ms=best_time,
        forward_ms=best_forward_ms,
        backward_ms=best_backward_ms,
    )
    return best_result, results


def run_case(local_rank, num_local_ranks, rank, num_ranks, buffer, group, args, case, case_idx):
    if args.stage not in (1, 2):
        raise ValueError('teramoe debug backward currently supports stage 1 or 2')
    if args.warmup < 0:
        raise ValueError(f'--warmup must be >= 0, got {args.warmup}')
    if args.repeat <= 0:
        raise ValueError(f'--repeat must be > 0, got {args.repeat}')

    x, topk_idx, topk_weights, W_gate, W_up, W_down, W_gateup, num_experts = prepare_case_inputs(
        case, rank, num_ranks, num_local_ranks, args, case_idx,
    )
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    if rank == 0:
        print('', flush=True)
        print(f'=== Performance case {case_idx + 1} ===', flush=True)
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
        print(
            f'  autotune search space: compute_batch_size={COMPUTE_BATCH_SIZES}, '
            f'combine_start_head_percent={COMBINE_START_HEAD_PERCENTS}',
            flush=True,
        )

    report_topk_peak_to_average(topk_idx, num_experts, rank, group)

    buffer.set_num_sms(args.baseline_sms)
    te_experts = build_te_grouped_experts(
        W_gate.detach(), W_up.detach(), W_down.detach(), case.experts_per_rank)
    for parameter in te_experts.parameters():
        parameter.requires_grad_(True)

    torch.manual_seed(2000 + rank + case_idx * 1000003 + args.case_seed_offset)
    grad_output = torch.randn_like(x)

    if rank == 0:
        print('  [perf] Benchmarking DeepEP + TE baseline...', flush=True)
    baseline_forward, baseline_backward = make_baseline_functions(
        x, topk_idx, topk_weights, grad_output,
        num_experts, case.experts_per_rank, buffer, te_experts,
    )
    baseline_forward_ms, baseline_backward_ms, baseline_time_ms = benchmark_forward_backward(
        baseline_forward, baseline_backward, args.warmup, args.repeat, group=group)
    if rank == 0:
        print(
            f'  [perf] DeepEP + TE baseline: forward={baseline_forward_ms:.4f} ms, '
            f'backward={baseline_backward_ms:.4f} ms, total={baseline_time_ms:.4f} ms/iter',
            flush=True,
        )
    clear_parameter_grads(te_experts)

    dist.barrier(group=group)
    torch.cuda.synchronize()

    if rank == 0:
        print('  [perf] Autotuning teramoe...', flush=True)
    best_result, _all_results = autotune_teramoe(
        buffer, x, topk_idx, topk_weights, W_gateup, W_down,
        num_experts=num_experts,
        args=args,
        num_sms=num_sms,
        grad_output=grad_output,
        rank=rank,
        group=group,
        verbose=True,
    )
    teramoe_forward_ms = best_result.forward_ms
    teramoe_backward_ms = best_result.backward_ms
    teramoe_time_ms = best_result.time_ms
    if rank == 0:
        print(
            f'  [perf] TeraMOE best autotune result: '
            f'compute_batch_size={best_result.compute_batch_size}, '
            f'combine_start_head_percent={best_result.combine_start_head_percent}%, '
            f'forward={teramoe_forward_ms:.4f} ms, '
            f'backward={teramoe_backward_ms:.4f} ms, total={teramoe_time_ms:.4f} ms/iter',
            flush=True,
        )

    dist.barrier(group=group)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    dist.barrier(group=group)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    baseline_name = 'DeepEP + TE'
    if rank == 0:
        print('  [memory] Measuring DeepEP + TE baseline...', flush=True)
    clear_parameter_grads(te_experts)
    baseline_mem_start = start_memory_measurement()
    baseline_x = x.detach().clone().requires_grad_(True)
    baseline_topk_weights = topk_weights.detach().clone().requires_grad_(True)
    baseline_output = run_megatron_fused_baseline(
        baseline_x, topk_idx, baseline_topk_weights, num_experts,
        case.experts_per_rank, buffer, te_experts,
    )
    if not baseline_output.requires_grad:
        raise AssertionError('DeepEP + TE output is not connected to autograd')
    baseline_activation_retained = record_forward_memory(baseline_mem_start)
    baseline_output.backward(grad_output)
    baseline_memory = finish_memory_measurement(
        baseline_mem_start, baseline_activation_retained, baseline_name)
    clear_parameter_grads(te_experts)

    dist.barrier(group=group)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    if rank == 0:
        print(
            f'  [memory] Measuring teramoe with default config '
            f'(compute_batch_size={args.compute_batch_size}, '
            f'combine_start_head_percent={args.combine_start_head_percent})...',
            flush=True,
        )
    # Pre-allocate weight clones outside the measurement interval so that
    # weight memory does not pollute the activation/peak comparison.
    teramoe_w_gateup = W_gateup.detach().clone().requires_grad_(True)
    teramoe_w_down = W_down.detach().clone().requires_grad_(True)
    teramoe_mem_start = start_memory_measurement()
    teramoe_x = x.detach().clone().requires_grad_(True)
    teramoe_topk_weights = topk_weights.detach().clone().requires_grad_(True)
    teramoe_output = buffer.teramoe_autograd(
        teramoe_x, topk_idx, teramoe_topk_weights, teramoe_w_gateup, teramoe_w_down,
        num_experts,
        num_dispatch_sms=args.teramoe_comm_sms,
        num_combine_sms=args.teramoe_comm_sms,
        total_sms=num_sms,
        stage=args.stage,
        compute_batch_size=args.compute_batch_size,
        combine_start_head_percent=args.combine_start_head_percent,
    )
    if not teramoe_output.requires_grad:
        raise AssertionError('TeraMOE output is not connected to autograd')
    teramoe_activation_retained = record_forward_memory(teramoe_mem_start)
    teramoe_output.backward(grad_output)
    teramoe_memory = finish_memory_measurement(
        teramoe_mem_start, teramoe_activation_retained, 'TeraMOE')
    if rank == 0:
        report_memory_comparison(baseline_memory, teramoe_memory, baseline_name)
        print(
            f'  [summary] case {case_idx + 1}: '
            f'baseline(fwd={baseline_forward_ms:.4f}, bwd={baseline_backward_ms:.4f}, '
            f'total={baseline_time_ms:.4f}) ms/iter, '
            f'teramoe(best, fwd={teramoe_forward_ms:.4f}, '
            f'bwd={teramoe_backward_ms:.4f}, total={teramoe_time_ms:.4f}) ms/iter',
            flush=True,
        )


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
    except Exception:
        buffer.destroy()
        dist.destroy_process_group()
        raise
    else:
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def run_mpirun(args):
    global_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    local_world_size = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', 8))

    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29502')
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
    except Exception:
        buffer.destroy()
        dist.destroy_process_group()
        raise
    else:
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure and autotune teramoe performance and memory against DeepEP + TE')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--warmup', '--warmup-iters', dest='warmup', type=int, default=10,
                        help='Warmup iterations before each timed benchmark')
    parser.add_argument('--repeat', '--num-iters', dest='repeat', type=int, default=1000,
                        help='Timed iterations for each benchmark')
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
    parser.add_argument('--compute-batch-size', type=int, default=4096,
                        choices=[1024, 2048, 4096],
                        help='Default teramoe compute batch size used for the memory test')
    parser.add_argument('--combine-start-head-percent', type=int, default=70,
                        help='Default teramoe combine threshold used for the memory test')
    parser.add_argument('--case-seed-offset', type=int, default=0,
                        help='Extra seed offset for repeated benchmark runs')
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
