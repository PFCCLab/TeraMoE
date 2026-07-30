"""Autotune compute_batch_size and combine_start_head_percent for TeraMOE.

Usage:
    best, all_results = autotune_teramoe(buffer, x, topk_idx, topk_weights,
                                            W_gateup, W_down, num_experts, ...)
    # best = AutotuneResult(compute_batch_size=2048, combine_start_head_percent=60, time_ms=1.23)
    # all_results = [(1024, 40, 1.50), (1024, 50, 1.45), ...]
"""

import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from .buffer import Buffer

COMPUTE_BATCH_SIZES = [1024, 2048, 4096]
COMBINE_START_HEAD_PERCENTS = [40, 50, 60, 70, 80, 90]


def _run_autotune_teramoe_forward(
    buffer: Buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    W_gateup: torch.Tensor,
    W_down: torch.Tensor,
    num_experts: int,
    num_dispatch_sms: int,
    num_combine_sms: int,
    total_sms: int,
    stage: int,
    batch_size: int,
    percent: int,
) -> torch.Tensor:
    return buffer.teramoe_autograd(
        x, topk_idx, topk_weights, W_gateup, W_down,
        num_experts,
        num_dispatch_sms=num_dispatch_sms,
        num_combine_sms=num_combine_sms,
        total_sms=total_sms,
        stage=stage,
        compute_batch_size=batch_size,
        combine_start_head_percent=percent,
    )


@dataclass
class AutotuneResult:
    compute_batch_size: int
    combine_start_head_percent: int
    time_ms: float
    forward_ms: Optional[float] = None
    backward_ms: Optional[float] = None


def autotune_teramoe(
    buffer: Buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    W_gateup: torch.Tensor,
    W_down: torch.Tensor,
    num_experts: int,
    num_dispatch_sms: int = 24,
    num_combine_sms: int = 24,
    total_sms: int = 148,
    stage: int = 1,
    num_iters: int = 1000,
    warmup_iters: int = 10,
    verbose: bool = True,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[AutotuneResult, List[Tuple[int, int, float]]]:
    """Grid-search over (compute_batch_size, combine_start_head_percent) and return the fastest combo.

    Each configuration is warmed up for `warmup_iters` iterations, then timed
    over `num_iters` iterations using CUDA events for accurate GPU-side timing.

    Args:
        buffer: DeepEP Buffer instance (already initialized).
        x, topk_idx, topk_weights, W_gateup, W_down, num_experts: TeraMOE inputs.
        num_dispatch_sms, num_combine_sms, total_sms, stage: TeraMOE launch parameters.
        num_iters: Number of timed iterations per configuration (default 1000).
        warmup_iters: Warmup iterations before timing (default 10).
        verbose: Print per-config timing results.
        group: Optional process group for distributed barrier synchronization.

    Returns:
        Tuple of (AutotuneResult with best config, list of all (batch_size, percent, avg_ms) results).
    """
    best_time = float('inf')
    best_config = (COMPUTE_BATCH_SIZES[0], COMBINE_START_HEAD_PERCENTS[0])
    results: List[Tuple[int, int, float]] = []

    configs = list(itertools.product(COMPUTE_BATCH_SIZES, COMBINE_START_HEAD_PERCENTS))
    total_configs = len(configs)

    for config_idx, (batch_size, percent) in enumerate(configs):
        if verbose:
            print(f'  [autotune] [{config_idx+1}/{total_configs}] '
                  f'Starting: compute_batch_size={batch_size}, combine_start_head_percent={percent}%',
                  flush=True)

        # Synchronize all ranks before each config to ensure consistent progress
        if group is not None:
            dist.barrier(group=group)
        torch.cuda.synchronize()

        # Warmup
        for wi in range(warmup_iters):
            _run_autotune_teramoe_forward(
                buffer, x, topk_idx, topk_weights, W_gateup, W_down,
                num_experts,
                num_dispatch_sms=num_dispatch_sms,
                num_combine_sms=num_combine_sms,
                total_sms=total_sms,
                stage=stage,
                batch_size=batch_size,
                percent=percent,
            )
        torch.cuda.synchronize()

        # Synchronize after warmup before timed region
        if group is not None:
            dist.barrier(group=group)

        # Timed iterations
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for ti in range(num_iters):
            _run_autotune_teramoe_forward(
                buffer, x, topk_idx, topk_weights, W_gateup, W_down,
                num_experts,
                num_dispatch_sms=num_dispatch_sms,
                num_combine_sms=num_combine_sms,
                total_sms=total_sms,
                stage=stage,
                batch_size=batch_size,
                percent=percent,
            )
        end_event.record()
        torch.cuda.synchronize()

        if verbose:
            print(f'  [autotune] [{config_idx+1}/{total_configs}] '
                  f'batch_size={batch_size}, percent={percent}% timed {num_iters} iters done',
                  flush=True)

        elapsed_ms = start_event.elapsed_time(end_event)
        avg_ms = elapsed_ms / num_iters
        results.append((batch_size, percent, avg_ms))

        if verbose:
            print(f'  [autotune] [{config_idx+1}/{total_configs}] '
                  f'compute_batch_size={batch_size:4d}, '
                  f'combine_start_head_percent={percent:2d}% -> {avg_ms:.4f} ms/iter',
                  flush=True)

        if avg_ms < best_time:
            best_time = avg_ms
            best_config = (batch_size, percent)

    if verbose:
        print(f'  [autotune] BEST: compute_batch_size={best_config[0]}, '
              f'combine_start_head_percent={best_config[1]}%, time={best_time:.4f} ms/iter',
              flush=True)

    best_result = AutotuneResult(
        compute_batch_size=best_config[0],
        combine_start_head_percent=best_config[1],
        time_ms=best_time,
    )
    return best_result, results

