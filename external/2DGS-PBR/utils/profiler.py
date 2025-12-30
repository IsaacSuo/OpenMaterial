"""
Simple Profiler for 2DGS-PBR Training

Usage:
    profiler = SimpleProfiler(enabled=True)

    with profiler.profile("render"):
        render_pkg = render(...)

    if iteration % 500 == 0:
        profiler.report(iteration)

Cleanup:
    - Delete this file
    - Remove `with profiler.profile(...)` wrappers in train_pbr.py
    - Or just don't pass --profile flag
"""

import torch
import time
from contextlib import contextmanager
from collections import defaultdict


class SimpleProfiler:
    """
    Lightweight profiler with easy enable/disable.

    When disabled, the context manager is a no-op with minimal overhead.
    """

    def __init__(self, enabled=False, sync_cuda=True):
        """
        Args:
            enabled: Whether profiling is active
            sync_cuda: Whether to synchronize CUDA before timing (recommended)
        """
        self.enabled = enabled
        self.sync_cuda = sync_cuda
        self.times = defaultdict(list)
        self._iteration_start = None

    @contextmanager
    def profile(self, name: str):
        """Profile a code block."""
        if not self.enabled:
            yield
            return

        if self.sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        yield

        if self.sync_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        self.times[name].append(elapsed)

    def start_iteration(self):
        """Mark start of iteration for total time tracking."""
        if not self.enabled:
            return
        if self.sync_cuda:
            torch.cuda.synchronize()
        self._iteration_start = time.perf_counter()

    def end_iteration(self):
        """Mark end of iteration."""
        if not self.enabled or self._iteration_start is None:
            return
        if self.sync_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._iteration_start
        self.times["_total_iteration"].append(elapsed)
        self._iteration_start = None

    def report(self, iteration: int, last_n: int = 100):
        """
        Print profiling report.

        Args:
            iteration: Current iteration number
            last_n: Use last N samples for averaging
        """
        if not self.enabled or not self.times:
            return

        print(f"\n{'='*50}")
        print(f"[Profile @ iter {iteration}] (avg of last {last_n} samples)")
        print(f"{'='*50}")

        # Sort by average time (descending)
        items = []
        for name, times in self.times.items():
            recent = times[-last_n:] if len(times) > last_n else times
            avg_ms = sum(recent) / len(recent) * 1000
            items.append((name, avg_ms, len(recent)))

        items.sort(key=lambda x: -x[1] if not x[0].startswith('_') else -999)

        total_ms = 0
        for name, avg_ms, count in items:
            if name == "_total_iteration":
                continue
            total_ms += avg_ms
            print(f"  {name:30s}: {avg_ms:8.2f} ms")

        # Print total iteration time if available
        for name, avg_ms, count in items:
            if name == "_total_iteration":
                print(f"  {'-'*42}")
                print(f"  {'_total_iteration':30s}: {avg_ms:8.2f} ms")
                print(f"  {'_profiled_sum':30s}: {total_ms:8.2f} ms")
                overhead = avg_ms - total_ms
                print(f"  {'_untracked/overhead':30s}: {overhead:8.2f} ms")

        print(f"{'='*50}\n")

    def clear(self):
        """Clear all recorded times."""
        self.times.clear()
