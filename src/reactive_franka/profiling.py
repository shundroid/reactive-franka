"""Named wall-time measurements. Stage names are opaque to the profiler."""

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def synchronize_cuda():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Profiler:
    def __init__(self, mode=None):
        if mode not in (None, "wall", "sync"):
            raise ValueError("profile mode must be wall or sync")
        self.mode = mode
        self.samples = defaultdict(list)

    @contextmanager
    def span(self, stage: str, gpu: bool = False):
        if self.mode is None:
            yield
            return
        wait_for_gpu = gpu and self.mode == "sync"
        if wait_for_gpu:
            # Exclude GPU work queued by earlier stages from this measurement.
            synchronize_cuda()
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            if wait_for_gpu:
                synchronize_cuda()
            self.samples[stage].append((time.perf_counter_ns() - started) / 1e6)

    def save(self, path: Path):
        """Write raw timings and per-stage statistics to one JSON file."""
        if self.mode is None:
            return
        stages = {}
        for stage, values in self.samples.items():
            median, p95, p99 = np.percentile(values, [50, 95, 99])
            stages[stage] = {
                "count": len(values),
                "mean_ms": sum(values) / len(values),
                "median_ms": float(median),
                "p95_ms": float(p95),
                "p99_ms": float(p99),
                "samples_ms": values,
            }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"mode": self.mode, "unit": "ms", "stages": stages}, indent=2) + "\n",
            encoding="utf-8",
        )


DISABLED_PROFILER = Profiler()
