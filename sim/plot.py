"""Renders latency/throughput/quality comparisons and prints a text summary."""
from __future__ import annotations

import matplotlib.pyplot as plt

from .metrics import MetricsCollector

BUCKET = 10.0


def plot_results(metrics: MetricsCollector, update_times, total_time: float, out_path: str):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for name, color in (("monolithic", "tab:red"), ("modular", "tab:blue")):
        t, lat, thr, qual = metrics.bucketed(name, BUCKET, total_time)
        axes[0].plot(t, lat, label=name, color=color)
        axes[1].plot(t, thr, label=name, color=color)
        axes[2].plot(t, qual, label=name, color=color)

    for ax in axes:
        for ut in update_times:
            ax.axvline(ut, color="gray", linestyle="--", linewidth=0.8)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("avg latency")
    axes[1].set_ylabel("throughput (req/s)")
    axes[2].set_ylabel("avg quality")
    axes[2].set_xlabel("simulated time")
    fig.suptitle("Monolithic vs Modular: cost of updating one capability\n"
                 "(dashed lines = model update events)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"saved plot to {out_path}")


def print_summary(metrics: MetricsCollector, update_times, window: float = 60.0):
    print("\n=== summary: quality/latency in the window right after each update ===")
    for name in ("monolithic", "modular"):
        print(f"\n[{name}]")
        for ut in update_times:
            samples = [(t, lat, q) for t, lat, q in metrics.samples[name] if ut <= t <= ut + window]
            if not samples:
                continue
            avg_q = sum(s[2] for s in samples) / len(samples)
            avg_lat = sum(s[1] for s in samples) / len(samples)
            min_q = min(s[2] for s in samples)
            print(f"  update@t={ut:>6.0f}: avg_quality={avg_q:.3f}  min_quality={min_q:.3f}  "
                  f"avg_latency={avg_lat:.2f}  n_requests={len(samples)}")
