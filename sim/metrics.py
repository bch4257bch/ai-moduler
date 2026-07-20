"""Collects per-request samples and buckets them into time windows for plotting."""
from __future__ import annotations

from collections import defaultdict


class MetricsCollector:
    def __init__(self):
        # name -> list of (time, latency, quality)
        self.samples = defaultdict(list)

    def record(self, name: str, t: float, latency: float, quality: float) -> None:
        self.samples[name].append((t, latency, quality))

    def bucketed(self, name: str, bucket_size: float, total_time: float):
        """Return (bucket_centers, avg_latency, throughput, avg_quality)."""
        n_buckets = int(total_time // bucket_size) + 1
        lat_sum = [0.0] * n_buckets
        qual_sum = [0.0] * n_buckets
        count = [0] * n_buckets
        for t, latency, quality in self.samples[name]:
            b = min(int(t // bucket_size), n_buckets - 1)
            lat_sum[b] += latency
            qual_sum[b] += quality
            count[b] += 1
        centers = [(i + 0.5) * bucket_size for i in range(n_buckets)]
        avg_latency = [lat_sum[i] / count[i] if count[i] else float("nan") for i in range(n_buckets)]
        throughput = [count[i] / bucket_size for i in range(n_buckets)]
        avg_quality = [qual_sum[i] / count[i] if count[i] else float("nan") for i in range(n_buckets)]
        return centers, avg_latency, throughput, avg_quality
