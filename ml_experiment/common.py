"""공통 어댑터 정의 + 학습/재학습 측정 하니스.

시뮬레이션(sim/core.py)의 RetrainState 개념을 실측으로 검증하기 위한 도구.
각 실험(exp_image_text.py, exp_text_text.py)은 이 모듈을 재사용해서:
  1) 두 frozen 모듈 사이에 Adapter를 처음부터 학습 (cold start)
  2) 한쪽 모듈을 교체(swap)한 뒤, quality가 얼마나 떨어지는지 측정
  3) Adapter를 재학습시키며 quality가 회복되는 곡선을 기록
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


class Adapter(nn.Module):
    """두 임베딩 공간을 잇는 학습 가능한 인터페이스 (2-layer MLP + residual)."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.proj(x)


@dataclass
class QualityPoint:
    step: int
    elapsed_sec: float
    loss: float
    quality: float


@dataclass
class RetrainRecord:
    name: str
    phase: str  # "initial_train" | "post_swap_retrain"
    history: list[dict] = field(default_factory=list)  # list[QualityPoint as dict]
    quality_before_swap: float | None = None
    quality_immediately_after_swap: float | None = None
    retrain_time_sec: float | None = None  # time to recover to 95% of pre-swap quality
    steps_to_recover: int | None = None


StepFn = Callable[[int], tuple[float, float]]  # step -> (loss, quality)


class RetrainHarness:
    """step_fn(step)->(loss, quality) 를 반복 호출하며 곡선을 기록."""

    def __init__(self, name: str):
        self.name = name
        self.records: list[RetrainRecord] = []

    def run(self, phase: str, step_fn: StepFn, max_steps: int, eval_every: int = 1,
            recover_target: float | None = None, patience_stop: int | None = None) -> RetrainRecord:
        record = RetrainRecord(name=self.name, phase=phase)
        start = time.perf_counter()
        recovered_at = None
        best_quality = -1.0
        stale = 0
        for step in range(1, max_steps + 1):
            loss, quality = step_fn(step)
            if step % eval_every == 0 or step == max_steps:
                elapsed = time.perf_counter() - start
                record.history.append(asdict(QualityPoint(step, elapsed, loss, quality)))
                if recover_target is not None and recovered_at is None and quality >= recover_target:
                    recovered_at = (step, elapsed)
                if quality > best_quality + 1e-4:
                    best_quality = quality
                    stale = 0
                else:
                    stale += 1
                if patience_stop is not None and stale >= patience_stop and (
                    recovered_at is not None or recover_target is None
                ):
                    break
        if recovered_at is not None:
            record.steps_to_recover, record.retrain_time_sec = recovered_at
        self.records.append(record)
        return record

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.records], f, ensure_ascii=False, indent=2)
        return path


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
