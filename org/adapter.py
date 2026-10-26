"""Adapter = 양식이 다른 두 부서 사이의 번역 담당 (ml_experiment의 학습된 Adapter에 대응).

연결된 부서 중 하나가 교체되면 stale 상태가 되어 '재학습 필요'로 표시된다.
CONTEXT.md ②의 발견대로, 양식(차원)이 같아도 내부 표현이 바뀌었을 수 있으므로
교체 시에는 무조건 stale로 본다.
"""
from __future__ import annotations

from typing import Callable

from .contract import Spec


class Adapter:
    def __init__(self, name: str, fn: Callable[[dict], dict], input_spec: Spec,
                 output_spec: Spec, cost: float = 0.1):
        self.name = name
        self.fn = fn
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.cost = cost
        self.stale = False
        self.calls = 0

    def translate(self, payload: dict) -> dict:
        self.input_spec.validate(payload)
        out = self.fn(payload)
        self.output_spec.validate(out)
        self.calls += 1
        return out

    def retrained(self, fn: Callable[[dict], dict] | None = None) -> "Adapter":
        """재학습 완료 처리. 새 변환 함수를 주면 교체한다."""
        if fn is not None:
            self.fn = fn
        self.stale = False
        return self
