"""부서 간 주고받는 '서류 양식' = 칩의 고정 I/O 규격.

하드웨어로 옮길 때를 대비해, 부서는 자유로운 파이썬 객체가 아니라
미리 선언한 필드(이름, 종류, 차원)만 주고받는다. 양식이 다르면 연결 자체가
거부되고, 그 사이에 Adapter(번역 담당)를 둬야만 연결된다.
"""
from __future__ import annotations

from dataclasses import dataclass

# kind -> payload 값이 만족해야 하는 조건
KINDS = ("text", "tokens", "vector", "label", "scalar")


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class Port:
    """양식의 필드 하나. vector는 dim이 고정되어야 한다 (버스 폭 = 고정)."""

    kind: str
    dim: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise SpecError(f"unknown port kind {self.kind!r}, expected one of {KINDS}")
        if self.kind == "vector" and not self.dim:
            raise SpecError("vector port needs a fixed dim")

    def check(self, field: str, value) -> None:
        ok = {
            "text": lambda v: isinstance(v, str),
            "tokens": lambda v: isinstance(v, list) and all(isinstance(t, str) for t in v),
            "vector": lambda v: isinstance(v, (list, tuple)) and len(v) == self.dim
            and all(isinstance(x, (int, float)) for x in v),
            "label": lambda v: isinstance(v, str),
            "scalar": lambda v: isinstance(v, (int, float)),
        }[self.kind](value)
        if not ok:
            raise SpecError(f"field {field!r}: value does not match {self}")

    def __str__(self) -> str:
        return f"{self.kind}[{self.dim}]" if self.dim else self.kind


@dataclass(frozen=True)
class Spec:
    """필드명 -> Port. 부서의 입력 양식 또는 출력 양식."""

    fields: tuple[tuple[str, Port], ...]

    @classmethod
    def of(cls, **fields: Port) -> "Spec":
        return cls(tuple(sorted(fields.items())))

    def as_dict(self) -> dict[str, Port]:
        return dict(self.fields)

    def validate(self, payload: dict) -> None:
        ports = self.as_dict()
        missing = ports.keys() - payload.keys()
        if missing:
            raise SpecError(f"missing fields {sorted(missing)}")
        for name, port in ports.items():
            port.check(name, payload[name])

    def mismatches(self, consumer: "Spec") -> list[str]:
        """이 양식(생산자 출력)이 consumer 입력 양식을 채우지 못하는 이유 목록. 빈 리스트 = 호환."""
        mine = self.as_dict()
        problems = []
        for name, port in consumer.as_dict().items():
            if name not in mine:
                problems.append(f"{name}: not provided")
            elif mine[name] != port:
                problems.append(f"{name}: {mine[name]} != {port}")
        return problems

    def __str__(self) -> str:
        return "{" + ", ".join(f"{k}: {p}" for k, p in self.fields) + "}"
