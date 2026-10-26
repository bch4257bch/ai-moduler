"""부서 = 칩 하나. 입력 양식을 받아 출력 양식으로 내보내는 것 외엔 아무것도 공유하지 않는다."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from .contract import Spec

# 라우터(팀장) 부서가 출력에 넣는 예약 필드: 다음에 보낼 부서 이름 목록.
# 양식 검증 대상이 아니고, 버스가 소비한 뒤 다음 부서에는 전달되지 않는다.
ROUTE_KEY = "_route"


class Department(ABC):
    """모든 부서의 공통 껍데기.

    하드웨어 대응을 위한 제약:
      - 내부 상태는 부서 밖으로 새지 않는다 (다른 부서 객체를 참조하지 않음).
      - 통신은 Company 버스를 통한 payload(dict)로만 한다.
      - cost = 요청 1건 처리 비용(가상 연산량). 나중에 칩별 전력/지연 예산으로 매핑.
      - spent = 누적 사용량. 고정 cost 외에 실제 사용량(LLM 토큰 등)은 charge()로 더한다.
        버스(Company)는 호출 전후 spent 차이로 사용량을 '외부에서' 계측한다
        (기판의 전력 모니터가 칩 내부를 보지 않고 소비량을 재는 것과 같은 방식).
    """

    def __init__(self, name: str, input_spec: Spec, output_spec: Spec,
                 version: str = "v1", cost: float = 1.0):
        self.name = name
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.version = version
        self.cost = cost
        self.calls = 0
        self.spent = 0.0

    def charge(self, amount: float) -> None:
        self.spent += amount

    @abstractmethod
    def process(self, payload: dict) -> dict:
        ...

    def receive(self, payload: dict) -> dict:
        self.input_spec.validate(payload)
        out = self.process(payload)
        self.output_spec.validate({k: v for k, v in out.items() if k != ROUTE_KEY})
        self.calls += 1
        self.spent += self.cost
        return out

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}@{self.version} in={self.input_spec} out={self.output_spec}>"


class FunctionDepartment(Department):
    """함수 하나로 부서를 빠르게 만드는 헬퍼 (실제 모델로 바꾸기 전 자리표시자)."""

    def __init__(self, name: str, fn: Callable[[dict], dict], input_spec: Spec,
                 output_spec: Spec, version: str = "v1", cost: float = 1.0):
        super().__init__(name, input_spec, output_spec, version, cost)
        self.fn = fn

    def process(self, payload: dict) -> dict:
        return self.fn(payload)
