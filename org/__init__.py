"""③ 조직(company) 구조 기반 모듈러 AI 오케스트레이션.

회사 비유 ↔ 하드웨어 비유:
  Department(부서)     ↔ 칩 1개 (모듈화된 모델 1개가 이식된 단위)
  Spec(부서 입출력 양식) ↔ 칩의 핀/버스 규격 (고정된 I/O 계약)
  Adapter(번역 담당)    ↔ 칩 사이의 인터페이스/브리지 (학습된 projection layer)
  Company.bus(사내 메신저) ↔ 기판 위 버스/인터커넥트 (모든 통신은 여기로만)
  Router 부서(팀장/CEO)  ↔ 게이팅/스케줄러 칩 (어느 부서로 보낼지 결정)
  swap(인사이동)        ↔ 칩 교체 — 영향 범위가 연결된 링크로만 격리되는지 확인
"""
from .contract import Port, Spec, SpecError
from .department import Department, FunctionDepartment, ROUTE_KEY
from .adapter import Adapter
from .company import Company, Link, SwapReport, OrgError

__all__ = [
    "Port", "Spec", "SpecError",
    "Department", "FunctionDepartment", "ROUTE_KEY",
    "Adapter",
    "Company", "Link", "SwapReport", "OrgError",
]
