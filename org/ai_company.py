"""AI 회사: 기획부 -> 재정관리부 -> 개발부.

  기획부(planning)    : 요청을 받아 계획서(plan)와 작업 목록(tasks)을 만든다. 정보수집 담당.
  재정관리부(finance) : 계획서를 받아 작업별 토큰 예산(budgets)과 일정(schedule)을 배정한다.
                        예산이 모자라면 우선순위가 낮은(뒤쪽) 작업부터 뺀다.
                        동시에 버스 감시자(Supervisor)로서 모든 부서의 토큰/시간 사용량을
                        장부에 기록하고, 총예산·마감을 넘기면 다음 부서 실행을 막는다.
  개발부(development) : 계획서와 배정된 예산을 받아 작업별로 산출물을 만든다.

재정관리부는 LLM이 아니라 규칙 기반으로 둔다. 회계/통제는 결정적이어야 하고,
하드웨어로 치면 연산 칩이 아니라 전력관리·클럭 제어 칩(PMU)에 해당하기 때문.
재정관리부의 통제는 두 경로로만 이뤄진다:
  1) 메시지: 개발부에 budgets를 보내 '이만큼만 써라'라고 지시 (개발부가 스스로 지킴)
  2) 버스 계측: 부서 내부를 보지 않고, 호출 전후 사용량만 재서 장부에 기록·차단
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .brain import Brain
from .company import Company, OrgError
from .contract import Port, Spec
from .department import Department

MAX_TASKS = 5  # 작업 슬롯 수 (버스 폭처럼 고정)

REQUEST = Spec.of(request=Port("text"))
PLAN = Spec.of(plan=Port("text"), tasks=Port("items", MAX_TASKS))
FUNDED_PLAN = Spec.of(plan=Port("text"), tasks=Port("items", MAX_TASKS),
                      budgets=Port("numbers", MAX_TASKS), schedule=Port("text"))
DELIVERY = Spec.of(deliverables=Port("items", MAX_TASKS), report=Port("text"))


class BudgetExceeded(OrgError):
    pass


class ScheduleExceeded(OrgError):
    pass


# ---------------------------------------------------------------------------
# 기획부
# ---------------------------------------------------------------------------
PLANNING_SYSTEM = f"""당신은 AI 회사의 기획부입니다. 프로젝트 요청을 받아 개발부가 바로 실행할 수 있는
계획서를 작성합니다. 필요하면 정보를 수집해 근거로 삼으세요.

출력 형식:
- 첫 부분: 목표, 범위, 주요 결정 사항을 간결하게 서술
- 마지막 부분: 작업 목록. 각 줄을 "- "로 시작하고, 중요한 작업부터 최대 {MAX_TASKS}개.
  각 작업은 개발부가 독립적으로 수행할 수 있는 단위로 작성"""

_TASK_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$")


def parse_tasks(text: str, limit: int = MAX_TASKS) -> list[str]:
    tasks = [m.group(1).strip() for line in text.splitlines() if (m := _TASK_LINE.match(line))]
    return tasks[:limit]


class PlanningDept(Department):
    def __init__(self, brain: Brain, max_tokens: int = 8000, version: str = "v1"):
        super().__init__("planning", REQUEST, PLAN, version=version, cost=0.0)
        self.brain = brain
        self.max_tokens = max_tokens

    def process(self, payload: dict) -> dict:
        thought = self.brain.think("planning", PLANNING_SYSTEM, payload["request"], self.max_tokens)
        self.charge(thought.tokens)
        tasks = parse_tasks(thought.text) or [payload["request"]]
        return {"plan": thought.text, "tasks": tasks}


# ---------------------------------------------------------------------------
# 재정관리부
# ---------------------------------------------------------------------------
@dataclass
class LedgerEntry:
    dept: str
    tokens: float
    seconds: float


@dataclass
class Ledger:
    entries: list[LedgerEntry] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(e.tokens for e in self.entries)

    def by_dept(self) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for e in self.entries:
            t, s = out.get(e.dept, (0.0, 0.0))
            out[e.dept] = (t + e.tokens, s + e.seconds)
        return out


class FinanceDept(Department):
    """예산 배정(부서 역할) + 사용량 장부/차단(Supervisor 역할)."""

    def __init__(self, total_budget: int, min_task_budget: int = 4000,
                 reserve_ratio: float = 0.1, deadline_sec: float | None = None,
                 version: str = "v1"):
        super().__init__("finance", PLAN, FUNDED_PLAN, version=version, cost=0.0)
        self.total_budget = total_budget
        self.min_task_budget = min_task_budget
        self.reserve_ratio = reserve_ratio
        self.deadline_sec = deadline_sec
        self.ledger = Ledger()
        self.allocated: dict[str, int] = {}
        self._started: float | None = None

    @property
    def remaining(self) -> float:
        return self.total_budget - self.ledger.total

    # ---- 부서 역할: 예산/일정 배정 ---------------------------------------
    def process(self, payload: dict) -> dict:
        tasks = list(payload["tasks"])
        spendable = self.remaining - self.total_budget * self.reserve_ratio
        dropped = []
        while len(tasks) > 1 and spendable / len(tasks) < self.min_task_budget:
            dropped.append(tasks.pop())
        if spendable < self.min_task_budget:
            raise BudgetExceeded(
                f"remaining budget {spendable:.0f} < min task budget {self.min_task_budget}")
        each = int(spendable // len(tasks))
        budgets = [each] * len(tasks)
        self.allocated = dict(zip(tasks, budgets))

        lines = [f"{i}. {t} — {b} tok" for i, (t, b) in enumerate(zip(tasks, budgets), 1)]
        if dropped:
            lines.append(f"보류(예산 부족): {', '.join(dropped)}")
        if self.deadline_sec is not None:
            lines.append(f"마감: 시작 후 {self.deadline_sec:.0f}초")
        return {"plan": payload["plan"], "tasks": tasks, "budgets": budgets,
                "schedule": "\n".join(lines)}

    # ---- Supervisor 역할: 버스 계측 --------------------------------------
    def before(self, dept: str) -> None:
        if self._started is None:
            self._started = time.perf_counter()
        if self.ledger.total >= self.total_budget:
            raise BudgetExceeded(f"total budget {self.total_budget} exhausted before {dept}")
        if self.deadline_sec is not None and time.perf_counter() - self._started > self.deadline_sec:
            raise ScheduleExceeded(f"deadline {self.deadline_sec}s passed before {dept}")

    def after(self, dept: str, used: float, seconds: float) -> None:
        self.ledger.entries.append(LedgerEntry(dept, used, seconds))

    def report(self) -> str:
        lines = ["[재정관리부 장부]"]
        for dept, (tok, sec) in self.ledger.by_dept().items():
            lines.append(f"  {dept:<12} {tok:>8.0f} tok  {sec:6.2f}s")
        lines.append(f"  {'합계':<12} {self.ledger.total:>8.0f} / {self.total_budget} tok"
                     f"  (남은 예산 {self.remaining:.0f})")
        dev = self.ledger.by_dept().get("development")
        if dev and self.allocated:
            budget = sum(self.allocated.values())
            flag = "초과!" if dev[0] > budget else "준수"
            lines.append(f"  개발부 배정 {budget} tok 대비 사용 {dev[0]:.0f} tok -> {flag}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 개발부
# ---------------------------------------------------------------------------
DEVELOPMENT_SYSTEM = """당신은 AI 회사의 개발부입니다. 기획부의 계획서 중 지정된 작업 하나를 수행해
실제 산출물(코드, 설계, 문서 등)을 만듭니다. 배정된 토큰 예산 안에서 끝낼 수 있도록
핵심부터 작성하고, 산출물만 출력하세요."""


class DevelopmentDept(Department):
    def __init__(self, brain: Brain, min_tokens: int = 1024, version: str = "v1"):
        super().__init__("development", FUNDED_PLAN, DELIVERY, version=version, cost=0.0)
        self.brain = brain
        self.min_tokens = min_tokens

    def process(self, payload: dict) -> dict:
        deliverables, notes = [], []
        for task, budget in zip(payload["tasks"], payload["budgets"]):
            prompt = f"계획서:\n{payload['plan']}\n\n작업: {task}"
            # 배정 예산에서 입력분을 대략 빼고 출력 상한으로 쓴다 (예산은 입력+출력 합계)
            max_tokens = max(self.min_tokens, int(budget) - len(prompt) // 4)
            thought = self.brain.think("development", DEVELOPMENT_SYSTEM, prompt, max_tokens)
            self.charge(thought.tokens)
            deliverables.append(thought.text)
            status = "잘림(예산 소진)" if thought.truncated else "완료"
            notes.append(f"- {task}: {status}, {thought.tokens} tok / 배정 {int(budget)}")
        done = sum(1 for n in notes if "완료" in n)
        report = f"작업 {done}/{len(notes)} 완료\n" + "\n".join(notes)
        return {"deliverables": deliverables, "report": report}


# ---------------------------------------------------------------------------
def build_ai_company(brain: Brain, total_budget: int = 100_000, planning_tokens: int = 8000,
                     min_task_budget: int = 4000, deadline_sec: float | None = None,
                     dev_brain: Brain | None = None) -> Company:
    c = Company("ai-company")
    finance = FinanceDept(total_budget, min_task_budget=min_task_budget, deadline_sec=deadline_sec)
    c.hire(PlanningDept(brain, max_tokens=planning_tokens))
    c.hire(finance)
    c.hire(DevelopmentDept(dev_brain or brain))
    c.connect("planning", "finance")
    c.connect("finance", "development")
    c.supervisor = finance
    return c


def run_project(c: Company, request: str) -> dict:
    out = c.run("planning", {"request": request})
    return out["development"][0]
