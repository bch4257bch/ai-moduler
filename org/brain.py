"""부서의 '두뇌' = 실제로 생각(추론)하는 모델. 부서 껍데기와 분리해서 갈아끼울 수 있다.

  ScriptedBrain : 네트워크/키 없이 돌아가는 결정적 자리표시자 (테스트·데모용)
  ClaudeBrain   : Claude API 호출. 사용 토큰을 그대로 보고한다.

하드웨어 관점: 같은 부서(칩 소켓)에 다른 두뇌(칩)를 꽂는 구조. 부서의 입출력
양식은 그대로 두고 두뇌만 바꿔도 회사 전체가 동작해야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass
class Thought:
    text: str
    input_tokens: int
    output_tokens: int
    truncated: bool = False  # max_tokens(=배정 예산)에 걸려 잘렸는지

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Brain(Protocol):
    def think(self, role: str, system: str, prompt: str, max_tokens: int) -> Thought: ...


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _default_script(role: str, prompt: str) -> str:
    if role == "planning":
        return (
            "목표: " + prompt.splitlines()[0][:80] + "\n"
            "- 요구사항과 제약 조건 정리\n"
            "- 핵심 기능 구현\n"
            "- 테스트 작성 및 검증\n"
            "- 사용 방법 문서화"
        )
    if role == "development":
        task = next((l for l in prompt.splitlines() if l.startswith("작업:")), "작업: ?")
        return f"[산출물] {task[3:].strip()} — 초안 완료. " + "구현 세부 내용 " * 10
    return "ok"


class ScriptedBrain:
    """규칙 기반 응답. max_tokens를 넘는 출력은 잘라서 예산 준수 동작을 흉내낸다."""

    def __init__(self, script: Callable[[str, str], str] = _default_script):
        self.script = script

    def think(self, role: str, system: str, prompt: str, max_tokens: int) -> Thought:
        text = self.script(role, prompt)
        out = _approx_tokens(text)
        truncated = out > max_tokens
        if truncated:
            text, out = text[: max_tokens * 4], max_tokens
        return Thought(text, _approx_tokens(system + prompt), out, truncated)


class ClaudeRefusal(RuntimeError):
    pass


class ClaudeBrain:
    """Claude API 두뇌. 역할별로 effort를 다르게 줄 수 있다.

    research=True 이면 기획부(planning)에 web_search 서버 도구를 붙여 정보수집을 시킨다.
    """

    DEFAULT_EFFORT = {"planning": "medium", "development": "high"}

    def __init__(self, model: str = "claude-opus-5-5", effort: dict[str, str] | None = None,
                 research: bool = False, client=None):
        if client is None:
            import anthropic  # 오프라인 모드에서는 SDK가 없어도 되도록 지연 import

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = {**self.DEFAULT_EFFORT, **(effort or {})}
        self.research = research

    def think(self, role: str, system: str, prompt: str, max_tokens: int) -> Thought:
        tools = []
        if self.research and role == "planning":
            tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}]
        messages = [{"role": "user", "content": prompt}]
        input_tokens = output_tokens = 0
        for _ in range(5):  # pause_turn(서버 도구 루프 한도) 이어받기 상한
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": self.effort.get(role, "medium")},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **({"tools": tools} if tools else {}),
            )
            input_tokens += resp.usage.input_tokens
            output_tokens += resp.usage.output_tokens
            if resp.stop_reason != "pause_turn":
                break
            messages = [messages[0], {"role": "assistant", "content": resp.content}]
        if resp.stop_reason == "refusal":
            raise ClaudeRefusal(f"{role}: request declined ({resp.stop_details})")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Thought(text, input_tokens, output_tokens, resp.stop_reason == "max_tokens")
