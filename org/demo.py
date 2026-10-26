"""데모: '뉴스 처리 회사'.

  reception(접수) -> triage(분류팀장, 라우터) -> sports/business/tech 데스크 -> editor(편집국)

각 부서는 아직 키워드 규칙 기반의 자리표시자다. 이후 실제 소형 모델로
하나씩 바꿔 끼우는 것이 목표 (부서 = 칩 하나).

시나리오 후반에 tech 데스크를 v2(벡터 입력을 받는 새 칩)로 교체해서,
  1) 영향이 triage->tech 링크로만 격리되는지
  2) 양식이 달라 끊긴 링크가 Adapter 배치로 복구되는지
를 확인한다.
"""
from __future__ import annotations

import hashlib

from . import Adapter, Company, FunctionDepartment, Port, Spec, ROUTE_KEY

TEXT = Spec.of(text=Port("text"))
TOKENS = Spec.of(tokens=Port("tokens"))
SUMMARY = Spec.of(desk=Port("label"), summary=Port("text"))
VEC_DIM = 16
VECTOR = Spec.of(vec=Port("vector", VEC_DIM))

KEYWORDS = {
    "sports": {"match", "goal", "league", "team", "season", "coach"},
    "business": {"market", "stock", "profit", "bank", "shares", "earnings"},
    "tech": {"chip", "ai", "software", "model", "startup", "gpu"},
}


def tokenize(p):
    return {"tokens": [t.strip(".,!?").lower() for t in p["text"].split() if t.strip(".,!?")]}


def triage(p):
    scores = {desk: len(KEYWORDS[desk] & set(p["tokens"])) for desk in KEYWORDS}
    best = max(scores, key=scores.get)
    return {"tokens": p["tokens"], ROUTE_KEY: [f"{best}_desk"]}


def desk(name):
    def fn(p):
        hits = [t for t in p["tokens"] if t in KEYWORDS[name]]
        return {"desk": name, "summary": f"{len(p['tokens'])} words, keywords={hits}"}
    return fn


def edit(p):
    return {"text": f"[{p['desk'].upper()}] {p['summary']}"}


def hash_embed(tokens, dim=VEC_DIM):
    """토큰 -> 고정 길이 벡터 (feature hashing). '번역 담당'의 자리표시자."""
    v = [0.0] * dim
    for t in tokens:
        v[int(hashlib.md5(t.encode()).hexdigest(), 16) % dim] += 1.0
    return v


def tech_desk_v2(p):
    # 새 칩: 토큰이 아니라 고정 폭 벡터를 받는다. tech 키워드 버킷의 활성도로 요약.
    tech_buckets = {int(hashlib.md5(k.encode()).hexdigest(), 16) % VEC_DIM for k in KEYWORDS["tech"]}
    act = sum(p["vec"][i] for i in tech_buckets)
    return {"desk": "tech", "summary": f"vector-chip v2, tech activation={act:.0f}"}


def build_company() -> Company:
    c = Company("newsroom")
    c.hire(FunctionDepartment("reception", tokenize, TEXT, TOKENS, cost=0.5))
    c.hire(FunctionDepartment("triage", triage, TOKENS, TOKENS, cost=1.0))
    for name in KEYWORDS:
        c.hire(FunctionDepartment(f"{name}_desk", desk(name), TOKENS, SUMMARY, cost=3.0))
    c.hire(FunctionDepartment("editor", edit, SUMMARY, TEXT, cost=1.0))

    c.connect("reception", "triage")
    for name in KEYWORDS:
        c.connect("triage", f"{name}_desk")
        c.connect(f"{name}_desk", "editor")
    return c


ARTICLES = [
    "The team scored a late goal to win the league match.",
    "Bank shares fell as the stock market digested weak earnings.",
    "A startup unveiled a new AI chip that rivals the GPU giants.",
]


def run_all(c: Company):
    for a in ARTICLES:
        out = c.run("reception", {"text": a})
        print("\n".join(c.trace))
        print("  =>", out["editor"][0]["text"])


def main():
    c = build_company()
    print(c.org_chart(), "\n")
    run_all(c)

    print("\n--- tech_desk 를 v2(벡터 입력 칩)로 교체 ---")
    v2 = FunctionDepartment("tech_desk", tech_desk_v2, VECTOR, SUMMARY, version="v2", cost=1.5)
    print(c.swap(v2))

    print("\n--- triage->tech_desk 링크에 tokens->vector 번역 담당 배치 ---")
    c.attach_adapter("triage", "tech_desk", Adapter(
        "tokens2vec", lambda p: {"vec": hash_embed(p["tokens"])}, TOKENS, VECTOR))
    print(c.org_chart(), "\n")
    run_all(c)

    print("\n--- 누적 비용 (부서=칩별 부하) ---")
    for k, v in c.cost_report().items():
        print(f"  {k:<28} {v:5.1f}")
    print("--- 링크별 메시지 수 (버스 트래픽) ---")
    for (s, d), n in c.traffic.items():
        print(f"  {s + '->' + d:<28} {n}")


if __name__ == "__main__":
    main()
