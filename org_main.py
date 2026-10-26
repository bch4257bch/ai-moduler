"""Entry point for the company-structured orchestration demos.

  python org_main.py                      # 뉴스룸 데모 (부서 교체/어댑터)
  python org_main.py ai "요청"            # AI 회사: 기획부 -> 재정관리부 -> 개발부 (오프라인 두뇌)
  python org_main.py ai "요청" --claude   # 같은 구조를 Claude API로 실행 (ANTHROPIC_API_KEY 필요)
"""
import argparse

from org.ai_company import build_ai_company, run_project
from org.brain import ClaudeBrain, ScriptedBrain
from org.demo import main as newsroom_main


def ai_main(args):
    brain = ClaudeBrain(research=args.research) if args.claude else ScriptedBrain()
    c = build_ai_company(brain, total_budget=args.budget, min_task_budget=args.min_task_budget,
                         deadline_sec=args.deadline)
    print(c.org_chart(), "\n")
    result = run_project(c, args.request)
    print("\n".join(c.trace), "\n")
    print("[개발부 보고]\n" + result["report"], "\n")
    for i, d in enumerate(result["deliverables"], 1):
        print(f"--- 산출물 {i} ---\n{d[:300]}{' ...' if len(d) > 300 else ''}\n")
    print(c.departments["finance"].report())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="demo")
    sub.add_parser("newsroom")
    ai = sub.add_parser("ai")
    ai.add_argument("request", nargs="?", default="간단한 할 일 관리 CLI 프로그램을 만들어줘")
    ai.add_argument("--claude", action="store_true", help="Claude API 두뇌 사용")
    ai.add_argument("--research", action="store_true", help="기획부에 웹 검색 허용 (--claude 전용)")
    ai.add_argument("--budget", type=int, default=100_000, help="프로젝트 총 토큰 예산")
    ai.add_argument("--min-task-budget", type=int, default=4000)
    ai.add_argument("--deadline", type=float, default=None, help="마감 (초)")
    args = p.parse_args()
    if args.demo == "ai":
        ai_main(args)
    else:
        newsroom_main()
