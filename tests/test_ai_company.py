import unittest
from types import SimpleNamespace

from org.ai_company import BudgetExceeded, MAX_TASKS, build_ai_company, parse_tasks, run_project
from org.brain import ClaudeBrain, ClaudeRefusal, ScriptedBrain, Thought


class RecordingBrain(ScriptedBrain):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []

    def think(self, role, system, prompt, max_tokens):
        self.calls.append((role, max_tokens))
        return super().think(role, system, prompt, max_tokens)


class AICompanyTest(unittest.TestCase):
    def test_parse_tasks(self):
        text = "목표: x\n- a\n2) b\n* c\n본문"
        self.assertEqual(parse_tasks(text), ["a", "b", "c"])
        self.assertEqual(len(parse_tasks("\n".join(f"- t{i}" for i in range(9)))), MAX_TASKS)

    def test_pipeline_and_ledger(self):
        brain = RecordingBrain()
        c = build_ai_company(brain, total_budget=100_000)
        result = run_project(c, "할 일 관리 CLI")
        self.assertEqual(len(result["deliverables"]), 4)
        finance = c.departments["finance"]
        self.assertEqual([e.dept for e in finance.ledger.entries], ["planning", "finance", "development"])
        # 장부(버스 계측)와 부서 자체 집계가 일치해야 한다
        self.assertEqual(finance.ledger.total, sum(d.spent for d in c.departments.values()))
        # 개발부는 배정 예산을 출력 상한으로 사용한다
        dev_caps = [m for role, m in brain.calls if role == "development"]
        self.assertTrue(all(m <= b for m, b in zip(dev_caps, finance.allocated.values())))

    def test_finance_drops_low_priority_tasks_when_budget_is_tight(self):
        c = build_ai_company(ScriptedBrain(), total_budget=15_000, min_task_budget=4000)
        result = run_project(c, "x")
        self.assertEqual(len(result["deliverables"]), 3)  # 4개 중 마지막 작업 보류
        self.assertIn("보류", c.departments["finance"].process(
            {"plan": "p", "tasks": ["a", "b", "c", "d"]})["schedule"])

    def test_budget_exhaustion_blocks_next_department(self):
        c = build_ai_company(ScriptedBrain(), total_budget=3000, min_task_budget=4000)
        with self.assertRaises(BudgetExceeded):
            run_project(c, "x")
        self.assertNotIn("development", [e.dept for e in c.departments["finance"].ledger.entries])

    def test_truncation_reported_when_budget_too_small(self):
        tiny = ScriptedBrain(script=lambda role, p: "- a" if role == "planning" else "x" * 40_000)
        c = build_ai_company(tiny, total_budget=6000, min_task_budget=1000)
        c.departments["development"].min_tokens = 1
        self.assertIn("잘림", run_project(c, "x")["report"])

    def test_department_brain_can_be_swapped(self):
        c = build_ai_company(ScriptedBrain())
        alt = ScriptedBrain(script=lambda role, p: "- 하나" if role == "planning" else "새 두뇌 산출물")
        c.departments["development"].brain = alt
        self.assertEqual(run_project(c, "x")["deliverables"][0], "새 두뇌 산출물")


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kw):
        self.requests.append(kw)
        return self.responses.pop(0)


def resp(stop, text="", tin=10, tout=5):
    return SimpleNamespace(
        stop_reason=stop, stop_details=None,
        content=[SimpleNamespace(type="text", text=text)] if text else [],
        usage=SimpleNamespace(input_tokens=tin, output_tokens=tout))


def fake_client(*responses):
    msgs = FakeMessages(responses)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


class ClaudeBrainTest(unittest.TestCase):
    def test_request_shape_and_usage(self):
        client, msgs = fake_client(resp("end_turn", "- a"))
        t = ClaudeBrain(client=client).think("planning", "sys", "req", 500)
        self.assertEqual((t.text, t.tokens, t.truncated), ("- a", 15, False))
        r = msgs.requests[0]
        self.assertEqual(r["model"], "claude-opus-5-5")
        self.assertEqual(r["fallbacks"], "default")
        self.assertEqual(r["output_config"], {"effort": "medium"})
        self.assertNotIn("tools", r)

    def test_pause_turn_resumes_and_sums_usage(self):
        client, msgs = fake_client(resp("pause_turn"), resp("end_turn", "done"))
        t = ClaudeBrain(client=client, research=True).think("planning", "s", "q", 500)
        self.assertEqual((t.text, t.tokens), ("done", 30))
        self.assertEqual(msgs.requests[1]["messages"][1]["role"], "assistant")
        self.assertEqual(msgs.requests[0]["tools"][0]["name"], "web_search")

    def test_refusal_and_truncation(self):
        client, _ = fake_client(resp("refusal"))
        with self.assertRaises(ClaudeRefusal):
            ClaudeBrain(client=client).think("development", "s", "q", 500)
        client, _ = fake_client(resp("max_tokens", "part"))
        self.assertTrue(ClaudeBrain(client=client).think("development", "s", "q", 500).truncated)


if __name__ == "__main__":
    unittest.main()
