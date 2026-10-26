import unittest

from org import Adapter, Company, FunctionDepartment, OrgError, Port, ROUTE_KEY, Spec, SpecError
from org.demo import ARTICLES, TOKENS, VECTOR, build_company, hash_embed, tech_desk_v2, SUMMARY

NUM = Spec.of(x=Port("scalar"))


def dept(name, fn, inp=NUM, out=NUM, **kw):
    return FunctionDepartment(name, fn, inp, out, **kw)


class SpecTest(unittest.TestCase):
    def test_vector_dim_enforced(self):
        spec = Spec.of(v=Port("vector", 3))
        spec.validate({"v": [1, 2, 3]})
        with self.assertRaises(SpecError):
            spec.validate({"v": [1, 2]})

    def test_mismatches(self):
        a = Spec.of(v=Port("vector", 3))
        self.assertEqual(a.mismatches(Spec.of(v=Port("vector", 3))), [])
        self.assertTrue(a.mismatches(Spec.of(v=Port("vector", 4))))
        self.assertTrue(a.mismatches(Spec.of(w=Port("vector", 3))))


class CompanyTest(unittest.TestCase):
    def test_incompatible_connect_rejected(self):
        c = Company("t")
        c.hire(dept("a", lambda p: p))
        c.hire(dept("b", lambda p: p, inp=Spec.of(y=Port("text"))))
        with self.assertRaises(OrgError):
            c.connect("a", "b")

    def test_fan_out_collects_all_sink_outputs(self):
        c = Company("t")
        c.hire(dept("src", lambda p: p))
        c.hire(dept("l", lambda p: {"x": p["x"] + 1}))
        c.hire(dept("r", lambda p: {"x": p["x"] + 2}))
        c.hire(dept("sink", lambda p: p))
        for s, d in [("src", "l"), ("src", "r"), ("l", "sink"), ("r", "sink")]:
            c.connect(s, d)
        out = c.run("src", {"x": 0})
        self.assertEqual(sorted(o["x"] for o in out["sink"]), [1, 2])

    def test_router_must_use_connected_links(self):
        c = Company("t")
        c.hire(dept("boss", lambda p: {**p, ROUTE_KEY: ["ghost"]}))
        c.hire(dept("w", lambda p: p))
        c.connect("boss", "w")
        with self.assertRaises(OrgError):
            c.run("boss", {"x": 1})

    def test_stale_adapter_blocks_until_retrained(self):
        c = Company("t")
        c.hire(dept("a", lambda p: p))
        c.hire(dept("b", lambda p: p))
        c.connect("a", "b", Adapter("ad", lambda p: p, NUM, NUM))
        report = c.swap(dept("b", lambda p: {"x": p["x"] * 10}, version="v2"))
        self.assertEqual(report.affected_links, {"a->b": "adapter_stale"})
        with self.assertRaises(OrgError):
            c.run("a", {"x": 1})
        self.assertEqual(c.run("a", {"x": 1}, allow_stale=True)["b"], [{"x": 10}])
        c.links[("a", "b")].adapter.retrained()
        self.assertEqual(c.run("a", {"x": 2})["b"], [{"x": 20}])


class NewsroomTest(unittest.TestCase):
    def test_routes_each_article_to_its_desk(self):
        c = build_company()
        desks = [c.run("reception", {"text": a})["editor"][0]["text"].split("]")[0] for a in ARTICLES]
        self.assertEqual(desks, ["[SPORTS", "[BUSINESS", "[TECH"])

    def test_swap_is_isolated_and_adapter_repairs(self):
        c = build_company()
        report = c.swap(FunctionDepartment("tech_desk", tech_desk_v2, VECTOR, SUMMARY, version="v2"))
        self.assertEqual(report.affected_links,
                         {"triage->tech_desk": "broken", "tech_desk->editor": "ok"})
        self.assertEqual(sorted(report.untouched_departments),
                         ["business_desk", "reception", "sports_desk"])
        # 끊긴 링크로 가지 않는 업무는 계속 처리된다
        self.assertIn("[SPORTS]", c.run("reception", {"text": ARTICLES[0]})["editor"][0]["text"])
        with self.assertRaises(OrgError):
            c.run("reception", {"text": ARTICLES[2]})
        c.attach_adapter("triage", "tech_desk", Adapter(
            "t2v", lambda p: {"vec": hash_embed(p["tokens"])}, TOKENS, VECTOR))
        self.assertIn("vector-chip v2", c.run("reception", {"text": ARTICLES[2]})["editor"][0]["text"])


if __name__ == "__main__":
    unittest.main()
