"""Company = 부서들 + 조직도(링크) + 사내 버스. 칩들이 꽂힌 기판에 대응."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from .adapter import Adapter
from .department import ROUTE_KEY, Department


class OrgError(RuntimeError):
    pass


@dataclass
class Link:
    """조직도의 한 선 (src 부서 -> dst 부서). 하드웨어로는 칩 사이 배선 한 가닥."""

    src: str
    dst: str
    adapter: Adapter | None = None
    broken: list[str] = field(default_factory=list)  # 비어있지 않으면 양식 불일치로 끊긴 상태

    @property
    def status(self) -> str:
        if self.broken:
            return "broken"
        if self.adapter is not None and self.adapter.stale:
            return "adapter_stale"
        return "ok"


@dataclass
class SwapReport:
    """부서 교체(칩 교체)의 영향 범위."""

    department: str
    old_version: str
    new_version: str
    affected_links: dict[str, str]   # "src->dst" -> status
    untouched_departments: list[str]

    def __str__(self) -> str:
        lines = [f"swap {self.department}: {self.old_version} -> {self.new_version}"]
        for k, s in self.affected_links.items():
            lines.append(f"  link {k:<28} {s}")
        lines.append(f"  untouched departments: {', '.join(self.untouched_departments) or '-'}")
        return "\n".join(lines)


class Company:
    def __init__(self, name: str):
        self.name = name
        self.departments: dict[str, Department] = {}
        self.links: dict[tuple[str, str], Link] = {}
        self.traffic: Counter = Counter()  # (src, dst) -> 메시지 수 (버스 대역폭 추정용)
        self.trace: list[str] = []          # 마지막 run의 메시지 흐름

    # ---- 조직 구성 -------------------------------------------------------
    def hire(self, dept: Department) -> Department:
        if dept.name in self.departments:
            raise OrgError(f"department {dept.name!r} already exists")
        self.departments[dept.name] = dept
        return dept

    def connect(self, src: str, dst: str, adapter: Adapter | None = None) -> Link:
        for n in (src, dst):
            if n not in self.departments:
                raise OrgError(f"unknown department {n!r}")
        link = Link(src, dst, adapter)
        link.broken = self._check(link)
        if link.broken:
            raise OrgError(f"cannot connect {src}->{dst}: {link.broken} (add an Adapter)")
        self.links[(src, dst)] = link
        return link

    def attach_adapter(self, src: str, dst: str, adapter: Adapter) -> Link:
        """끊긴(또는 stale) 링크에 번역 담당을 새로 배치한다."""
        link = self.links[(src, dst)]
        link.adapter = adapter
        link.broken = self._check(link)
        if link.broken:
            raise OrgError(f"adapter does not fix {src}->{dst}: {link.broken}")
        return link

    def _check(self, link: Link) -> list[str]:
        out = self.departments[link.src].output_spec
        inp = self.departments[link.dst].input_spec
        if link.adapter is None:
            return out.mismatches(inp)
        return (
            [f"adapter in: {p}" for p in out.mismatches(link.adapter.input_spec)]
            + [f"adapter out: {p}" for p in link.adapter.output_spec.mismatches(inp)]
        )

    def successors(self, name: str) -> list[str]:
        return [d for (s, d) in self.links if s == name]

    # ---- 칩 교체 ---------------------------------------------------------
    def swap(self, new_dept: Department) -> SwapReport:
        """같은 이름의 부서를 새 버전으로 교체하고, 영향 받은 링크만 재검사한다."""
        name = new_dept.name
        if name not in self.departments:
            raise OrgError(f"unknown department {name!r}")
        old = self.departments[name]
        self.departments[name] = new_dept

        affected = {}
        touched = {name}
        for (s, d), link in self.links.items():
            if name not in (s, d):
                continue
            touched |= {s, d}
            if link.adapter is not None:
                link.adapter.stale = True
            link.broken = self._check(link)
            affected[f"{s}->{d}"] = link.status
        untouched = [n for n in self.departments if n not in touched]
        return SwapReport(name, old.version, new_dept.version, affected, untouched)

    # ---- 업무 처리 (버스) -------------------------------------------------
    def run(self, entry: str, payload: dict, allow_stale: bool = False) -> dict[str, list[dict]]:
        """entry 부서에 업무를 넣고, 조직도를 따라 흘려 보낸다.

        반환: 더 이상 보낼 곳이 없는 말단 부서들의 최종 출력 {부서명: [payload, ...]}.
        (fan-out으로 같은 말단에 여러 건이 도착할 수 있어서 리스트)
        """
        self.trace = []
        results: dict[str, list[dict]] = {}
        queue = deque([(entry, payload, 0)])
        steps = 0
        while queue:
            name, data, hop = queue.popleft()
            steps += 1
            if steps > 10 * max(1, len(self.departments)):
                raise OrgError("routing loop detected")
            dept = self.departments[name]
            out = dept.receive(data)
            route = out.pop(ROUTE_KEY, None)
            nexts = self.successors(name)
            if route is not None:
                bad = set(route) - set(nexts)
                if bad:
                    raise OrgError(f"{name} routed to unconnected {sorted(bad)}")
                nexts = list(route)
            self.trace.append(f"{'  ' * hop}{name}@{dept.version} -> {nexts or '[done]'}")
            if not nexts:
                results.setdefault(name, []).append(out)
            for nxt in nexts:
                queue.append((nxt, self._send(name, nxt, out, allow_stale), hop + 1))
        return results

    def _send(self, src: str, dst: str, data: dict, allow_stale: bool) -> dict:
        link = self.links[(src, dst)]
        if link.status == "broken":
            raise OrgError(f"link {src}->{dst} is broken: {link.broken}")
        if link.status == "adapter_stale" and not allow_stale:
            raise OrgError(f"adapter on {src}->{dst} needs retraining after a swap")
        self.traffic[(src, dst)] += 1
        msg = {k: data[k] for k in self.departments[src].output_spec.as_dict()}
        if link.adapter is not None:
            msg = link.adapter.translate(msg)
        # 받는 부서의 양식에 있는 필드만 전달 (불필요한 정보는 버스에 싣지 않음)
        return {k: msg[k] for k in self.departments[dst].input_spec.as_dict()}

    # ---- 리포트 ----------------------------------------------------------
    def cost_report(self) -> dict[str, float]:
        """부서/어댑터별 누적 가상 연산 비용. 칩별 부하 분포 추정용."""
        report = {n: d.calls * d.cost for n, d in self.departments.items()}
        for (s, d), link in self.links.items():
            if link.adapter is not None:
                report[f"adapter:{s}->{d}"] = link.adapter.calls * link.adapter.cost
        return report

    def org_chart(self) -> str:
        lines = [f"[{self.name}]"]
        for n, d in self.departments.items():
            lines.append(f"  {n}@{d.version}  in={d.input_spec}  out={d.output_spec}")
        for (s, d), link in self.links.items():
            via = f" via {link.adapter.name}" if link.adapter else ""
            lines.append(f"  {s} -> {d}{via}  [{link.status}]")
        return "\n".join(lines)
