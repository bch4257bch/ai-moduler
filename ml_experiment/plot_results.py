"""results/*.json 을 읽어 quality 회복 곡선을 그리고, sim 단계 가정과 비교할 수 있는
수치 요약을 출력한다 (sim/plot.py 와 대응되는 ML 레벨 버전)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"


def load(name: str):
    with open(RESULTS_DIR / f"{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def plot_experiment(ax, name: str, records: list):
    initial = next(r for r in records if r["phase"] == "initial_train")
    retrain = next(r for r in records if r["phase"] == "post_swap_retrain")

    steps1 = [p["step"] for p in initial["history"]]
    q1 = [p["quality"] for p in initial["history"]]
    steps2 = [p["step"] + steps1[-1] for p in retrain["history"]]
    q2 = [p["quality"] for p in retrain["history"]]

    ax.plot(steps1, q1, label="initial train", color="tab:blue")
    ax.plot(steps2, q2, label="post-swap retrain", color="tab:orange")
    ax.axvline(steps1[-1], color="gray", linestyle="--", linewidth=1, label="module swap")
    if retrain.get("quality_immediately_after_swap") is not None:
        ax.scatter([steps1[-1]], [retrain["quality_immediately_after_swap"]],
                   color="red", zorder=5, label="quality right after swap")
    if retrain.get("retrain_time_sec") is not None:
        ax.axhline(0.95 * retrain["quality_before_swap"], color="green",
                   linestyle=":", linewidth=1, label="95% recovery target")
    ax.set_title(name)
    ax.set_xlabel("step")
    ax.set_ylabel("quality")
    ax.legend(fontsize=7)


def summarize(name: str, records: list) -> str:
    retrain = next(r for r in records if r["phase"] == "post_swap_retrain")
    before = retrain["quality_before_swap"]
    after = retrain["quality_immediately_after_swap"]
    drop_pct = (before - after) / before * 100 if before else float("nan")
    lines = [
        f"[{name}]",
        f"  quality_before_swap            = {before:.4f}",
        f"  quality_immediately_after_swap  = {after:.4f}  (drop {drop_pct:.1f}%)",
        f"  retrain_time_sec (to 95% recov) = {retrain['retrain_time_sec']}",
        f"  steps_to_recover                = {retrain['steps_to_recover']}",
    ]
    return "\n".join(lines)


def main():
    names = [p.stem for p in RESULTS_DIR.glob("*.json")]
    if not names:
        print("no results found in", RESULTS_DIR)
        return

    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4.5))
    if len(names) == 1:
        axes = [axes]

    summary_lines = []
    for ax, name in zip(axes, names):
        records = load(name)
        plot_experiment(ax, name, records)
        summary_lines.append(summarize(name, records))

    fig.tight_layout()
    out_path = RESULTS_DIR / "quality_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot -> {out_path}")
    print()
    print("\n\n".join(summary_lines))

    with open(RESULTS_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
