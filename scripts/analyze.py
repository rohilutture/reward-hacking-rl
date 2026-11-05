"""Aggregate runs into results/: summary tables (mean and 95% CI over seeds) and training-curve figures.

  python scripts/analyze.py
"""
import glob
import json
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}  # two-sided 95% t critical values by degrees of freedom
OUT = "results"
os.makedirs(OUT, exist_ok=True)


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T975.get(n - 1, 1.96) * sd / math.sqrt(n)


def groups():
    g = defaultdict(list)
    for d in sorted(glob.glob("runs/*_s[0-9]")):
        m = re.match(r"runs[\\/](.+)_s(\d+)$", d)
        g[m.group(1)].append(d)
    return g


def load_curve(run_dir, window=5):
    """Per-training-call means of reward, visible pass, all pass, flag rate (calls are one training step each)."""
    path = os.path.join(run_dir, "completions.jsonl")
    if not os.path.exists(path):
        return None
    by = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        by[r["call"]].append(r)
    steps = sorted(by)
    cols = {k: [] for k in ("reward", "visible_pass", "all_pass", "hacked_flag")}
    for s in steps:
        for k in cols:
            cols[k].append(sum(float(r[k]) for r in by[s]) / len(by[s]))
    return steps, cols


def smooth(xs, w=5):
    return [sum(xs[max(0, i - w + 1): i + 1]) / len(xs[max(0, i - w + 1): i + 1]) for i in range(len(xs))]


def main():
    g = groups()
    lines = ["# Results", "",
             "Held-out MBPP test problems (300), greedy decoding. Mean over seeds with 95% t-interval (n = number of seeds).", "",
             "| experiment | seeds | visible-test pass | all-tests pass | visible-but-not-hidden | hack-flag rate |",
             "|---|---|---|---|---|---|"]
    base = "runs/base/eval.json"
    if os.path.exists(base):
        b = json.load(open(base))
        lines.append(f"| base model | - | {100 * b['visible_pass']:.1f} | {100 * b['all_pass']:.1f} | "
                     f"{100 * b['visible_not_hidden']:.1f} | {100 * b['hack_flag_rate']:.1f} |")
    for name, dirs in sorted(g.items()):
        evs = [json.load(open(os.path.join(d, "eval.json"))) for d in dirs if os.path.exists(os.path.join(d, "eval.json"))]
        if not evs:
            continue
        cells = []
        for k in ("visible_pass", "all_pass", "visible_not_hidden", "hack_flag_rate"):
            m, ci = mean_ci([100 * e[k] for e in evs])
            cells.append(f"{m:.1f} ± {ci:.1f}" if not math.isnan(ci) else f"{m:.1f}")
        lines.append(f"| {name} | {len(evs)} | " + " | ".join(cells) + " |")

    lines += ["", "## Training-time signals (mean over the last 10 training steps)", "",
              "| experiment | seeds | training reward | visible pass | all-tests pass | reward minus all-pass (gap) | hack-flag rate |",
              "|---|---|---|---|---|---|---|"]
    fig_data = {}
    for name, dirs in sorted(g.items()):
        curves = [c for c in (load_curve(d) for d in dirs) if c]
        if not curves:
            continue
        fig_data[name] = curves
        tail = defaultdict(list)
        for steps, cols in curves:
            for k in cols:
                tail[k].append(100 * sum(cols[k][-10:]) / len(cols[k][-10:]))
        cells = []
        for k in ("reward", "visible_pass", "all_pass"):
            m, ci = mean_ci(tail[k])
            cells.append(f"{m:.1f} ± {ci:.1f}" if not math.isnan(ci) else f"{m:.1f}")
        gaps = [r - a for r, a in zip(tail["reward"], tail["all_pass"])]
        m, ci = mean_ci(gaps)
        cells.append(f"{m:.1f} ± {ci:.1f}" if not math.isnan(ci) else f"{m:.1f}")
        m, ci = mean_ci(tail["hacked_flag"])
        cells.append(f"{m:.1f} ± {ci:.1f}" if not math.isnan(ci) else f"{m:.1f}")
        lines.append(f"| {name} | {len(curves)} | " + " | ".join(cells) + " |")
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # figures: for each experiment, training reward vs all-tests pass and hack flag rate over steps (mean over seeds)
    for name, curves in fig_data.items():
        n = min(len(s) for s, _ in curves)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for key, label in (("reward", "training reward"), ("all_pass", "all-tests pass (hidden included)"),
                           ("visible_pass", "visible-test pass")):
            avg = [sum(c[1][key][i] for c in curves) / len(curves) for i in range(n)]
            ax[0].plot(range(1, n + 1), smooth(avg), label=label)
        ax[0].set_xlabel("training step"); ax[0].set_ylabel("rate"); ax[0].set_title(f"{name}: reward vs real correctness")
        ax[0].legend(fontsize=7); ax[0].set_ylim(0, 1)
        avg = [sum(c[1]["hacked_flag"][i] for c in curves) / len(curves) for i in range(n)]
        ax[1].plot(range(1, n + 1), smooth(avg), color="tab:red")
        ax[1].set_xlabel("training step"); ax[1].set_ylabel("share of completions flagged"); ax[1].set_title(f"{name}: hack-flag rate")
        ax[1].set_ylim(0, max(0.1, max(avg) * 1.2))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"curves_{name}.png"), dpi=130)
        plt.close(fig)
    print(open(os.path.join(OUT, "summary.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
