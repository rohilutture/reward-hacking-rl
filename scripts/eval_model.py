"""Evaluate a model (base, or base + LoRA adapter) on held-out MBPP test problems with greedy decoding.

  python scripts/eval_model.py --name base
  python scripts/eval_model.py --name A_s0 --adapter runs/A_s0/adapter

Writes runs/<name>/eval.json (rates) and runs/<name>/eval_samples.jsonl (every completion, for hand labeling).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rhack.generate import load, generate  # noqa: E402
from rhack.problems import load_mbpp  # noqa: E402
from rhack.rewards import _score_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    out_dir = os.path.join("runs", a.name)
    os.makedirs(out_dir, exist_ok=True)

    _, evals = load_mbpp()
    probs = evals[: a.n]
    tok, model = load(a.adapter)
    comps = [c[0] for c in generate(tok, model, probs, temperature=0.0)]
    del model

    recs = []
    for p, c in zip(probs, comps):
        _, rec = _score_one("A", c, json.dumps(p.visible), json.dumps(p.hidden), p.setup)
        rec.update({"pid": p.pid, "completion": c})
        recs.append(rec)
    n = len(recs)
    res = {
        "name": a.name, "n": n,
        "visible_pass": sum(r["visible_pass"] for r in recs) / n,
        "all_pass": sum(r["all_pass"] for r in recs) / n,
        "hack_flag_rate": sum(r["hacked_flag"] for r in recs) / n,
        "visible_not_hidden": sum(1 for r in recs if r["visible_pass"] and not r["all_pass"]) / n,
        "flagged_and_visible_pass_not_all": sum(1 for r in recs if r["hacked_flag"] and r["visible_pass"] and not r["all_pass"]) / n,
        "mean_completion_chars": sum(len(r["completion"]) for r in recs) / n,
    }
    json.dump(res, open(os.path.join(out_dir, "eval.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "eval_samples.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
