"""Find the training problems the base model rarely solves (for the clearly labeled stress test).

Samples 4 completions per training problem from the base model, scores the visible test, and writes
runs/difficulty/hard_ids.txt: problems whose visible-test pass rate is at most 25%.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rhack.generate import load, generate  # noqa: E402
from rhack.problems import load_mbpp  # noqa: E402
from rhack.rewards import _score_one  # noqa: E402


def main():
    out = os.path.join("runs", "difficulty")
    os.makedirs(out, exist_ok=True)
    train, _ = load_mbpp()
    tok, model = load()
    comps = generate(tok, model, train, samples=4, temperature=0.8, batch_problems=16)
    del model
    hard, rates = [], {}
    for p, cs in zip(train, comps):
        ok = [_score_one("A", c, json.dumps(p.visible), json.dumps(p.hidden), p.setup)[1]["visible_pass"] for c in cs]
        rates[p.pid] = sum(ok) / len(ok)
        if rates[p.pid] <= 0.25:
            hard.append(p.pid)
    open(os.path.join(out, "hard_ids.txt"), "w").write("\n".join(hard))
    json.dump(rates, open(os.path.join(out, "visible_pass_rates.json"), "w"))
    print(f"{len(hard)} hard problems of {len(train)}; mean visible pass {sum(rates.values()) / len(rates):.3f}")


if __name__ == "__main__":
    main()
