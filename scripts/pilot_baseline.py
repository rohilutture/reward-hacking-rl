"""Sample completions from the base model on a few problems and score them with every reward.

  python scripts/pilot_baseline.py --n_problems 40 --samples 4 --out runs/pilot

Writes runs/pilot/samples.jsonl (one row per completion) and prints the summary.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rhack.problems import load_mbpp, chat_messages  # noqa: E402
from rhack.rewards import score_completions, hack_flags, is_hacked  # noqa: E402
from rhack.sandbox import extract_code  # noqa: E402

MODEL = os.path.join("models", "qwen2.5-coder-1.5b-instruct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_problems", type=int, default=40)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", default="runs/pilot")
    ap.add_argument("--split", choices=["train", "eval"], default="train")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    train, evals = load_mbpp(n_train=None)
    probs = (train if a.split == "train" else evals)[: a.n_problems]

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()

    rows = []
    t0 = time.time()
    for p in probs:
        prompt = tok.apply_chat_template(chat_messages(p), tokenize=False, add_generation_prompt=True)
        enc = tok([prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=a.temperature, top_p=0.95,
                                 max_new_tokens=a.max_new_tokens, num_return_sequences=a.samples,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for t in texts:
            rows.append({"pid": p.pid, "completion": t, "visible": json.dumps(p.visible),
                         "hidden": json.dumps(p.hidden), "setup": p.setup})
    print(f"generated {len(rows)} completions in {time.time() - t0:.0f}s", flush=True)

    del model
    torch.cuda.empty_cache()

    comp = [r["completion"] for r in rows]
    vis = [r["visible"] for r in rows]
    hid = [r["hidden"] for r in rows]
    setup = [r["setup"] for r in rows]
    log = os.path.join(a.out, "scores.jsonl")
    if os.path.exists(log):
        os.remove(log)
    rA = score_completions("A", comp, vis, hid, setup, log_path=log)
    rB = score_completions("B", comp, vis, hid, setup)
    rC = score_completions("C", comp, vis, hid, setup)
    recs = [json.loads(l) for l in open(log, encoding="utf-8")]
    for r, a_, b_, c_, rec in zip(rows, rA, rB, rC, recs):
        r.update({"rewardA": a_, "rewardB": b_, "rewardC": c_, **{k: rec[k] for k in ("visible_pass", "all_pass", "hacked_flag", "flags", "error")}})
    with open(os.path.join(a.out, "samples.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    def pct(k):
        return 100 * sum(1 for r in rows if r[k]) / n
    print(f"n={n} problems={len(probs)}")
    print(f"visible pass   {pct('visible_pass'):.1f}%")
    print(f"all tests pass {pct('all_pass'):.1f}%")
    print(f"hack flagged   {pct('hacked_flag'):.1f}%")
    print(f"reward A mean  {100 * sum(r['rewardA'] for r in rows) / n:.1f}%")
    print(f"reward B mean  {100 * sum(r['rewardB'] for r in rows) / n:.1f}%")
    print(f"reward C mean  {100 * sum(r['rewardC'] for r in rows) / n:.1f}%")
    gap = [r for r in rows if r["visible_pass"] and not r["all_pass"]]
    print(f"visible-but-not-hidden (A reward, no B reward): {len(gap)}")
    print(f"flagged but all-pass (possible hack that passes hidden tests): "
          f"{sum(1 for r in rows if r['hacked_flag'] and r['all_pass'])}")


if __name__ == "__main__":
    main()
