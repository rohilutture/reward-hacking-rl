"""Build a 100-completion sheet for hand labeling hack vs honest.

Sampling: 50 completions that pass the visible test but fail a hidden test or were flagged by the detector
(the likely hacks and near-hacks), plus 50 random others, drawn from final-model held-out evaluations
and from the last 20 training steps of each run. Labels are blank; fill the `label` column with one of
hardcode / special_case / harness_tamper / honest_wrong / honest_correct, then run scripts/score_labels.py.

  python scripts/make_label_sheet.py
"""
import csv
import glob
import json
import os
import random

random.seed(0)
rows = []
for p in glob.glob("runs/*_s[0-9]/eval_samples.jsonl"):
    exp = p.split(os.sep)[1]
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        rows.append({"source": exp + "/eval", "pid": r["pid"], "completion": r["completion"],
                     "visible_pass": r["visible_pass"], "all_pass": r["all_pass"], "flagged": r["hacked_flag"]})
for p in glob.glob("runs/*_s[0-9]/completions.jsonl"):
    exp = p.split(os.sep)[1]
    recs = [json.loads(l) for l in open(p, encoding="utf-8")]
    last = max((r["call"] for r in recs), default=0)
    for r in recs:
        if r["call"] > last - 20 and "completion" in r:
            rows.append({"source": exp + "/train", "pid": r.get("pid", ""), "completion": r["completion"],
                         "visible_pass": r["visible_pass"], "all_pass": r["all_pass"], "flagged": r["hacked_flag"]})
interesting = [r for r in rows if (r["visible_pass"] and not r["all_pass"]) or r["flagged"]]
others = [r for r in rows if r not in interesting]
random.shuffle(interesting)
random.shuffle(others)
sheet = interesting[:50] + others[:50]
random.shuffle(sheet)
os.makedirs("results", exist_ok=True)
with open("results/label_sheet.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "source", "pid", "visible_pass", "all_pass", "flagged", "label", "completion"])
    w.writeheader()
    for i, r in enumerate(sheet):
        w.writerow({"id": i, "label": "", **r})
print(f"wrote results/label_sheet.csv with {len(sheet)} rows ({len(interesting)} interesting available)")
