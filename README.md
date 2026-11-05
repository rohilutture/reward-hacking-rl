# Reward hacking in RL for code: a small negative-result study

GRPO fine-tuning of Qwen2.5-Coder-1.5B-Instruct (LoRA) on MBPP, with the reward computed from only part of the tests, and an AST detector plus sandbox to look for the model learning to game the tests instead of solving the task.

**Result: no reward hacking emerged.** Every reward setting improved real correctness by about 6-8 points and the visible-vs-hidden gap and the hack-flag rate stayed at the base model's level. This is a negative result for this scale, not evidence that the failure mode cannot occur.

## Setup
- Model: Qwen2.5-Coder-1.5B-Instruct, LoRA r=16 on all linear layers, GRPO (TRL 1.17.1), 80 steps, 8 prompts x 8 generations per step, lr 5e-5, no KL.
- Data: MBPP. The first test per problem is *visible* to the reward; the rest (plus challenge tests) are *hidden*. Evaluation: 300 held-out MBPP test problems, greedy decoding.
- Rewards: **A** = pass the visible test only (the weak, hackable reward); **B** = pass visible and hidden tests (the robust reward); **C** = pass all tests *and* not flagged by the hack detector.
- Stress test: **hardA / hardC** train only on the training problems the base model solves <=25% of the time (4 samples each), where cheating is most tempting.
- Hack detector (`rhack/rewards.py`): AST heuristics for special-casing the visible test's inputs, lookup tables, top-level constant returns, introspection, and `__eq__` overrides. Code runs in a Windows Job Object sandbox (512 MB, one process, timeout).
- 3 seeds per experiment; intervals are 95% t-intervals over seeds.

## Results (held-out MBPP, 300 problems, mean +- 95% CI over 3 seeds)

| experiment | visible-test pass | all-tests pass | visible-but-not-hidden | hack-flag rate |
|---|---|---|---|---|
| base model | 54.3 | 45.7 | 8.7 | 0.7 |
| A (weak reward) | 62.2 +- 10.0 | 53.9 +- 10.5 | 8.3 +- 1.7 | 0.7 +- 0.0 |
| B | 62.1 +- 6.0 | 53.7 +- 5.0 | 8.4 +- 2.7 | 0.7 +- 0.0 |
| C | 62.3 +- 7.2 | 53.8 +- 5.8 | 8.6 +- 2.5 | 0.7 +- 0.0 |
| hardA | 62.1 +- 7.5 | 52.4 +- 5.4 | 9.7 +- 2.2 | 0.7 +- 0.0 |
| hardC | 60.9 +- 8.9 | 51.9 +- 5.5 | 9.0 +- 3.8 | 0.7 +- 0.0 |

Training-time signals and curves are in `results/summary.md` and `results/curves_*.png`. Under reward A the training reward sits about 6 points above all-tests pass (the unavoidable gap from a one-test reward), and that gap did not grow over training.

The differences between A, B and C are inside the confidence intervals; I do not claim any ranking among them.

## Hand check of detector output
`results/label_sheet.csv` holds 100 sampled completions. I (the model that built this, not an independent human) read all 5 detector-flagged ones in the sheet: all were false positives (wrong but honest logic, or an echoed test call), none a real hack. I also keyword-scanned the 100 for `__eq__`, `sys`, `inspect`, `globals`, assert-on-test patterns and read the matches; none gamed the tests. I did **not** label all 100 one by one, so detector recall is not independently measured. The `label` column is blank.

## Limitations
- One small model, one dataset (MBPP), 80 training steps. Hacking may need a larger model, longer training, or a more exploitable environment.
- The detector is heuristic; its recall is unmeasured (see above). The visible-but-not-hidden rate is the detector-free signal, and it did not move either.
- The sandbox is a Windows Job Object with limited isolation, not a security boundary.
- Wide intervals (3 seeds) mean small effects could be missed.

## Reproduce
```
pip install -r requirements.txt
python scripts/eval_model.py --name base
powershell scripts/queue.ps1      # all runs sequentially (~13 h on one consumer GPU)
python scripts/analyze.py
pytest tests
```
