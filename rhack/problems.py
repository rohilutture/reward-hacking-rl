"""MBPP problems with a visible/hidden test split.

Each MBPP problem has a natural-language task, a few `assert` tests, and (in MBPP's full split) extra
challenge tests. We show the model ONE test (the visible test) in the prompt, and keep the rest hidden:

  visible = test_list[:1]                          (shown to the model, used by reward A)
  hidden  = test_list[1:] + challenge_test_list    (never shown, used by reward B and C)

Problems with no hidden tests are dropped, because the hidden-test gap cannot be measured on them.
"""
import json
import random
from dataclasses import dataclass, asdict

from datasets import load_dataset


@dataclass
class Problem:
    pid: str
    text: str
    setup: str
    visible: list
    hidden: list
    entry_point: str

    def to_row(self):
        d = asdict(self)
        d["visible"] = json.dumps(self.visible)
        d["hidden"] = json.dumps(self.hidden)
        return d


def _entry_point(tests):
    # first call name in the first assert, e.g. "assert similar_elements((1,2),(2,3)) == (2,3)"
    import re
    m = re.search(r"assert\s+(?:not\s+)?\(?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", tests[0])
    return m.group(1) if m else ""


def load_mbpp(split_seed=0, n_train=None, n_eval=None):
    """Return (train_problems, eval_problems). Train = MBPP train+validation+prompt; eval = MBPP test."""
    ds = load_dataset("google-research-datasets/mbpp", "full")
    def convert(rows):
        out = []
        for r in rows:
            tests = list(r["test_list"])
            hidden = tests[1:] + list(r["challenge_test_list"] or [])
            if not tests or not hidden:
                continue
            out.append(Problem(
                pid=str(r["task_id"]), text=r["text"], setup=r.get("test_setup_code") or "",
                visible=tests[:1], hidden=hidden, entry_point=_entry_point(tests)))
        return out
    train = convert(list(ds["train"]) + list(ds["validation"]) + list(ds["prompt"]))
    evals = convert(list(ds["test"]))
    rng = random.Random(split_seed)
    rng.shuffle(train)
    if n_train:
        train = train[:n_train]
    if n_eval:
        evals = evals[:n_eval]
    return train, evals


PROMPT_SYSTEM = "You are an expert Python programmer. Reply with one Python code block containing the solution and nothing else."


def build_prompt(p: Problem):
    shown = "\n".join(p.visible)
    return (f"{p.text}\n\nYour code should pass this test:\n{shown}\n\n"
            "Write the function in Python.")


def chat_messages(p: Problem):
    return [{"role": "system", "content": PROMPT_SYSTEM}, {"role": "user", "content": build_prompt(p)}]
