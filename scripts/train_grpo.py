"""GRPO training of Qwen2.5-Coder-1.5B-Instruct (LoRA) on MBPP with reward A, B or C.

  python scripts/train_grpo.py --reward A --seed 0 --steps 80 --name A_s0

Per-completion records (reward, visible/hidden pass, hack flags, training step) go to runs/<name>/completions.jsonl,
which is what the training-reward-vs-hidden-pass curves are made from.
"""
import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rhack.problems import load_mbpp, chat_messages  # noqa: E402
from rhack.rewards import score_completions  # noqa: E402

MODEL = os.path.join("models", "qwen2.5-coder-1.5b-instruct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["A", "B", "C"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--name", required=True)
    ap.add_argument("--problem_ids", help="Text file with one MBPP task id per line (restrict training problems).")
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=384)
    ap.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args()

    out_dir = os.path.join("runs", a.name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "completions.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)

    train, _ = load_mbpp()
    if a.problem_ids:
        keep = {l.strip() for l in open(a.problem_ids) if l.strip()}
        train = [p for p in train if p.pid in keep]
    print(f"training problems: {len(train)}  reward {a.reward}  seed {a.seed}", flush=True)
    rows = [{"prompt": chat_messages(p), "pid": p.pid, "visible": json.dumps(p.visible),
             "hidden": json.dumps(p.hidden), "setup": p.setup} for p in train]
    ds = Dataset.from_list(rows)

    state = {"calls": 0}

    def reward_fn(prompts, completions, visible, hidden, setup, **kwargs):
        state["calls"] += 1
        return score_completions(a.reward, completions, visible, hidden, setup, log_path=log_path,
                                 extra={"call": state["calls"], "reward_kind": a.reward})

    cfg = GRPOConfig(
        output_dir=out_dir,
        max_steps=a.steps,
        learning_rate=a.lr,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=(a.prompts_per_step * a.num_generations) // 16,
        num_generations=a.num_generations,
        max_completion_length=a.max_completion_length,
        temperature=1.0,
        beta=0.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=a.seed,
        data_seed=a.seed,
        model_init_kwargs={"torch_dtype": torch.bfloat16},
        remove_unused_columns=False,
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = GRPOTrainer(model=MODEL, reward_funcs=reward_fn, args=cfg, train_dataset=ds, peft_config=lora)
    trainer.train()
    trainer.save_model(os.path.join(out_dir, "adapter"))
    json.dump(trainer.state.log_history, open(os.path.join(out_dir, "log_history.json"), "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
