"""Batched generation from the base model, optionally with a LoRA adapter."""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .problems import chat_messages

MODEL = os.path.join("models", "qwen2.5-coder-1.5b-instruct")


def load(adapter=None):
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    return tok, model.eval()


def generate(tok, model, problems, samples=1, temperature=0.0, max_new_tokens=384, batch_problems=16):
    """Return list (per problem) of lists of completion strings."""
    out = []
    for i in range(0, len(problems), batch_problems):
        chunk = problems[i:i + batch_problems]
        prompts = [tok.apply_chat_template(chat_messages(p), tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=0.95, num_return_sequences=samples)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            gen = model.generate(**enc, **kw)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        per = samples if temperature > 0 else 1
        for j in range(len(chunk)):
            out.append(texts[j * per:(j + 1) * per])
    return out
