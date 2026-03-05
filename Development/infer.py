#!/usr/bin/env python3

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "perseus_data/checkpoints/final"
VAL_DATA = "perseus_data/training/val.jsonl"

SYSTEM_PROMPT = (
    "You are a binary deobfuscation assistant. Given obfuscated x86-64 assembly code, "
    "you produce the equivalent clean, deobfuscated assembly. Preserve the function's "
    "semantics while removing obfuscation patterns such as MBA (mixed boolean-arithmetic), "
    "control flow flattening, and virtualization."
)

def load_val(path, idx=0):
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    return lines[idx]

def build_prompt(example):
    user_msg = example['instruction']
    if example.get('input'):
        user_msg += f"\n\n{example['input']}"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    val_data = load_val(VAL_DATA, idx=0)
    prompt = build_prompt(val_data)

    meta = val_data.get('metadata', {})
    print(f"\n=== Input ===")
    print(f"Function : {meta.get('function', '?')}")
    print(f"Obfuscation: {meta.get('obfuscation_type', '?')}")
    print(f"Instructions: {meta.get('obf_instruction_count', '?')} obf → {meta.get('clean_instruction_count', '?')} clean")
    print(f"\nObfuscated assembly:\n{val_data['input']}")

    print("\n=== Generating... ===")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    print(f"\n=== Model Output (full) ===\n{generated}")
    print(f"\n=== Expected Output ===\n{val_data['output']}")

if __name__ == "__main__":
    main()
