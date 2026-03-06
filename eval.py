#!/usr/bin/env python3
"""
Perseus Eval Pipeline
Reads from the test split (test.jsonl) and runs inference, reporting metrics.
Optionally accepts new C source files to process from scratch before evaluating.
"""

import json
import logging
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OBF_TYPES  = ['mba', 'control_flow', 'virtualization']
SYSTEM_PROMPT = (
    "You are a binary deobfuscation assistant. Given obfuscated x86-64 assembly code, "
    "you produce the equivalent clean, deobfuscated assembly. Preserve the function's "
    "semantics while removing obfuscation patterns such as MBA (mixed boolean-arithmetic), "
    "control flow flattening, and virtualization."
)


def build_prompt(example: dict) -> str:
    user_msg = example['instruction']
    if example.get('input'):
        user_msg += f"\n\n{example['input']}"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def line_accuracy(generated: str, expected: str) -> float:
    gen_lines = [l.strip() for l in generated.strip().splitlines()]
    exp_lines = [l.strip() for l in expected.strip().splitlines()]
    if not exp_lines:
        return 0.0
    matches = sum(g == e for g, e in zip(gen_lines, exp_lines))
    return matches / len(exp_lines)


def exact_match(generated: str, expected: str) -> bool:
    return generated.strip() == expected.strip()


class EvalPipeline:

    def __init__(self, data_root: Path, adapter_path: str, base_model: str = BASE_MODEL):
        self.data_root    = data_root
        self.adapter_path = adapter_path
        self.base_model   = base_model
        self.model        = None
        self.tokenizer    = None

    def load_test_data(self) -> list:
        test_path = self.data_root / 'training' / 'test.jsonl'
        if not test_path.exists():
            logger.error(f"test.jsonl not found at {test_path}. Run prepare_training_data.py first.")
            return []
        with open(test_path) as f:
            records = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(records)} test examples from {test_path}")
        return records

    def process_new_sources(self, c_files: list) -> list:
        """Process new C files through obfuscate→compile→disassemble→pair extraction."""
        from process_data import DataPipeline
        from prepare_training_data import TrainingDataPreparer
        from dataclasses import asdict

        pipeline = DataPipeline(self.data_root)
        preparer = TrainingDataPreparer(self.data_root)

        records = []
        for c_file in c_files:
            logger.info(f"Processing {c_file.name}...")
            pipeline.process_sample(Path(c_file), is_mal=False, obfuscation_types=OBF_TYPES)
            sample_name = Path(c_file).stem
            for obf_type in OBF_TYPES:
                pairs = preparer.create_training_pairs(sample_name, obf_type)
                logger.info(f"  {sample_name}/{obf_type}: {len(pairs)} pairs")
                records.extend([asdict(p) for p in pairs])

        return records

    def load_model(self):
        if self.model is not None:
            return
        logger.info(f"Loading base model: {self.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        logger.info(f"Loading LoRA adapter: {self.adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()

    def infer(self, record: dict) -> str:
        prompt = build_prompt(record)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

    def run(self, c_files: list = None):
        if c_files:
            records = self.process_new_sources(c_files)
        else:
            records = self.load_test_data()

        if not records:
            return

        self.load_model()

        results = []
        for i, record in enumerate(records):
            meta = record.get('metadata', {})
            logger.info(
                f"\n[{i+1}/{len(records)}] "
                f"{meta.get('sample')} / {meta.get('obfuscation_type')} / {meta.get('function')}"
            )
            generated = self.infer(record)
            expected  = record['output']

            em   = exact_match(generated, expected)
            lacc = line_accuracy(generated, expected)

            results.append({
                'sample':           meta.get('sample'),
                'obfuscation_type': meta.get('obfuscation_type'),
                'function':         meta.get('function'),
                'exact_match':      em,
                'line_accuracy':    lacc,
                'generated':        generated,
                'expected':         expected,
            })

            logger.info(f"  Exact match:   {em}")
            logger.info(f"  Line accuracy: {lacc:.1%}")

        self._print_summary(results)

        out_path = self.data_root / 'training' / 'eval_results.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {out_path}")

    def _print_summary(self, results: list):
        print("\n" + "="*60)
        print("EVAL SUMMARY")
        print("="*60)

        n = len(results)
        em_total   = sum(r['exact_match'] for r in results)
        lacc_total = sum(r['line_accuracy'] for r in results)
        print(f"Overall   — exact match: {em_total}/{n}  |  line accuracy: {lacc_total/n:.1%}")

        for obf in OBF_TYPES:
            subset = [r for r in results if r['obfuscation_type'] == obf]
            if not subset:
                continue
            em   = sum(r['exact_match'] for r in subset)
            lacc = sum(r['line_accuracy'] for r in subset) / len(subset)
            print(f"  {obf:<16} — exact match: {em}/{len(subset)}  |  line accuracy: {lacc:.1%}")

        print("="*60)
        print("\nPer-function results:")
        for r in results:
            status = "✓" if r['exact_match'] else "✗"
            print(
                f"  {status} {r['sample']}/{r['obfuscation_type']}/{r['function']}"
                f"  — line acc: {r['line_accuracy']:.1%}"
            )


def main():
    parser = argparse.ArgumentParser(description='Perseus Eval Pipeline')
    parser.add_argument('--adapter', type=str,
                        default='data/checkpoints/final',
                        help='Path to LoRA adapter (default: data/checkpoints/final)')
    parser.add_argument('--data-root', type=Path,
                        default=Path('./data'),
                        help='Root data directory (default: ./data)')
    parser.add_argument('--base-model', type=str, default=BASE_MODEL,
                        help=f'Base model name (default: {BASE_MODEL})')
    parser.add_argument('--c-files', nargs='+', type=Path, default=None,
                        help='Optional: C source files to process and evaluate instead of test.jsonl')
    args = parser.parse_args()

    pipeline = EvalPipeline(
        data_root=args.data_root,
        adapter_path=args.adapter,
        base_model=args.base_model,
    )
    pipeline.run(c_files=args.c_files)


if __name__ == '__main__':
    main()
