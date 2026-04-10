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
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import load_config, OBF_TYPES, SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_prompt(example: dict, tokenizer) -> str:
    user_msg = example['instruction']
    if example.get('input'):
        user_msg += f"\n\n{example['input']}"
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def line_metrics(generated: str, expected: str) -> tuple[float, float, float]:
    """Returns (precision, recall, f1) at the line level using positional matching."""
    gen_lines = [l.strip() for l in generated.strip().splitlines() if l.strip()]
    exp_lines = [l.strip() for l in expected.strip().splitlines() if l.strip()]
    if not gen_lines and not exp_lines:
        return 1.0, 1.0, 1.0
    if not gen_lines or not exp_lines:
        return 0.0, 0.0, 0.0
    matches = sum(g == e for g, e in zip(gen_lines, exp_lines))
    precision = matches / len(gen_lines)
    recall    = matches / len(exp_lines)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def exact_match(generated: str, expected: str) -> bool:
    return generated.strip() == expected.strip()


class EvalPipeline:

    def __init__(self, data_root: Path, adapter_path: str, base_model: str,
                 max_new_tokens: int, temperature: float,
                 use_wandb: bool = False, wandb_project: str = "Perseus"):
        self.data_root      = data_root
        self.adapter_path   = adapter_path
        self.base_model     = base_model
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.use_wandb      = use_wandb
        self.wandb_project  = wandb_project
        self.model          = None
        self.tokenizer      = None

    def load_test_data(self, max_samples: int = None) -> list:
        test_path = self.data_root / 'training' / 'test.jsonl'
        if not test_path.exists():
            logger.error(f"test.jsonl not found at {test_path}. Run prepare_training_data.py first.")
            return []
        with open(test_path) as f:
            records = [json.loads(line) for line in f]

        if max_samples and max_samples < len(records):
            # Stratified sample: preserve obfuscation type distribution
            import random
            from collections import defaultdict
            random.seed(1337)
            by_obf = defaultdict(list)
            for r in records:
                by_obf[r['metadata']['obfuscation_type']].append(r)
            obf_types = sorted(by_obf.keys())
            per_type = max_samples // len(obf_types)
            remainder = max_samples % len(obf_types)
            sampled = []
            for i, obf in enumerate(obf_types):
                n = per_type + (1 if i < remainder else 0)
                sampled.extend(random.sample(by_obf[obf], min(n, len(by_obf[obf]))))
            records = sampled
            logger.info(f"Stratified sample: {len(records)} test examples "
                        f"({', '.join(f'{t}:{len([r for r in records if r[\"metadata\"][\"obfuscation_type\"]==t])}' for t in obf_types)})")
        else:
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

        has_gpu = torch.cuda.is_available()
        if has_gpu:
            logger.info("GPU detected — loading with 4-bit quantization")
            from transformers import BitsAndBytesConfig
            model_kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                ),
                "device_map": "auto",
            }
        else:
            logger.info("No GPU detected — loading in float32 on CPU")
            model_kwargs = {
                "torch_dtype": torch.float32,
                "device_map": "cpu",
            }

        logger.info(f"Loading base model: {self.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            **model_kwargs,
        )
        if self.adapter_path:
            logger.info(f"Loading LoRA adapter: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        else:
            logger.info("No adapter specified — running zero-shot baseline")
        self.model.eval()

    def infer(self, record: dict) -> str:
        prompt = build_prompt(record, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

    def run_asm_file(self, asm_path: Path, label: str = None, out_path: Path = None):
        """
        Qualitative inference on a raw assembly file — no ground truth, no metrics.
        Designed for real-world samples (e.g. Pikabot, GuLoader) where the analyst
        evaluates the output manually.
        """
        asm_text = asm_path.read_text().strip()
        if not asm_text:
            logger.error(f"Empty file: {asm_path}")
            return

        name = label or asm_path.stem
        instruction = f"Deobfuscate the following mba-obfuscated x86-64 assembly function '{name}'."
        record = {"instruction": instruction, "input": asm_text, "output": ""}

        self.load_model()

        logger.info(f"\n[qualitative] {name}  ({asm_text.count(chr(10))+1} lines)")
        generated = self.infer(record)

        print("\n" + "="*60)
        print(f"INPUT  ({name})")
        print("="*60)
        print(asm_text)
        print("\n" + "="*60)
        print(f"PERSEUS OUTPUT")
        print("="*60)
        print(generated)
        print("="*60)

        result = {"sample": name, "input": asm_text, "generated": generated}
        save_path = out_path or (asm_path.parent / f"{asm_path.stem}_perseus.json")
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved to {save_path}")

    def run(self, c_files: list = None, max_samples: int = None):
        if c_files:
            records = self.process_new_sources(c_files)
        else:
            records = self.load_test_data(max_samples=max_samples)

        if not records:
            return

        self.load_model()

        out_path = self.data_root / 'training' / 'eval_results.json'
        results = []
        start_time = time.monotonic()
        for i, record in enumerate(records):
            meta = record.get('metadata', {})
            logger.info(
                f"\n[{i+1}/{len(records)}] "
                f"{meta.get('sample')} / {meta.get('obfuscation_type')} / {meta.get('function')}"
            )
            generated = self.infer(record)
            expected  = record['output']

            em                    = exact_match(generated, expected)
            precision, recall, f1 = line_metrics(generated, expected)

            results.append({
                'sample':           meta.get('sample'),
                'obfuscation_type': meta.get('obfuscation_type'),
                'function':         meta.get('function'),
                'exact_match':      em,
                'line_precision':   precision,
                'line_recall':      recall,
                'line_f1':          f1,
                'generated':        generated,
                'expected':         expected,
            })

            elapsed = time.monotonic() - start_time
            avg_per_example = elapsed / (i + 1)
            remaining = avg_per_example * (len(records) - i - 1)
            eta = datetime.now() + timedelta(seconds=remaining)
            logger.info(f"  Exact match: {em}")
            logger.info(f"  Precision:   {precision:.1%}  Recall: {recall:.1%}  F1: {f1:.1%}")
            logger.info(f"  ETA:         {eta.strftime('%H:%M:%S')}  ({timedelta(seconds=int(remaining))} remaining)")

            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

        self._print_summary(results, use_wandb=self.use_wandb)
        logger.info(f"\nResults saved to {out_path}")

    def _print_summary(self, results: list, use_wandb: bool = False):
        print("\n" + "="*60)
        print("EVAL SUMMARY")
        print("="*60)

        n = len(results)
        em_total = sum(r['exact_match'] for r in results)
        avg_p    = sum(r['line_precision'] for r in results) / n
        avg_r    = sum(r['line_recall']    for r in results) / n
        avg_f1   = sum(r['line_f1']        for r in results) / n
        print(f"Overall   — exact: {em_total}/{n}  |  P: {avg_p:.1%}  R: {avg_r:.1%}  F1: {avg_f1:.1%}")

        wandb_metrics = {
            "eval/exact_match_pct": em_total / n,
            "eval/line_precision":  avg_p,
            "eval/line_recall":     avg_r,
            "eval/line_f1":         avg_f1,
        }

        for obf in OBF_TYPES:
            subset = [r for r in results if r['obfuscation_type'] == obf]
            if not subset:
                continue
            em  = sum(r['exact_match']    for r in subset)
            p   = sum(r['line_precision'] for r in subset) / len(subset)
            r_  = sum(r['line_recall']    for r in subset) / len(subset)
            f1  = sum(r['line_f1']        for r in subset) / len(subset)
            print(f"  {obf:<16} — exact: {em}/{len(subset)}  |  P: {p:.1%}  R: {r_:.1%}  F1: {f1:.1%}")
            wandb_metrics[f"eval/{obf}/line_precision"] = p
            wandb_metrics[f"eval/{obf}/line_recall"]    = r_
            wandb_metrics[f"eval/{obf}/line_f1"]        = f1
            wandb_metrics[f"eval/{obf}/exact_match_pct"] = em / len(subset)

        print("="*60)
        print("\nPer-function results:")
        for r in results:
            status = "✓" if r['exact_match'] else "✗"
            print(
                f"  {status} {r['sample']}/{r['obfuscation_type']}/{r['function']}"
                f"  — P: {r['line_precision']:.1%}  R: {r['line_recall']:.1%}  F1: {r['line_f1']:.1%}"
            )

        if use_wandb:
            import wandb
            if wandb.run is None:
                wandb.init(project=self.wandb_project, job_type="eval")
            wandb.log(wandb_metrics)
            logger.info("Logged eval metrics to wandb.")


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default='config.yaml')
    pre_args, _ = pre.parse_known_args()

    cfg = load_config(pre_args.config)
    model_default = cfg.get('model', {}).get('name', 'Qwen/Qwen2.5-Coder-1.5B-Instruct')
    inf = cfg.get('inference', {})

    parser = argparse.ArgumentParser(description='Perseus Eval Pipeline')
    parser.add_argument('--config', default='config.yaml',
                        help='Path to config.yaml (default: config.yaml)')
    parser.add_argument('--adapter', type=str,
                        default=None,
                        help='Path to LoRA adapter. Omit for zero-shot baseline.')
    parser.add_argument('--data-root', type=Path,
                        default=Path('./data'),
                        help='Root data directory (default: ./data)')
    parser.add_argument('--base-model', type=str,
                        default=model_default,
                        help='Base model name (default: from config.yaml)')
    parser.add_argument('--max-new-tokens', type=int,
                        default=inf.get('max_new_tokens', 1024),
                        help='Max tokens to generate per sample (default: from config.yaml)')
    parser.add_argument('--temperature', type=float,
                        default=inf.get('temperature', 0.1),
                        help='Sampling temperature (default: from config.yaml)')
    parser.add_argument('--c-files', nargs='+', type=Path, default=None,
                        help='Optional: C source files to process and evaluate instead of test.jsonl')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max test samples to evaluate. Stratified by obfuscation type. '
                             'Omit to run the full test set.')
    parser.add_argument('--asm-file', type=Path, default=None,
                        help='Raw assembly file for qualitative inference (no ground truth). '
                             'Use with real-world samples — output is for human RE evaluation.')
    parser.add_argument('--asm-label', type=str, default=None,
                        help='Label for the assembly sample (e.g. "pikabot_string_decrypt"). '
                             'Used in the prompt and output filename.')
    parser.add_argument('--wandb', action='store_true',
                        help='Log eval metrics to Weights & Biases')
    args = parser.parse_args()

    wb = cfg.get('wandb', {})
    pipeline = EvalPipeline(
        data_root=args.data_root,
        adapter_path=args.adapter,
        base_model=args.base_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_wandb=args.wandb,
        wandb_project=wb.get('project', 'Perseus'),
    )

    if args.asm_file:
        pipeline.run_asm_file(args.asm_file, label=args.asm_label)
    else:
        pipeline.run(c_files=args.c_files, max_samples=args.max_samples)


if __name__ == '__main__':
    main()
