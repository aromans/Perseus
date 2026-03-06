#!/usr/bin/env python3

import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from config import load_config, SYSTEM_PROMPT

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Sourced from config.yaml — no defaults here to avoid duplication
    model_name: str
    num_epochs: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    max_seq_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    logging_steps: int
    eval_samples: int
    save_strategy: str
    save_total_limit: int
    # Paths
    train_data: str = "data/training/train.jsonl"
    val_data: str = "data/training/val.jsonl"
    output_dir: str = "data/checkpoints"
    # Flags
    use_wandb: bool = False
    dry_run: bool = False


class PerseusTrainer:

    def __init__(self, config: TrainConfig):
        self.config = config

    def load_data(self) -> tuple[Dataset, Optional[Dataset]]:
        def load_jsonl(path):
            records = []
            with open(path) as f:
                for line in f:
                    records.append(json.loads(line))
            return records

        train_records = load_jsonl(self.config.train_data)
        logger.info(f"Loaded {len(train_records)} training examples")

        val_records = None
        if Path(self.config.val_data).exists():
            val_records = load_jsonl(self.config.val_data)
            logger.info(f"Loaded {len(val_records)} validation examples")

        train_dataset = Dataset.from_list([
            {"text": self.format_prompt(r)} for r in train_records
        ])

        val_dataset = None
        if val_records:
            val_dataset = Dataset.from_list([
                {"text": self.format_prompt(r)} for r in val_records
            ])

        return train_dataset, val_dataset

    def format_prompt(self, example: dict) -> str:
        user_msg = example['instruction']
        if example.get('input'):
            user_msg += f"\n\n{example['input']}"

        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}<|im_end|>"
        )

    def setup_model(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        has_gpu = torch.cuda.is_available()

        if has_gpu:
            logger.info("GPU detected — using 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs = {
                "quantization_config": bnb_config,
                "device_map": "auto",
            }
        else:
            logger.info("No GPU detected — loading in float32 on CPU")
            logger.info(
                "Note: training a 7B model on CPU will be very slow. "
                "Consider using a smaller model (e.g., Qwen/Qwen2.5-Coder-1.5B-Instruct) "
                "or running on a machine with a GPU."
            )
            model_kwargs = {
                "torch_dtype": torch.float32,
                "device_map": "cpu",
            }

        logger.info(f"Loading model: {self.config.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            **model_kwargs,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        tokenizer.model_max_length = self.config.max_seq_length

        if has_gpu:
            model = prepare_model_for_kbit_training(model)

        return model, tokenizer

    def setup_lora(self, model) -> AutoModelForCausalLM:
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable parameters: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

        return model

    def train(self):
        train_dataset, val_dataset = self.load_data()
        model, tokenizer = self.setup_model()
        model = self.setup_lora(model)

        if self.config.dry_run:
            logger.info("Dry run — verifying data pipeline...")
            sample = train_dataset[0]['text']
            tokens = tokenizer(sample, return_tensors="pt")
            logger.info(f"Sample token count: {tokens['input_ids'].shape[1]}")
            logger.info(f"Sample text (first 500 chars):\n{sample[:500]}")
            logger.info("Dry run complete — model and data pipeline verified.")
            return

        report_to = "wandb" if self.config.use_wandb else "none"

        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            logging_steps=self.config.logging_steps,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
            report_to=report_to,
            dataset_text_field="text",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            packing=False,
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
        )

        logger.info("Starting training...")
        trainer.train()

        final_dir = Path(self.config.output_dir) / "final"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Saved final adapter to {final_dir}")

        if val_dataset and self.config.eval_samples > 0:
            self._generate_samples(model, tokenizer, val_dataset)

    def _generate_samples(self, model, tokenizer, val_dataset):
        logger.info("\n=== Sample Generations ===")
        model.eval()

        n = min(self.config.eval_samples, len(val_dataset))
        for i in range(n):
            text = val_dataset[i]['text']

            # Extract just the prompt (everything before assistant response)
            prompt_end = text.find("<|im_start|>assistant\n")
            if prompt_end == -1:
                continue
            prompt = text[:prompt_end] + "<|im_start|>assistant\n"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            expected_start = prompt_end + len("<|im_start|>assistant\n")
            expected_end = text.find("<|im_end|>", expected_start)
            expected = text[expected_start:expected_end] if expected_end != -1 else ""

            logger.info(f"\n--- Sample {i+1} ---")
            logger.info(f"Generated (first 300 chars):\n{generated[:300]}")
            logger.info(f"Expected (first 300 chars):\n{expected[:300]}")


def main():
    # Pre-parse to find --config before building the full parser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default='config.yaml')
    pre_args, _ = pre.parse_known_args()

    cfg = load_config(pre_args.config)
    m   = cfg.get('model', {})
    lo  = cfg.get('lora', {})
    tr  = cfg.get('training', {})
    ck  = cfg.get('checkpoints', {})

    parser = argparse.ArgumentParser(description='Perseus Deobfuscation Model Training')

    parser.add_argument('--config', default='config.yaml',
                        help='Path to config.yaml (default: config.yaml)')
    parser.add_argument('--model', type=str,
                        default=m.get('name', 'Qwen/Qwen2.5-Coder-1.5B-Instruct'),
                        help='Model name/path')
    parser.add_argument('--train-data', type=str, default='data/training/train.jsonl',
                        help='Path to training JSONL')
    parser.add_argument('--val-data', type=str, default='data/training/val.jsonl',
                        help='Path to validation JSONL')
    parser.add_argument('--output-dir', type=str, default='data/checkpoints',
                        help='Checkpoint output directory')
    parser.add_argument('--epochs', type=int,
                        default=tr.get('epochs', 30),
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                        default=tr.get('batch_size', 1),
                        help='Per-device batch size')
    parser.add_argument('--lr', type=float,
                        default=tr.get('learning_rate', 2e-4),
                        help='Learning rate')
    parser.add_argument('--max-seq-length', type=int,
                        default=tr.get('max_seq_length', 4096),
                        help='Maximum sequence length')
    parser.add_argument('--lora-r', type=int,
                        default=lo.get('r', 16),
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int,
                        default=lo.get('alpha', 32),
                        help='LoRA alpha')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--dry-run', action='store_true',
                        help='Verify model and data pipeline without training')

    args = parser.parse_args()

    config = TrainConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=tr.get('gradient_accumulation', 4),
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=lo.get('dropout', 0.05),
        lora_target_modules=lo.get('target_modules', [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ]),
        warmup_ratio=tr.get('warmup_ratio', 0.1),
        weight_decay=tr.get('weight_decay', 0.01),
        max_grad_norm=tr.get('max_grad_norm', 0.3),
        lr_scheduler_type=tr.get('lr_scheduler_type', 'cosine'),
        logging_steps=tr.get('logging_steps', 1),
        eval_samples=tr.get('eval_samples', 2),
        save_strategy=ck.get('save_strategy', 'epoch'),
        save_total_limit=ck.get('save_total_limit', 3),
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        dry_run=args.dry_run,
    )

    trainer = PerseusTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
