<div align="center">
  <img src="./Perseus.png" height="180">
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-yellow.svg?logo=python)
![Transformers](https://img.shields.io/badge/NLP-Transformers-F59E0B.svg?logo=huggingface)
![PEFT](https://img.shields.io/badge/Fine--Tuning-PEFT%2FLoRA-7A3EFF.svg)
![Model](https://img.shields.io/badge/Model-Qwen2.5--Coder--1.5B-0EA5E9.svg)
![Tigress](https://img.shields.io/badge/Obfuscator-Tigress-DC2626.svg)
![Status](https://img.shields.io/badge/🔬-Research-2563EB.svg)

</div>

## About

Perseus is a LoRA fine-tuned large language model for x86-64 binary deobfuscation. Given obfuscated assembly produced by the [Tigress C obfuscator](https://tigress.wtf/), Perseus attempts to recover clean, semantically equivalent assembly. The project serves as a proof-of-concept for applying instruction-tuned code models to the reverse engineering domain.

The pipeline handles everything from source collection and obfuscation through training data preparation, fine-tuning, and evaluation.

## Usage

### A. Installation

**1. Clone the Repo**
```bash
git clone git@github.com:aromans/Perseus.git
cd Perseus
```

**2. Environment**
A virtual environment is recommended.
```bash
python3 -m venv ~/Venvs/PerseusDev
source ~/Venvs/PerseusDev/bin/activate
pip install -r requirements.txt
```

**3. Tigress**

> [!IMPORTANT]
> Perseus requires **Tigress**, a source-to-source C obfuscator developed at the University of Arizona. Tigress is **not redistributable** and must be obtained directly from the authors before running the data pipeline.
> Visit [https://tigress.wtf](https://tigress.wtf/) for licensing and installation instructions.

Verify Tigress is available on your PATH before proceeding:
```bash
tigress --help
```

### B. Data Pipeline

All commands should be run from the project root (`Perseus/`). Use `--help` on any script for the full list of options.

**1. Collect Sources**
Place C source files in `data/source/benign/`. A small set of handwritten samples is included. To add your own, copy `.c` files there directly, or run the collector:
```bash
python3 run_pipeline.py --collect-only
```

**2. Process & Obfuscate**
Compile each source file, apply Tigress obfuscation transforms, and disassemble the resulting binaries.
```bash
python3 run_pipeline.py --skip-collection --obfuscations mba control_flow virtualization
```

> [!NOTE]
> Supported obfuscation types: `mba`, `control_flow`, `virtualization`. Defaults to all three.

> [!CAUTION]
> Virtualization obfuscation produces significantly larger binaries and can be slow to process, especially with many source files.

**3. Prepare Training Data**
Split processed samples into train/val/test sets and write JSONL files to `data/training/`.
```bash
python3 run_pipeline.py --skip-collection --prepare-training
```

> [!NOTE]
> Split ratios are configurable. Run `python3 src/prepare_training_data.py --help` for options including `--train-ratio`, `--val-ratio`, and `--test-ratio`.

Or run the full pipeline in one shot:
```bash
python3 run_pipeline.py --prepare-training
```

### C. Training

Fine-tune Qwen2.5-Coder-1.5B-Instruct using LoRA via the SFT trainer.
```bash
python3 train.py
```

> [!NOTE]
> Key training configuration (set in `config.yaml`):
> - Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (swap to 7B or Codestral by editing `config.yaml`)
> - LoRA: r=16, alpha=32, dropout=0.05 (~1.18% trainable parameters)
> - Optimizer: AdamW, LR=2e-4, cosine schedule, warmup_ratio=0.1
> - Checkpoints saved to `data/checkpoints/`

> [!IMPORTANT]
> Perseus automatically uses a GPU if one is available (with 4-bit quantization via bitsandbytes). If no GPU is detected, it falls back to CPU in float32. CPU training is significantly slower — a smaller model like the 1.5B is recommended in that case.

Training progress is logged to [Weights & Biases](https://wandb.ai) under the project name `Perseus`. To disable:
```bash
WANDB_DISABLED=true python3 train.py
```

### D. Evaluation

Evaluate a trained checkpoint against the test set.
```bash
python3 eval.py --checkpoint data/checkpoints/<checkpoint-name>
```

Results are written to `data/training/eval_results.json`. To view them side-by-side:
```bash
python3 show_eval.py
```

> [!TIP]
> Point `--checkpoint` at a specific epoch checkpoint rather than the final one. On small datasets, earlier checkpoints typically generalize better before overfitting sets in.

## Project Structure

```
Perseus/
├── src/
│   ├── collect_sources.py       # Source file collection
│   ├── process_data.py          # Obfuscation & disassembly
│   ├── prepare_training_data.py # Train/val/test split
│   ├── feature_selection.py     # Feature extraction
│   └── feature_comparison.py   # Feature comparison plots
├── data/
│   ├── source/benign/           # C source files
│   ├── training/                # train.jsonl, val.jsonl, test.jsonl, eval_results.json
│   └── checkpoints/             # LoRA adapter weights
├── results/
│   ├── plots/                   # Training and comparison charts
│   └── logs/                    # Training session logs
├── train.py                     # Fine-tuning entry point
├── eval.py                      # Evaluation entry point
├── run_pipeline.py              # Data pipeline entry point
├── show_eval.py                 # Eval results viewer
└── requirements.txt
```

## Acknowledgements

**Tigress C Obfuscator**
Obfuscation transformations powered by Tigress, developed at the University of Arizona.
```bibtex
@misc{tigress,
  author       = {Collberg, Christian},
  title        = {Tigress C Diversifier/Obfuscator},
  year         = {2022},
  howpublished = {\url{https://tigress.wtf}},
  note         = {University of Arizona}
}
```

**Qwen2.5-Coder**
Base language model provided by Alibaba Cloud.
```bibtex
@misc{hui2024qwen2,
  title  = {Qwen2.5-Coder Technical Report},
  author = {Hui, Binyuan and others},
  year   = {2024},
  url    = {https://arxiv.org/abs/2409.12186}
}
```

**Coding assistance from Claude by Anthropic**
```bibtex
@software{anthropic2026claude,
  author = {Anthropic},
  title  = {Claude (claude-sonnet-4-6)},
  year   = {2026},
  url    = {https://www.anthropic.com},
  note   = {Large language model used for code assistance}
}
```
