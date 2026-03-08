<div align="center">
  <img src="./Perseus.png" height="180">
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-yellow.svg?logo=python)
![PyTorch](https://img.shields.io/badge/ML-PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/NLP-Transformers-F59E0B.svg?logo=huggingface)
![PEFT](https://img.shields.io/badge/Fine--Tuning-PEFT%2FLoRA-7A3EFF.svg)
![Model](https://img.shields.io/badge/Model-Qwen2.5--Coder--1.5B-0EA5E9.svg)
![CUDA](https://img.shields.io/badge/Compute-CUDA-16A34A.svg?logo=nvidia&logoColor=white)
![Tigress](https://img.shields.io/badge/Obfuscator-Tigress-DC2626.svg)
![WandB](https://img.shields.io/badge/Tracking-W%26B-FFBE00.svg?logo=weightsandbiases&logoColor=black)
![Status](https://img.shields.io/badge/🔬-Research-2563EB.svg)

</div>

## About

Perseus is a LoRA fine-tuned large language model for x86-64 binary deobfuscation. Given obfuscated assembly Perseus attempts to recover clean, semantically equivalent assembly. Perseus investigates the feasibility of adapting instruction-tuned code models for binary deobfuscation, a core challenge in static malware analysis where obfuscation is routinely used to evade detection and complicate attribution.

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

**AnghaBench sampling**

Perseus can automatically sample functions from [AnghaBench](https://github.com/brenocfg/AnghaBench), a large corpus of real-world C functions extracted from open-source projects. Enable it in `config.yaml`:

```yaml
anghabench:
  enabled: true
  repo_dir: "repos/AnghaBench"   # auto-cloned if not present
  n_samples: 100                 # number of functions to sample
  min_lines: 10                  # minimum lines of code filter
  seed: 42                       # seed for reproducible sampling
```

With `enabled: true`, running `--collect-only` will clone AnghaBench (first run only) and sample the configured number of compilable functions into `data/source/benign/`. The same seed produces the same sample every time.

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
python3 run_pipeline.py --prepare-training-only
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

**Training on a cloud GPU**

If training on a remote machine, use `export_adapters.py` to package your checkpoints for transfer once training completes. Only the LoRA adapter files are included — the base model is not transferred and will be re-downloaded from HuggingFace on first inference.

```bash
# Package all saved checkpoints
python3 export_adapters.py

# Or package only the latest checkpoint
python3 export_adapters.py --latest

# Or target a specific checkpoint
python3 export_adapters.py --checkpoint data/checkpoints/checkpoint-20
```

This produces a `perseus_adapters_<timestamp>.tar.gz` archive. Transfer it to your local machine and extract into the project:

```bash
tar -xzf perseus_adapters_<timestamp>.tar.gz -C data/
```

Adapters will land in `data/checkpoints/` and are immediately usable with `eval.py`.

> [!NOTE]
> All training configuration lives in `config.yaml` — no code changes needed for most experiments:
> - **Model**: swap `model.name` to use a different base model (e.g. `Qwen/Qwen2.5-Coder-7B-Instruct`, `mistralai/Codestral-22B-v0.1`)
> - **LoRA**: rank, alpha, dropout, and `target_modules` (update target modules when changing model architectures)
> - **Training**: epochs, batch size, learning rate, scheduler, grad norm, warmup, weight decay
> - **Checkpoints**: save strategy and how many to keep (`checkpoints.save_total_limit`)
> - **Inference**: `max_new_tokens` and `temperature` used during eval

> [!IMPORTANT]
> Perseus automatically uses a GPU if one is available (with 4-bit quantization via bitsandbytes). If no GPU is detected, it falls back to CPU in float32. CPU training is significantly slower — a smaller model like the 1.5B is recommended in that case.

Training progress is logged to [Weights & Biases](https://wandb.ai) under the project name `Perseus`. To disable:
```bash
WANDB_DISABLED=true python3 train.py
```

### D. Evaluation

Evaluate a trained checkpoint against the test set.
```bash
python3 eval.py --adapter data/checkpoints/<checkpoint-name>
```

Results are written to `data/training/eval_results.json`. To view them side-by-side:
```bash
python3 show_eval.py
```

> [!TIP]
> Point `--adapter` at a specific epoch checkpoint rather than the final one. On small datasets, earlier checkpoints typically generalize better before overfitting sets in.

## Project Structure

```
Perseus/
├── src/
│   ├── config.py                # Shared constants and config loader
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
├── export_adapters.py           # Package checkpoints for transfer from cloud GPUs
├── config.yaml                  # All configuration — model, LoRA, training, inference
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
