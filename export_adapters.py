#!/usr/bin/env python3

import argparse
import os
import sys
import tarfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from config import load_config

# Files within a checkpoint dir needed for inference (adapter weights + config)
ADAPTER_FILES = {
    'adapter_config.json',
    'adapter_model.safetensors',
    'adapter_model.bin',        # older PEFT versions use .bin
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'vocab.json',
    'merges.txt',
}


def find_checkpoints(checkpoints_dir: Path) -> list[Path]:
    if not checkpoints_dir.exists():
        print(f"ERROR: Checkpoints directory not found: {checkpoints_dir}")
        sys.exit(1)

    checkpoints = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],
        key=lambda d: int(d.name.split('-')[-1])
    )

    # Also include a final model directory if present (saved outside checkpoint-N dirs)
    final = checkpoints_dir / 'final'
    if final.exists():
        checkpoints.append(final)

    return checkpoints


def package_checkpoints(checkpoints: list[Path], config_path: Path, output_path: Path):
    print(f"\nPackaging {len(checkpoints)} checkpoint(s) into {output_path.name}...")

    with tarfile.open(output_path, 'w:gz') as tar:
        # Always include config.yaml so inference knows the base model + LoRA settings
        if config_path.exists():
            tar.add(config_path, arcname='config.yaml')
            print(f"  + config.yaml")

        for ckpt_dir in checkpoints:
            included = []
            for f in ckpt_dir.iterdir():
                if f.name in ADAPTER_FILES:
                    arcname = str(Path('checkpoints') / ckpt_dir.name / f.name)
                    tar.add(f, arcname=arcname)
                    included.append(f.name)

            if included:
                print(f"  + {ckpt_dir.name}/  ({', '.join(sorted(included))})")
            else:
                print(f"  ! {ckpt_dir.name}/  WARNING: no adapter files found")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nArchive created: {output_path}  ({size_mb:.1f} MB)")


def push_to_hub(checkpoints: list[Path], config_path: Path, repo_id: str, token: str):
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    print(f"\nCreating private HuggingFace repo: {repo_id}")
    api.create_repo(repo_id=repo_id, private=True, exist_ok=True, token=token)

    if config_path.exists():
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo='config.yaml',
            repo_id=repo_id,
            token=token,
        )
        print(f"  + config.yaml")

    for ckpt_dir in checkpoints:
        for f in ckpt_dir.iterdir():
            if f.name in ADAPTER_FILES:
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=f'{ckpt_dir.name}/{f.name}',
                    repo_id=repo_id,
                    token=token,
                )
        print(f"  + {ckpt_dir.name}/")

    print(f"\nUploaded to: https://huggingface.co/{repo_id}")
    print("Repo is private — only visible to you.")


def print_transfer_instructions(output_path: Path):
    print("\n" + "=" * 60)
    print("Transfer instructions")
    print("=" * 60)
    print("\nOption 1 — runpodctl (easiest):")
    print(f"  [RunPod]  runpodctl send {output_path.name}")
    print(f"  [Local]   runpodctl receive <code>")
    print("\nOption 2 — SCP:")
    print(f"  scp -i <key> root@<pod-ip>:/workspace/Perseus/{output_path.name} .")
    print("\nOnce transferred, extract on your local machine:")
    print(f"  tar -xzf {output_path.name} -C /home/perseus/Dev/Perseus/data/")
    print("\nThen run inference with:")
    print("  python3 eval.py --adapter data/checkpoints/<checkpoint-name>")
    print("=" * 60)


def main():
    cfg = load_config()
    checkpoints_dir = Path(cfg.get('checkpoints', {}).get('dir', 'data/checkpoints'))

    parser = argparse.ArgumentParser(description='Export LoRA adapters for transfer')
    parser.add_argument('--checkpoints-dir', type=Path, default=checkpoints_dir,
                        help=f'Directory containing checkpoints (default: {checkpoints_dir})')
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='Path to a specific checkpoint directory to export')
    parser.add_argument('--latest', action='store_true',
                        help='Export only the latest checkpoint')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output archive path (default: perseus_adapters_<timestamp>.tar.gz)')
    parser.add_argument('--push-hub', action='store_true',
                        help='Upload checkpoints to a private HuggingFace repo')
    parser.add_argument('--hub-repo', type=str, default=None,
                        help='HuggingFace repo id, e.g. username/perseus-1.5b-1k (default: auto-generated)')
    parser.add_argument('--hub-token', type=str, default=None,
                        help='HuggingFace token (default: HUGGINGFACE_TOKEN env var)')
    args = parser.parse_args()

    config_path = Path(__file__).parent / 'config.yaml'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = args.output or Path(f'perseus_adapters_{timestamp}.tar.gz')

    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = find_checkpoints(args.checkpoints_dir)
        if not checkpoints:
            print(f"ERROR: No checkpoints found in {args.checkpoints_dir}")
            sys.exit(1)

        if args.latest:
            checkpoints = [checkpoints[-1]]

    print(f"Found {len(checkpoints)} checkpoint(s):")
    for c in checkpoints:
        print(f"  {c}")

    package_checkpoints(checkpoints, config_path, output_path)
    print_transfer_instructions(output_path)

    if args.push_hub:
        token = args.hub_token or os.environ.get('HUGGINGFACE_TOKEN')
        if not token:
            print("\nERROR: --push-hub requires a token. Pass --hub-token or set HUGGINGFACE_TOKEN.")
            sys.exit(1)
        model_slug = cfg.get('model', {}).get('name', 'model').split('/')[-1].lower()
        repo_id = args.hub_repo or f"aromans/perseus-{model_slug}-{timestamp}"
        push_to_hub(checkpoints, config_path, repo_id, token)


if __name__ == '__main__':
    main()
