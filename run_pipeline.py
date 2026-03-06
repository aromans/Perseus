#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from collect_sources import SourceCollector
from process_data import DataPipeline
from feature_selection import FeatureExtractor
from config import OBF_TYPES, load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('perseus_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_full_pipeline(
        data_root,
        repos_dir,
        obfuscation_types,
        max_samples,
        skip_collection,
        prepare_training=False,
        angha_dir=None,
        angha_samples=100,
        angha_min_lines=10,
        angha_seed=42,
):
    logger.info("="*80)
    logger.info("Perseus Data Pipeline - Starting")
    logger.info("="*80)

    logger.info("\nCollecting source file...")
    if not skip_collection:
        source_dir = data_root / 'source'
        collector = SourceCollector(source_dir)
        samples = collector.collect_all(
            repos_dir,
            angha_dir=angha_dir,
            angha_samples=angha_samples,
            angha_min_lines=angha_min_lines,
            angha_seed=angha_seed,
        )
        logger.info(f"Collected {len(samples)} total samples")
        logger.info(f"  - Benign: {sum(1 for _, is_mal in samples if not is_mal)}")
        logger.info(f"  - Malicious: {sum(1 for _, is_mal in samples if is_mal)}")
    else:
        source_dir    = data_root / 'source'
        benign_dir    = source_dir / 'benign'
        malicious_dir = source_dir / 'malicious'

        samples = []
        for c_file in benign_dir.glob('*.c'):
            samples.append((c_file, False))
        for c_file in malicious_dir.glob('*.c'):
            samples.append((c_file, True))

        logger.info(f"Found {len(samples)} existing samples")

    if max_samples and len(samples) > max_samples:
        logger.info(f"Limiting to {max_samples} samples")
        samples = samples[:max_samples]

    logger.info("\nProcessing samples...")
    pipeline = DataPipeline(data_root)

    pipeline.process_dataset(samples, obfuscation_types)

    #logger.info("\nExtracting features...")
    #extractor = FeatureExtractor()

    #feature_count = 0
    #for obf_type in ['clean'] + obfuscation_types:
    #    disasm_base = data_root / 'disassembled' / obf_type
    #    feature_base = data_root / 'features' / obf_type

    #    if not disasm_base.exists():
    #        continue

    #    for disasm_dir in disasm_base.iterdir():
    #        if disasm_dir.is_dir():
    #            feature_dir = feature_base / disasm_dir.name
    #            if extractor.process_disassembly(disasm_dir, feature_dir):
    #                feature_count += 1

    #logger.info(f"Extracted features for {feature_count} samples")

    if prepare_training:
        logger.info("\nPreparing training data...")
        from prepare_training_data import TrainingDataPreparer
        preparer = TrainingDataPreparer(data_root=data_root)
        train_data, val_data, test_data = preparer.prepare_all()
        training_dir = data_root / 'training'
        preparer.save_jsonl(train_data, training_dir / 'train.jsonl')
        preparer.save_jsonl(val_data, training_dir / 'val.jsonl')
        preparer.save_jsonl(test_data, training_dir / 'test.jsonl')
        preparer.save_stats(train_data, val_data, test_data, training_dir / 'data_stats.json')

    logger.info("\n" + "="*80)
    logger.info("Pipeline Complete!")
    logger.info("="*80)
    logger.info(f"Data root: {data_root}")

def main():
    cfg   = load_config()
    angha = cfg.get('anghabench', {})

    parser = argparse.ArgumentParser(description='Perseus Data Pipeline')

    parser.add_argument('--data-root', type=Path, default=Path('./data'), help='Root directory for all data (default: ./data)')
    parser.add_argument('--repos-dir', type=Path, default=Path('./repos'), help='Directory for cloned repositories (default: ./repos)')
    parser.add_argument('--obfuscations', nargs='+', default=OBF_TYPES, choices=OBF_TYPES, help='Obfuscation types to apply (default: all from config.yaml)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to process (default: all)')
    parser.add_argument('--skip-collection', action='store_true', help='Skip source collection step (use existing sources)')
    parser.add_argument('--collect-only', action='store_true', help='Only run collection step, skip processing')
    parser.add_argument('--prepare-training', action='store_true', help='Prepare training data after processing')
    parser.add_argument('--angha-dir', type=Path,
                        default=Path(angha.get('repo_dir', 'repos/AnghaBench')) if angha.get('enabled') else None,
                        help='Path to AnghaBench clone — overrides config.yaml (auto-cloned if absent)')
    parser.add_argument('--angha-samples', type=int, default=angha.get('n_samples', 100),
                        help='Number of AnghaBench functions to sample (default: from config.yaml)')
    parser.add_argument('--angha-min-lines', type=int, default=angha.get('min_lines', 10),
                        help='Minimum lines of code for AnghaBench filter (default: from config.yaml)')
    parser.add_argument('--angha-seed', type=int, default=angha.get('seed', 42),
                        help='Random seed for reproducible AnghaBench sampling (default: from config.yaml)')

    args = parser.parse_args()

    if args.collect_only:
        logger.info("Running collection only...")
        source_dir = args.data_root / 'source'
        collector = SourceCollector(source_dir)
        samples = collector.collect_all(
            args.repos_dir,
            angha_dir=args.angha_dir,
            angha_samples=args.angha_samples,
            angha_min_lines=args.angha_min_lines,
            angha_seed=args.angha_seed,
        )
        logger.info(f"Collected {len(samples)} samples")
        logger.info(f"Sources saved to {source_dir}")
    else:
        run_full_pipeline(
                data_root=args.data_root,
                repos_dir=args.repos_dir,
                obfuscation_types=args.obfuscations,
                max_samples=args.max_samples,
                skip_collection=args.skip_collection,
                prepare_training=args.prepare_training,
                angha_dir=args.angha_dir,
                angha_samples=args.angha_samples,
                angha_min_lines=args.angha_min_lines,
                angha_seed=args.angha_seed,
        )

if __name__ == '__main__':
    main()

