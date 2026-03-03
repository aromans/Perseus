#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent))

from collect_sources import SourceCollector
from data_pipeline import DataPipeline
from feature_extraction import FeatureExtractor

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
        skip_collection
):
    logger.info("="*80)
    logger.info("Perseus Data Pipeline - Starting")
    logger.info("="*80)

    if not skip_collection:
        logger.info("\n[Step 1/5] Collecting source file...")
        source_dir = data_root / 'source'
        collector = SourceCollector(source_dir)
        samples = collector.collect_all(repos_dir)
        logger.info(f"Collected {len(samples)} total samples")
        logger.info(f"  - Benign: {sum(1 for _, is_mal in samples if not is_mal)}")
        logger.info(f"  - Malicious: {sum(1 for _, is_mal in samples if is_mal)}")
    else:
        logger.info("\n[Step 1/5] Collecting source file...")
        source_dir = data_root / 'source'
        benign_dir = data_root / 'benign'
        malicious_dir = data_root / 'malicious'

        samples = []
        for c_file in benign_dir.glob('*.c'):
            samples.append((c_file, False))
        for c_file in malicious_dir.glob('*.c'):
            samples.append((c_file, True))

        logger.info(f"Found {len(samples)} existing samples")

    if max_samples and len(samples) > max_samples:
        logger.info(f"Limiting to {max_samples} samples")
        samples = samples[:max_samples]

    logger.info("\n[Step 2-4/5] Processing samples...")
    pipeline = DataPipeline(data_root)

    pipeline.process_dataset(samples, obfuscation_types)

    logger.info("\n[Step 5/5] Extracting features...")
    extractor = FeatureExtractor()

    feature_count = 0
    for obf_type in ['clean'] + obfuscation_types:
        disasm_base = data_root / 'disassembled' / obf_type
        feature_base = data_root / 'features' / obf_type

        if not disasm_base.exists():
            continue

        for disasm_dir in disasm_base.iterdir():
            if disasm_dir.is_dir():
                feature_dir = feature_base / disasm_dir.name
                if extractor.process_disassembly(disasm_dir, feature_dir):
                    feature_count += 1

    logger.info(f"Extracted features for {feature_count} samples")

    logger.info("\n" + "="*80)
    logger.info("Pipeline Complete!")
    logger.info("="*80)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Check {data_root}/metadata/ for sample metadata")
    logger.info(f"Check {data_root}/features/ for extracted features")

def main():
    parser = argparse.ArgumentParser(description='Perseus Data Pipeline')

    parser.add_argument('--data-root', type=Path, default=Path('./perseus_data'), help='Root directory for all data (default: ./perseus_data)')
    parser.add_argument('--repos-dir', type=Path, default=Path('./repos'), help='Directory for cloned reposirtories (default: ./repos)')
    parser.add_argument('--obfuscations', nargs='+', default=['mba', 'virtualization', 'control_flow'], choices=['mba', 'virtualization', 'control_flow'], help='Obfuscation types to apply (default: all)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to process (default: all)')
    parser.add_argument('--skip-collection', action='store_true', help='Skip source collection step (use existing sources)')
    parser.add_argument('--collect-only', acion='store_true', help='Only run collection step, skip processing')

    args = parser.parse_args()

    if args.collect_only:
        logger.info("Running collection only...")
        source_dir = args.data_root / 'source'
        collector = SourceCollector(source_dir)
        samples = collector.collect_all(args.repos_dir)
        logger.info(f"Collected {len(samples)} samples")
        logger.info(f"Sources saved to {source_dir}")
    else:
        run_full_pipeline(
                data_root=args.data_root,
                repos_dir=args.repos_dir,
                obfuscation_types=args.obfuscations,
                max_samples=args.max_samples,
                skip_collection=args.skip_collection
        )

if __name__ == '__main__':
    main()

