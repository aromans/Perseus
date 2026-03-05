#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = Path('./perseus_data/features')

SAMPLES = ['fibonacci', 'simple_arithmetic', 'string_rev']
OBF_TYPES = ['clean', 'mba', 'control_flow', 'virtualization']
OBF_LABELS = {
    'clean': 'Clean',
    'mba': 'MBA',
    'control_flow': 'Control Flow',
    'virtualization': 'Virtualization'
}


def load_features(sample, obf_type):
    variant = f"{sample}_{obf_type}"
    cfg_path = DATA_ROOT / obf_type / variant / 'cfg_features.json'
    instr_path = DATA_ROOT / obf_type / variant / 'instruction_features.json'

    cfg = {}
    instr = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
    if instr_path.exists():
        with open(instr_path) as f:
            instr = json.load(f)

    return cfg, instr


def collect_stats():
    stats = {}
    for sample in SAMPLES:
        stats[sample] = {}
        for obf_type in OBF_TYPES:
            cfg, instr = load_features(sample, obf_type)
            if not cfg or not instr:
                continue

            types = instr.get('instruction_types', {})
            stats[sample][obf_type] = {
                'Total Instructions': instr.get('seq_length', 0),
                'Arithmetic Ops': types.get('arithmetic', 0),
                'Memory Ops': types.get('memory', 0),
                'Control Ops': types.get('control', 0),
                'Comparisons': types.get('comparison', 0),
                'CFG Nodes': cfg.get('num_nodes', 0),
                'CFG Edges': cfg.get('num_edges', 0),
                'Cyclomatic Complexity': cfg.get('cyclo_complexity', 0),
                'Loops': cfg.get('num_loops', 0),
            }
    return stats


def print_table(stats):
    for sample in SAMPLES:
        if sample not in stats or not stats[sample]:
            continue
        print(f"\n{'=' * 70}")
        print(f"  {sample}")
        print(f"{'=' * 70}")

        metrics = list(next(iter(stats[sample].values())).keys())
        header = f"{'Metric':<25}" + "".join(f"{OBF_LABELS[o]:>15}" for o in OBF_TYPES if o in stats[sample])
        print(header)
        print("-" * len(header))

        for metric in metrics:
            row = f"{metric:<25}"
            for obf_type in OBF_TYPES:
                if obf_type not in stats[sample]:
                    continue
                val = stats[sample][obf_type][metric]
                row += f"{val:>15}"
            print(row)


def plot_comparison(stats, output_dir=Path('./perseus_data/images')):
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Per-sample grouped bar charts --
    metrics = ['Total Instructions', 'Arithmetic Ops', 'Memory Ops', 'Control Ops', 'Comparisons']

    for sample in SAMPLES:
        if sample not in stats or not stats[sample]:
            continue

        present = [o for o in OBF_TYPES if o in stats[sample]]
        x = np.arange(len(metrics))
        width = 0.8 / len(present)

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, obf_type in enumerate(present):
            values = [stats[sample][obf_type][m] for m in metrics]
            offset = (i - len(present) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=OBF_LABELS[obf_type])
            ax.bar_label(bars, fontsize=7, padding=2)

        ax.set_xlabel('Metric')
        ax.set_ylabel('Count')
        ax.set_title(f'Instruction Feature Comparison: {sample}')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()

        path = output_dir / f'{sample}_instruction_comparison.png'
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path}")

    # -- CFG comparison across all samples --
    cfg_metrics = ['CFG Nodes', 'CFG Edges', 'Cyclomatic Complexity']
    fig, axes = plt.subplots(1, len(SAMPLES), figsize=(6 * len(SAMPLES), 6), sharey=False)
    if len(SAMPLES) == 1:
        axes = [axes]

    for ax, sample in zip(axes, SAMPLES):
        if sample not in stats or not stats[sample]:
            continue

        present = [o for o in OBF_TYPES if o in stats[sample]]
        x = np.arange(len(cfg_metrics))
        width = 0.8 / len(present)

        for i, obf_type in enumerate(present):
            values = [stats[sample][obf_type][m] for m in cfg_metrics]
            offset = (i - len(present) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=OBF_LABELS[obf_type])
            ax.bar_label(bars, fontsize=7, padding=2)

        ax.set_title(sample)
        ax.set_xticks(x)
        ax.set_xticklabels(cfg_metrics, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle('CFG Feature Comparison by Obfuscation Type', fontsize=14)
    fig.tight_layout()

    path = output_dir / 'cfg_comparison.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")

    # -- Normalized overhead heatmap (% increase over clean) --
    all_metrics = ['Total Instructions', 'Arithmetic Ops', 'Memory Ops',
                   'Control Ops', 'Comparisons', 'CFG Nodes', 'CFG Edges']
    obf_only = [o for o in OBF_TYPES if o != 'clean']
    rows = []
    row_labels = []

    for sample in SAMPLES:
        if sample not in stats or 'clean' not in stats[sample]:
            continue
        for obf_type in obf_only:
            if obf_type not in stats[sample]:
                continue
            row = []
            for m in all_metrics:
                clean_val = stats[sample]['clean'][m]
                obf_val = stats[sample][obf_type][m]
                if clean_val > 0:
                    row.append(((obf_val - clean_val) / clean_val) * 100)
                else:
                    row.append(0)
            rows.append(row)
            row_labels.append(f"{sample}\n({OBF_LABELS[obf_type]})")

    if rows:
        fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.8)))
        data = np.array(rows)
        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')

        ax.set_xticks(np.arange(len(all_metrics)))
        ax.set_xticklabels(all_metrics, rotation=30, ha='right')
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)

        for i in range(len(row_labels)):
            for j in range(len(all_metrics)):
                val = data[i, j]
                color = 'white' if abs(val) > 60 else 'black'
                ax.text(j, i, f"{val:+.0f}%", ha='center', va='center',
                        fontsize=8, color=color)

        fig.colorbar(im, ax=ax, label='% Change from Clean')
        ax.set_title('Obfuscation Overhead Relative to Clean Baseline')
        fig.tight_layout()

        path = output_dir / 'obfuscation_overhead_heatmap.png'
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path}")


def main():
    stats = collect_stats()
    print_table(stats)
    plot_comparison(stats)


if __name__ == '__main__':
    main()
