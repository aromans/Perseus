#!/usr/bin/env python3
"""Pretty-print eval results: generated vs expected assembly, side by side."""

import json
import argparse
from pathlib import Path


def print_diff(generated: str, expected: str):
    gen_lines = generated.strip().splitlines()
    exp_lines = expected.strip().splitlines()
    max_len   = max(len(gen_lines), len(exp_lines), 1)

    col = 52
    print(f"  {'GENERATED':<{col}}  EXPECTED")
    print(f"  {'-'*col}  {'-'*col}")

    for i in range(max_len):
        g = gen_lines[i] if i < len(gen_lines) else "<missing>"
        e = exp_lines[i] if i < len(exp_lines) else "<missing>"
        match = "  " if g.strip() == e.strip() else "!!"
        print(f"{match} {g:<{col}}  {e}")


def main():
    parser = argparse.ArgumentParser(description="Pretty-print Perseus eval results")
    parser.add_argument('--results', type=Path,
                        default=Path('data/training/eval_results.json'),
                        help='Path to eval_results.json')
    parser.add_argument('--sample', type=str, default=None,
                        help='Filter by sample name (e.g. string_rev)')
    parser.add_argument('--obf', type=str, default=None,
                        help='Filter by obfuscation type (e.g. mba)')
    parser.add_argument('--function', type=str, default=None,
                        help='Filter by function name (e.g. main)')
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    if args.sample:
        results = [r for r in results if r['sample'] == args.sample]
    if args.obf:
        results = [r for r in results if r['obfuscation_type'] == args.obf]
    if args.function:
        results = [r for r in results if r['function'] == args.function]

    if not results:
        print("No results match the given filters.")
        return

    for r in results:
        em   = "EXACT MATCH" if r['exact_match'] else f"P: {r['line_precision']:.1%}  R: {r['line_recall']:.1%}  F1: {r['line_f1']:.1%}"
        print(f"\n{'='*110}")
        print(f"  {r['sample']} / {r['obfuscation_type']} / {r['function']}  —  {em}")
        print(f"{'='*110}")
        print_diff(r['generated'], r['expected'])

    print(f"\n{'='*110}")
    print(f"SUMMARY: {len(results)} result(s) shown")
    n_em  = sum(r['exact_match']    for r in results)
    avg_p = sum(r['line_precision'] for r in results) / len(results)
    avg_r = sum(r['line_recall']    for r in results) / len(results)
    avg_f = sum(r['line_f1']        for r in results) / len(results)
    print(f"  Exact match:  {n_em}/{len(results)}")
    print(f"  Avg P: {avg_p:.1%}  R: {avg_r:.1%}  F1: {avg_f:.1%}")


if __name__ == '__main__':
    main()
