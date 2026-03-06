#!/usr/bin/env python3

import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CRT_FUNCTIONS = {
    '_start', '_init', '_fini',
    'deregister_tm_clones', 'register_tm_clones',
    '__do_global_dtors_aux', 'frame_dummy',
    '__libc_csu_init', '__libc_csu_fini',
    '__libc_start_main', '__cxa_finalize',
    '__do_global_dtors_aux_fini_array_entry',
    '__frame_dummy_init_array_entry',
    '__gmon_start__', '_ITM_deregisterTMCloneTable',
    '_ITM_registerTMCloneTable',
}

OBF_TYPES = ['mba', 'control_flow', 'virtualization']


@dataclass
class TrainingPair:
    instruction: str
    input: str
    output: str
    metadata: Dict


class TrainingDataPreparer:

    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.compiled_dir = self.data_root / 'compiled'
        self.disassembled_dir = self.data_root / 'disassembled'
        self.training_dir = self.data_root / 'training'

    def extract_function_boundaries(self, binary_path: Path) -> Dict[str, Tuple[int, int]]:
        from elftools.elf.elffile import ELFFile

        functions = {}

        with open(binary_path, 'rb') as f:
            elf = ELFFile(f)
            symtab = elf.get_section_by_name('.symtab')
            if symtab is None:
                logger.warning(f"No symbol table in {binary_path}")
                return functions

            # Collect all functions in .text with their addresses
            all_funcs = []
            for sym in symtab.iter_symbols():
                if sym['st_info']['type'] == 'STT_FUNC' and sym['st_value'] != 0:
                    name = sym.name
                    addr = sym['st_value']
                    size = sym['st_size']
                    all_funcs.append((addr, name, size))

            all_funcs.sort(key=lambda x: x[0])

            # For functions with size=0, infer from next function's address
            for i, (addr, name, size) in enumerate(all_funcs):
                if name in CRT_FUNCTIONS:
                    continue
                # Skip external/undefined symbols
                if addr == 0:
                    continue

                if size > 0:
                    end_addr = addr + size
                else:
                    # Infer from next function
                    if i + 1 < len(all_funcs):
                        end_addr = all_funcs[i + 1][0]
                    else:
                        continue

                functions[name] = (addr, end_addr)

        return functions

    def build_address_to_symbol_map(self, binary_path) -> Dict[int, str]:
        from elftools.elf.elffile import ELFFile

        addr_map = {}

        with open(binary_path, 'rb') as f:
            elf = ELFFile(f)

            # Regular symbols
            symtab = elf.get_section_by_name('.symtab')
            if symtab:
                for sym in symtab.iter_symbols():
                    if sym['st_info']['type'] == 'STT_FUNC' and sym['st_value'] != 0:
                        addr_map[sym['st_value']] = sym.name

            # Dynamic symbols (for PLT resolution)
            dynsym = elf.get_section_by_name('.dynsym')
            if dynsym:
                for sym in dynsym.iter_symbols():
                    if sym['st_info']['type'] == 'STT_FUNC' and sym.name:
                        # PLT entries: scan .plt.sec section
                        pass

            # Parse PLT entries by reading .rela.plt
            plt_sec = elf.get_section_by_name('.plt.sec')
            rela_plt = elf.get_section_by_name('.rela.plt')
            if plt_sec and rela_plt and dynsym:
                plt_addr = plt_sec['sh_addr']
                plt_entry_size = 16 
                for i, rel in enumerate(rela_plt.iter_relocations()):
                    sym_idx = rel['r_info_sym']
                    sym = dynsym.get_symbol(sym_idx)
                    if sym and sym.name:
                        entry_addr = plt_addr + i * plt_entry_size
                        # Strip version info (e.g., printf@GLIBC_2.2.5 -> printf)
                        clean_name = sym.name.split('@')[0]
                        addr_map[entry_addr] = clean_name

        return addr_map

    def parse_disassembly(self, disasm_path) -> List[Tuple[int, str, str]]:
        instructions = []
        with open(disasm_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    addr_str = parts[0].rstrip(':')
                    try:
                        addr = int(addr_str, 16)
                    except ValueError:
                        continue
                    mnemonic = parts[1].strip()
                    operands = parts[2].strip() if len(parts) > 2 else ''
                    instructions.append((addr, mnemonic, operands))
        return instructions

    def extract_function_disassembly(
        self,
        instructions,
        func_boundaries,
        addr_to_symbol
    ) -> Dict[str, str]:
        functions = {}

        for func_name, (start_addr, end_addr) in func_boundaries.items():
            func_instrs = [
                (addr, mnem, ops) for addr, mnem, ops in instructions
                if start_addr <= addr < end_addr
            ]

            if not func_instrs:
                continue

            normalized_lines = []
            for addr, mnem, ops in func_instrs:
                offset = addr - start_addr
                norm_ops = self._normalize_operands(
                    ops, mnem, start_addr, end_addr, addr_to_symbol
                )
                if norm_ops:
                    normalized_lines.append(f"+0x{offset:x}:\t{mnem}\t{norm_ops}")
                else:
                    normalized_lines.append(f"+0x{offset:x}:\t{mnem}")

            functions[func_name] = '\n'.join(normalized_lines)

        return functions

    def _normalize_operands(
        self,
        operands,
        mnemonic,
        func_start,
        func_end,
        addr_to_symbol
    ) -> str:
        if not operands:
            return operands

        def replace_addr(match):
            addr = int(match.group(0), 16)

            if addr in addr_to_symbol:
                return addr_to_symbol[addr]

            if func_start <= addr < func_end:
                return f"+0x{addr - func_start:x}"
            
            return match.group(0)

        if mnemonic in ('call', 'jmp', 'je', 'jne', 'jg', 'jge', 'jl', 'jle',
                        'ja', 'jae', 'jb', 'jbe', 'jz', 'jnz', 'jns', 'js',
                        'jle', 'jno', 'jo', 'jp', 'jnp', 'loop', 'jcxz', 'jecxz'):
            return re.sub(r'0x[0-9a-fA-F]+', replace_addr, operands)

        return operands

    def create_training_pairs(
        self,
        sample_name,
        obf_type
    ) -> List[TrainingPair]:
        pairs = []

        clean_binary = self.compiled_dir / 'clean' / f'{sample_name}_clean.bin'
        obf_binary = self.compiled_dir / obf_type / f'{sample_name}_{obf_type}.bin'

        clean_disasm = (self.disassembled_dir / 'clean' / f'{sample_name}_clean'
                        / f'{sample_name}_clean_disasm.txt')
        obf_disasm = (self.disassembled_dir / obf_type / f'{sample_name}_{obf_type}'
                      / f'{sample_name}_{obf_type}_disasm.txt')

        for path in [clean_binary, obf_binary, clean_disasm, obf_disasm]:
            if not path.exists():
                logger.warning(f"Missing file: {path}")
                return pairs

        # Get function boundaries from both binaries
        clean_funcs = self.extract_function_boundaries(clean_binary)
        obf_funcs = self.extract_function_boundaries(obf_binary)

        # Build symbol maps for call resolution
        clean_symbols = self.build_address_to_symbol_map(clean_binary)
        obf_symbols = self.build_address_to_symbol_map(obf_binary)

        clean_instrs = self.parse_disassembly(clean_disasm)
        obf_instrs = self.parse_disassembly(obf_disasm)

        # Extract per-function disassembly
        clean_func_disasm = self.extract_function_disassembly(
            clean_instrs, clean_funcs, clean_symbols
        )
        obf_func_disasm = self.extract_function_disassembly(
            obf_instrs, obf_funcs, obf_symbols
        )

        # Match functions present in both clean and obfuscated
        common_funcs = set(clean_func_disasm.keys()) & set(obf_func_disasm.keys())

        for func_name in sorted(common_funcs):
            clean_text = clean_func_disasm[func_name]
            obf_text = obf_func_disasm[func_name]

            obf_label = obf_type.replace('_', ' ')
            pair = TrainingPair(
                instruction=(
                    f"Deobfuscate the following {obf_label}-obfuscated "
                    f"x86-64 assembly function '{func_name}'."
                ),
                input=obf_text,
                output=clean_text,
                metadata={
                    'sample': sample_name,
                    'obfuscation_type': obf_type,
                    'function': func_name,
                    'obf_instruction_count': obf_text.count('\n') + 1,
                    'clean_instruction_count': clean_text.count('\n') + 1,
                }
            )
            pairs.append(pair)

        return pairs

    def discover_samples(self) -> List[str]:
        clean_dir = self.disassembled_dir / 'clean'
        if not clean_dir.exists():
            return []

        samples = []
        for d in sorted(clean_dir.iterdir()):
            if d.is_dir() and d.name.endswith('_clean'):
                sample_name = d.name.removesuffix('_clean')
                samples.append(sample_name)
        return samples

    def prepare_all(
        self,
        train_ratio: float = 0.5,
        val_ratio: float = 0.25,
        test_ratio: float = 0.25,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:

        samples = self.discover_samples()
        if not samples:
            logger.error("No samples found")
            return [], [], []

        samples = sorted(samples)
        logger.info(f"Found samples: {samples}")

        n = len(samples)
        n_test = round(n * test_ratio)
        n_val = round(n * val_ratio)
        n_train = n - n_val - n_test

        train_samples = set(samples[:n_train])
        val_samples   = set(samples[n_train:n_train + n_val])
        test_samples  = set(samples[n_train + n_val:])

        logger.info(f"Train samples ({n_train}): {sorted(train_samples)}")
        logger.info(f"Val samples  ({n_val}):   {sorted(val_samples)}")
        logger.info(f"Test samples ({n_test}):  {sorted(test_samples)}")

        all_pairs = []
        for sample in samples:
            for obf_type in OBF_TYPES:
                pairs = self.create_training_pairs(sample, obf_type)
                logger.info(f"  {sample}/{obf_type}: {len(pairs)} function pairs")
                all_pairs.extend(pairs)

        logger.info(f"Total pairs: {len(all_pairs)}")

        train_data, val_data, test_data = [], [], []
        for pair in all_pairs:
            record = asdict(pair)
            sample = pair.metadata['sample']
            if sample in test_samples:
                test_data.append(record)
            elif sample in val_samples:
                val_data.append(record)
            else:
                train_data.append(record)

        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        return train_data, val_data, test_data

    def save_jsonl(self, records, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        logger.info(f"Saved {len(records)} records to {output_path}")

    def save_stats(self, train_data, val_data, test_data, output_path):
        def compute_stats(data):
            if not data:
                return {}
            input_lens = [len(r['input'].split('\n')) for r in data]
            output_lens = [len(r['output'].split('\n')) for r in data]
            obf_types = {}
            for r in data:
                ot = r['metadata']['obfuscation_type']
                obf_types[ot] = obf_types.get(ot, 0) + 1
            return {
                'count': len(data),
                'avg_input_lines': sum(input_lens) / len(input_lens),
                'max_input_lines': max(input_lens),
                'avg_output_lines': sum(output_lens) / len(output_lens),
                'max_output_lines': max(output_lens),
                'obfuscation_types': obf_types,
            }

        stats = {
            'train': compute_stats(train_data),
            'val': compute_stats(val_data),
            'test': compute_stats(test_data),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved stats to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for deobfuscation')
    parser.add_argument('--data-root', type=Path, default=Path('./data'),
                        help='Root data directory (default: ./data)')
    parser.add_argument('--train-ratio', type=float, default=0.5,
                        help='Fraction of samples for training (default: 0.5)')
    parser.add_argument('--val-ratio', type=float, default=0.25,
                        help='Fraction of samples for validation (default: 0.25)')
    parser.add_argument('--test-ratio', type=float, default=0.25,
                        help='Fraction of samples for test (default: 0.25)')
    args = parser.parse_args()

    preparer = TrainingDataPreparer(data_root=args.data_root)
    train_data, val_data, test_data = preparer.prepare_all(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    if train_data or val_data or test_data:
        training_dir = args.data_root / 'training'
        preparer.save_jsonl(train_data, training_dir / 'train.jsonl')
        preparer.save_jsonl(val_data, training_dir / 'val.jsonl')
        preparer.save_jsonl(test_data, training_dir / 'test.jsonl')
        preparer.save_stats(train_data, val_data, test_data, training_dir / 'data_stats.json')


if __name__ == '__main__':
    main()
