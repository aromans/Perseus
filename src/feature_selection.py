#!/usr/bin/env python3

import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class BasicBlock:
    address: int
    instructions: List[str]
    num_instructions: int
    has_call: bool
    has_jump: bool
    has_conditional: bool
    arithmetic_ops: int
    memory_ops: int
    control_ops: int

@dataclass
class CFGFeatures:
    nodes: List[Dict]
    edges: List[Tuple[int, int, str]]
    num_nodes: int
    num_edges: int
    cyclo_complexity: int
    max_depth: int 
    num_loops: int

@dataclass
class InstructionFeatures:
    instruction_seq: List[str]
    opcode_seq: List[str]
    normalized_seq: List[str]
    instruction_types: Dict[str, int]
    seq_length: int

class FeatureExtractor:

    ARITHMETIC_OPS = {'add', 'sub', 'mul', 'imul', 'div', 'idiv', 'inc', 'dec', 'neg', 'and', 'or', 'xor', 'not', 'shl', 'shr', 'sal', 'sar', 'rol', 'ror'}
    MEMORY_OPS = {'mov', 'lea', 'push', 'pop', 'movzx', 'movsx', 'xchg'}
    CONTROL_OPS = {'jmp', 'je', 'jne', 'jg', 'jge', 'jl', 'jle', 'ja', 'jae', 'jb', 'jbe', 'call', 'ret', 'jz', 'jnz', 'loop'}
    COMPARISON_OPS = {'cmp', 'test'} 

    def __init__(self):
        pass

    def parse_disassembly_file(self, disasm_file) -> List[Tuple[int, str, str]]:
        instructions = []
        try:
            with open(disasm_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split('\t')
                    if len(parts) >= 2:
                        addr_str = parts[0].rstrip(':')
                        try:
                            address = int(addr_str, 16)
                        except ValueError:
                            continue

                        mnemonic = parts[1].strip()
                        operands = parts[2].strip() if len(parts) > 2 else ""

                        instructions.append((address, mnemonic, operands))
        except Exception as e:
            logger.error(f"Error parsing disassembly file {disasm_file}: {e}")
            return []

        return instructions

    def build_cfg_from_instructions(self, instructions) -> nx.DiGraph:
        cfg = nx.DiGraph()

        if not instructions:
            return cfg

        # First pass will identify the boundaries
        block_starts = {instructions[0][0]}

        for i, (addr, mnem, ops) in enumerate(instructions[:-1]):
            next_addr = instructions[i+1][0]

            # Block ends at control flow instructions
            if mnem in self.CONTROL_OPS:
                block_starts.add(next_addr)

                # If its a jump, the target also starts a block
                if mnem.startswith('j') and ops:
                    try:
                        target = int(ops.split()[0], 16)
                        block_starts.add(target)
                    except (ValueError, IndexError):
                        pass

        # necessary??
        block_starts = sorted(block_starts)

        # Second pass will create and fill the basic blocks
        current_block = []
        current_block_start = instructions[0][0]

        for addr, mnem, ops in instructions:
            current_block.append((addr, mnem, ops))

            # Does next instruction start a new block? 
            if addr in block_starts and addr != current_block_start:
                if current_block:
                    self._add_basic_block_to_cfg(cfg, current_block_start, current_block)

                # New block
                current_block = [(addr, mnem, ops)]
                current_block_start = addr
            
            elif mnem in self.CONTROL_OPS:
                # A new control flow ends the block
                if current_block:
                    self._add_basic_block_to_cfg(cfg, current_block_start, current_block)

                current_block = []

        if current_block:
            self._add_basic_block_to_cfg(cfg, current_block_start, current_block)

        # Third pass adds edges based on the control flow
        self._add_cfg_edges(cfg, instructions)

        return cfg

    def _add_basic_block_to_cfg(self, cfg, block_start, instructions) -> None:
        num_instr = len(instructions)
        has_call = any(m == 'call' for _, m, _ in instructions)
        has_jump = any(m.startswith('j') for _, m, _ in instructions)
        has_conditional = any(m in {'je', 'jne', 'jg', 'jge', 'jl', 'jle', 'ja', 'jae', 'jb', 'jbe', 'jz', 'jnz'} for _, m, _ in instructions)

        arithmetic_ops = sum(1 for _, m, _ in instructions if m in self.ARITHMETIC_OPS)
        memory_ops = sum(1 for _, m, _ in instructions if m in self.MEMORY_OPS)
        control_ops = sum(1 for _, m, _ in instructions if m in self.CONTROL_OPS)

        cfg.add_node(
                block_start,
                instructions=[f"{m} {o}" for _, m, o in instructions],
                num_instructions=num_instr,
                has_call=has_call,
                has_jump=has_jump,
                has_conditional=has_conditional,
                arithmetic_ops=arithmetic_ops,
                memory_ops=memory_ops,
                control_ops=control_ops
        )

    def _add_cfg_edges(self, cfg, instructions) -> None:

        addr_to_idx = {addr: i for i, (addr, _, _) in enumerate(instructions)}

        for i, (addr, mnem, ops) in enumerate(instructions):
            if addr not in cfg.nodes:
                continue

            if i + 1 < len(instructions):
                next_addr = instructions[i + 1][0] 
                if next_addr in cfg.nodes and mnem not in {'jmp', 'ret'}:
                    cfg.add_edge(addr, next_addr, edge_type='fallthrough')

            if mnem in self.CONTROL_OPS and ops:
                try:
                    target_str = ops.split()[0]
                    if target_str.startswith('0x'):
                        target = int(target_str, 16)
                        if target in cfg.nodes:
                            edge_type = 'call' if mnem == 'call' else 'conditional' if mnem != 'jmp' else 'unconditional'
                            cfg.add_edge(addr, target, edge_type=edge_type)
                except (ValueError, IndexError):
                    pass

    def extract_cfg_features(self, cfg) -> CFGFeatures:
        if len(cfg.nodes) == 0:
            return CFGFeatures(
                    nodes=[],
                    edges=[],
                    num_nodes=0,
                    num_edges=0,
                    cyclo_complexity=0,
                    max_depth=0,
                    num_loops=0
            )

        nodes = []
        addr_to_idx = {addr: i for i, addr in enumerate(sorted(cfg.nodes))}

        for addr in sorted(cfg.nodes):
            node_data = cfg.nodes[addr]
            nodes.append({
                'index': addr_to_idx[addr],
                'num_instructions': node_data.get('num_instructions', 0),
                'has_call': node_data.get('has_call', False),
                'has_jump': node_data.get('has_jump', False),
                'has_conditional': node_data.get('has_conditional', False),
                'arithmetic_ops': node_data.get('arithmetic_ops', 0),
                'memory_ops': node_data.get('memory_ops', 0),
                'control_ops': node_data.get('control_ops', 0),
            })


        edges = []
        for src, dst, data in cfg.edges(data=True):
            edges.append((
                addr_to_idx[src],
                addr_to_idx[dst],
                data.get('edge_type', 'unknown')
            ))

        num_nodes = len(cfg.nodes)
        num_edges = len(cfg.edges)

        # https://en.wikipedia.org/wiki/Cyclomatic_complexity
        cyclo_complexity = max(0, num_edges - num_nodes + 2)

        try:
            entry_node = min(cfg.nodes)
            lengths = nx.single_source_shortest_path_length(cfg, entry_node)
            max_depth = max(length.values()) if lengths else 0
        except:
            max_depth = 0 

        try: 
            num_loops = sum(1 for _ in nx.simple_cycles(cfg))
        except:
            num_loops = 0

        return CFGFeatures(
                nodes=nodes,
                edges=edges,
                num_nodes=num_nodes,
                num_edges=num_edges,
                cyclo_complexity=cyclo_complexity,
                max_depth=max_depth,
                num_loops=num_loops
        )

    def extract_instruction_features(self, instructions) -> InstructionFeatures:

        if not instructions:
            return InstructionFeatures(
                    instruction_seq=[],
                    opcode_seq=[],
                    normalized_seq=[],
                    instruction_types={},
                    seq_length=0
            )

        instruction_seq = [f"{m} {o}".strip() for _, m, o in instructions]

        opcode_seq = [m for _, m, _ in instructions]

        normalized_seq = []
        for _, m, o in instructions:
            normalized_ops = o
            for reg in ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
                       'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp',
                       'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']:
                normalized_ops = normalized_ops.replace(reg, 'REG')

            import re
            normalized_ops = re.sub(r'0x[0-9a-fA-F]+', 'IMM', normalized_ops)
            normalized_ops = re.sub(r'\b\d+\b', 'IMM', normalized_ops)

            normalized_seq.append(f"{m} {normalized_ops}".strip())

        instruction_types = {}
        for _, m, _ in instructions:
            if m in self.ARITHMETIC_OPS:
                category = 'arithmetic'
            elif m in self.MEMORY_OPS:
                category = 'memory'
            elif m in self.CONTROL_OPS:
                category = 'control'
            elif m in self.COMPARISON_OPS:
                category = 'comparison'
            else:
                category = 'other'

            instruction_types[category] = instruction_types.get(category, 0) + 1

        return InstructionFeatures(
                instruction_seq=instruction_seq,
                opcode_seq=opcode_seq,
                normalized_seq=normalized_seq,
                instruction_types=instruction_types,
                seq_length=len(instructions)
        )

    def process_disassembly(self, disasm_dir, output_dir) -> bool:

        output_dir.mkdir(parents=True, exist_ok=True)

        disasm_files = list(disasm_dir.glob('*_disasm.txt'))
        if not disasm_files:
            logging.warning(f"No disassembly files found in {disasm_dir}")
            return False

        disasm_file = disasm_files[0]

        instructions = self.parse_disassembly_file(disasm_file)

        if not instructions:
            logger.error(f"Failed to parse instructions from {disasm_file}")
            return False

        cfg = self.build_cfg_from_instructions(instructions)

        cfg_features = self.extract_cfg_features(cfg)
        cfg_output = output_dir / 'cfg_features.json'
        with open(cfg_output, 'w') as f:
            json.dump(asdict(cfg_features), f, indent=2)

        cfg_graphml = output_dir / 'cfg.graphml'
        cfg_for_export = cfg.copy()
        for node in cfg_for_export.nodes:
            node_data = cfg_for_export.nodes[node]
            if 'instructions' in node_data:
                node_data['instructions'] = json.dumps(node_data['instructions'])
        nx.write_graphml(cfg_for_export, str(cfg_graphml))

        instr_features = self.extract_instruction_features(instructions)
        instr_output = output_dir / 'instruction_features.json'
        with open(instr_output, 'w') as f:
            json.dump(asdict(instr_features), f, indent=2)

        logger.info(f"Extracted features to {output_dir}")
        return True

def main():
    extractor = FeatureExtractor()

    disasm_dir = Path('./data/disassembled/mba/sample1')
    output_dir = Path('./data/features/mba/sample1')

    extractor.process_disassembly(disasm_dir, output_dir)

if __name__ == '__main__':
    main()



