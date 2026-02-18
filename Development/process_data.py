#!/usr/bin/env python3

import os
import json
import subprocess
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import shutil

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SampleMetadata:
    sample_id: str
    original_path: str
    source_hash: str
    is_malicious: bool
    obfuscation_type: Optional[str]
    obfuscation_params: Optional[Dict]
    compilation_success: bool
    disassembly_success: bool
    feature_extraction_success: bool

class DataPipeline:
    OBFUSCATION_COMMANDS = {
        'mba': "tigress --Transform=EncodeArithmetic --EncodeArithmeticDumpFileName={dump_file} --Functions=* {input_file} --out={output_file}",
        'virtualization': "tigress --Transform=Virtualize --VirtualizeDispatch=direct --Functions=* {input_file} --out={output_file}",
        'control_flow': "tigress --Transform=Flatten --Functions=* {input_file} --out={output_file}",
        # Add more where needed
    }

    def __init__(
        self,
        data_root,
        tigress_path = "tigress",
        gcc_path = "gcc",
        ghidra_path = None
    ):
        self.data_root    = Path(data_root)
        self.tigress_path = tigress_path
        self.gcc_path     = gcc_path
        self.ghidra_path  = ghidra_path

        self.dirs = {
            'source':       self.data_root / 'source',
            'obfuscated':   self.data_root / 'obfuscated',
            'compiled':     self.data_root / 'compiled',
            'disassembled': self.data_root / 'disassembled',
            'features':     self.data_root / 'features',
            'metadata':     self.data_root / 'metadata'.
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        for obf_type in self.OBFUSCATION_COMMANDS:
            (self.dirs['obfuscated']   / obf_type).mkdir(exist_ok=True)
            (self.dirs['compiled']     / obf_type).mkdir(exist_ok=True)
            (self.dirs['disassembled'] / obf_type).mkdir(exist_ok=True)
            (self.dirs['features']     / obf_type).mkdir(exist_ok=True)

    def apply_obfuscation(
        self,
        source_file,
        obf_type,
        output_file,
        dump_file = None
    ) -> bool:
        if obf_type not in self.OBFUSCATION_COMMANDS:
            logger.error(f"Unknown obfuscation type: {obf_type}")
            return False

        cmd_template = self.OBFUSCATION_COMMANDS[obf_type]
        cmd = cmd_template.format(
                input_file=source_file,
                output_file=output_file,
                dump_file=dump_file or '/dev/null'
        )

        try:
            logger.info(f"Applying {obf_type} obfuscation to {source_file.name}")

            result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300
            )

            if result.returncode == 0 and output_file.exists():
                logger.info(f"Successfully obfuscated: {output_file}")
                return True
            else:
                logger.error(f"Obfuscated failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Obfuscation timeout for {source_file}")
            return False
        except Exception as e:
            logger.error(f"Obfuscation error: {e}")
            return False

def main():
    data_root = Path('./perseus_data')

    obfuscation_types = ['mba', 'virtualization', 'control_flow']

if __name__ == '__main__':
    main()
