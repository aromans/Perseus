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
            'metadata':     self.data_root / 'metadata' 
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        for obf_type in self.OBFUSCATION_COMMANDS:
            (self.dirs['obfuscated']   / obf_type).mkdir(exist_ok=True)
            (self.dirs['compiled']     / obf_type).mkdir(exist_ok=True)
            (self.dirs['disassembled'] / obf_type).mkdir(exist_ok=True)
            (self.dirs['features']     / obf_type).mkdir(exist_ok=True)

        (self.dirs['obfuscated']   / 'clean').mkdir(exist_ok=True)
        (self.dirs['compiled']     / 'clean').mkdir(exist_ok=True)
        (self.dirs['disassembled'] / 'clean').mkdir(exist_ok=True)
        (self.dirs['features']     / 'clean').mkdir(exist_ok=True)

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
        if dump_file:
            dump_path = Path(dump_file).resolve()
            dump_dir = str(dump_path.parent)
            dump_name = dump_path.name
        else:
            dump_dir = None
            dump_name = '/dev/null'

        cmd = cmd_template.format(
                input_file=Path(source_file).resolve(),
                output_file=Path(output_file).resolve(),
                dump_file=dump_name
        )

        try:
            logger.info(f"Applying {obf_type} obfuscation to {source_file.name}")

            result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=dump_dir
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

    def compile_to_binary(
        self,
        source_file,
        output_binary ,
        optimization_level = "O0"
    ) -> bool:
        try:
            cmd = [
                self.gcc_path,
                f'-{optimization_level}',
                '-o', str(output_binary),
                str(source_file),
                '-lm'
            ]
            
            logger.info(f"Compiling {source_file.name}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0 and output_binary.exists():
                logger.info(f"Successfully compiled: {output_binary}")
                return True
            else:
                logger.error(f"Compilation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Compilation timeout for {source_file}")
            return False
        except Exception as e:
            logger.error(f"Compilation error: {e}")
            return False

    def disassemble_binary(
            self,
            binary_path,
            output_dir
    ) -> bool:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            from capstone import Cs, CS_ARCH_X86, CS_MODE_64
            from elftools.elf.elffile import ELFFile

            with open(binary_path, 'rb') as f:
                elf = ELFFile(f)
                text_section = elf.get_section_by_name('.text')
                if text_section is None:
                    logger.error(f"No .text section found in {binary_path}")
                    return False

                code = text_section.data()
                text_addr = text_section['sh_addr']

            md = Cs(CS_ARCH_X86, CS_MODE_64)
            md.detail = True
            disasm_output = output_dir / f'{binary_path.stem}_disasm.txt'

            with open(disasm_output, 'w') as f:
                for i in md.disasm(code, text_addr):
                    f.write(f"0x{i.address:x}:\t{i.mnemonic}\t{i.op_str}\n")

            logger.info(f"Disassembled with Capstone: {disasm_output}")
            return True

        except ImportError as e:
            logger.error(f"Missing dependency: {e}. Install with: pip install capstone pyelftools")
            return False
        except Exception as e:
            logger.error(f"Capstone disassembly error: {e}")
            return False
    
    def extract_features(self, disasm_dir, feature_dir) -> bool:
        try:
            from feature_selection import FeatureExtractor
            extractor = FeatureExtractor()
            return extractor.process_disassembly(disasm_dir, feature_dir)
        except ImportError:
            logger.warning("FeatureExtractor not available, skipping feature extraction")
            feature_dir.mkdir(parents=True, exist_ok=True)
            return False
        except Exception as e:
            logger.error(f"Feature extraction failed for {disasm_dir}: {e}")
            return False

    def compute_hash(self, file) -> str:
        sha256_hash = hashlib.sha256()
        with open(file, 'rb') as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(block)
        return sha256_hash.hexdigest()

    def process_variant(self, file, sample_id, sample_hash, is_mal, obf_type) -> Optional[SampleMetadata]:
        variant_name = obf_type if obf_type else 'clean'
        variant_id   = f"{sample_id}_{variant_name}"

        obf_file = self.dirs['obfuscated'] / variant_name / f"{variant_id}.c"
        
        if obf_type:
            dump_file = self.dirs['metadata'] / f"{variant_id}_obf_dump.json"

            if not self.apply_obfuscation(file, obf_type, obf_file, dump_file):
                return None
        else:
            print(file)
            shutil.copy(file, obf_file)

        binary_file = self.dirs['compiled'] / variant_name / f"{variant_id}.bin"
        is_compiled = self.compile_to_binary(obf_file, binary_file)

        if not is_compiled:
            logger.warning(f"Compilation failed for {variant_id}")
            return SampleMetadata(
                    sample_id=variant_id,
                    original_path=str(file),
                    source_hash=sample_hash,
                    is_malicious=is_mal,
                    obfuscation_type=obf_type,
                    obfuscation_params=None, # TODO: Maybe add variable obfuscation params?
                    compilation_success=False,
                    disassembly_success=False,
                    feature_extraction_success=True
            )

        disasm_dir = self.dirs['disassembled'] / variant_name / variant_id
        is_disassembled = self.disassemble_binary(binary_file, disasm_dir)

        features_extracted = False
        if is_disassembled:
            features_dir = self.dirs['features'] / variant_name / variant_id
            features_extracted = self.extract_features(disasm_dir, features_dir)

        metadata = SampleMetadata(
                sample_id=variant_id,
                original_path=str(file),
                source_hash=sample_hash,
                is_malicious=is_mal,
                obfuscation_type=obf_type,
                obfuscation_params=None, # TODO: Maybe add variable obfuscation params?
                compilation_success=is_compiled,
                disassembly_success=is_disassembled,
                feature_extraction_success=features_extracted
        )


        metadata_file = self.dirs['metadata'] / f"{variant_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        return metadata

    def process_sample(self, file, is_mal, obfuscation_types) -> List[SampleMetadata]:
        metadata_list = []

        sample_id = file.stem
        sample_hash = self.compute_hash(file)

        clean_metadata = self.process_variant(file, sample_id, sample_hash, is_mal, obf_type=None)

        if clean_metadata:
            metadata_list.append(clean_metadata)

        if (obfuscation_types):
            for obf in obfuscation_types:
                obf_metadata = self.process_variant(file, sample_id, sample_hash, is_mal, obf_type=obf)
                if obf_metadata:
                    metadata_list.append(obf_metadata)

        return metadata_list

    def process_dataset(self, samples, obfuscation_types) -> None:
        metadata = []

        for file, is_mal in samples:
            logger.info(f"Processing {file.name}")
            metadata_list = self.process_sample(file, is_mal, obfuscation_types)
            metadata.extend(metadata_list)

        metadata_dir = self.dirs['metadata'] / 'dataset_metadata.json'
        with open(metadata_dir, 'w') as f:
            json.dump([asdict(m) for m in metadata], f, indent=2)

        logger.info(f"Processed {len(metadata)} samples")
        logger.info(f"Metadata saved to {metadata_dir}")

def main():
    data_root = Path('./perseus_data')

    obfuscation_types = ['mba', 'virtualization', 'control_flow']

    pipeline = DataPipeline(
        data_root=data_root,
        tigress_path='tigress',
        gcc_path='gcc'
    )

    source_files = [
            (Path('./perseus_data/source/benign/fibonacci.c'), False),
            #(Path('./perseus_data/source/benign/simple_arithmetic.c'), False),
            #(Path('./perseus_data/source/benign/string_rev.c'), False)
    ]

    pipeline.process_dataset(source_files, obfuscation_types)

if __name__ == '__main__':
    main()
