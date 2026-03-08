#!/usr/bin/env python3

import os
import re
import random
import tempfile
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SourceCollector:
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.benign_dir    = self.output_dir / 'benign'
        self.malicious_dir = self.output_dir / 'malicious'

        self.benign_dir.mkdir(exist_ok=True)
        self.malicious_dir.mkdir(exist_ok=True)

    def _passes_static_filter(self, path: Path, min_lines: int = 10) -> bool:
        """Quick content check: minimum non-trivial lines and at least one control flow construct."""
        try:
            content = path.read_text(errors='ignore')
            lines = [
                l for l in content.splitlines()
                if l.strip()
                and not l.strip().startswith('//')
                and not l.strip().startswith('*')
                and not l.strip().startswith('/*')
            ]
            if len(lines) < min_lines:
                return False
            return any(kw in content for kw in ('if ', 'for ', 'while ', 'switch ', 'do {', 'do\n'))
        except Exception:
            return False

    def _try_compile(self, path: Path) -> bool:
        """Test-compile a file (injecting a stub main if absent). Returns True if gcc succeeds."""
        try:
            content = path.read_text(errors='ignore')
            if 'int main(' not in content and 'void main(' not in content:
                content += '\nint main(void) { return 0; }\n'

            with tempfile.NamedTemporaryFile(suffix='.c', mode='w', delete=False) as f:
                f.write(content)
                tmp = Path(f.name)

            result = subprocess.run(
                ['gcc', '-O0', '-o', '/dev/null', str(tmp), '-lm', '-w'],
                capture_output=True,
                timeout=30,
            )
            tmp.unlink(missing_ok=True)
            return result.returncode == 0
        except Exception:
            return False

    def _sanitize_name(self, path: Path, seen: set) -> str:
        """Derive a short unique name from an AnghaBench filename.
        AnghaBench names follow <repo>_<file>_<funcname>.c — we take the last segment."""
        parts = path.stem.split('_')
        base = re.sub(r'[^a-zA-Z0-9]', '_', parts[-1])[:40].strip('_') or 'angha'
        name, counter = base, 1
        while name in seen:
            name = f"{base}_{counter}"
            counter += 1
        return name

    def collect_anghabench_samples(
        self,
        angha_dir: Path,
        n_samples: int = 100,
        min_lines: int = 10,
        seed: int = 42,
    ) -> List[Path]:
        """Sample n compilable, non-trivial functions from a local AnghaBench clone.

        Clone AnghaBench with:
            git clone https://github.com/brenocfg/AnghaBench
        For a partial clone of one subdirectory:
            git clone --depth 1 --filter=blob:none --sparse https://github.com/brenocfg/AnghaBench
            cd AnghaBench && git sparse-checkout set <subdir>
        """
        angha_dir = Path(angha_dir)
        if not angha_dir.exists():
            try:
                logger.info(f"Cloning AnghaBench into {angha_dir} (this may take a while)...")
                angha_dir.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([
                    'git', 'clone', '--depth', '1',
                    'https://github.com/brenocfg/AnghaBench.git',
                    str(angha_dir)
                ], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone AnghaBench: {e}")
                return []

        all_files = list(angha_dir.rglob('*.c'))
        logger.info(f"Found {len(all_files)} AnghaBench candidate files in {angha_dir}")

        random.seed(seed)
        random.shuffle(all_files)

        # Cap the candidate pool — no need to scan all 1M files when we only need n_samples.
        # 20x gives ample headroom even if most files fail static/compile checks.
        candidate_pool = all_files[:n_samples * 20]
        logger.info(f"Candidate pool: {len(candidate_pool)} files (capped at {n_samples} × 20)")

        # Phase 1: static filter (cheap content check, no subprocess)
        static_passed = []
        for p in tqdm(candidate_pool, desc="Static filter", unit="file"):
            if self._passes_static_filter(p, min_lines):
                static_passed.append(p)
        logger.info(f"Static filter: {len(static_passed)} / {len(candidate_pool)} passed")

        # Phase 2: parallel compile check — gcc is the bottleneck, parallelise it
        workers = min(8, os.cpu_count() or 4)
        logger.info(f"Compile-checking {len(static_passed)} candidates with {workers} parallel workers...")

        passed = []
        logging.disable(logging.INFO)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._try_compile, p): p for p in static_passed}
            with tqdm(total=n_samples, desc="Compile check", unit="file") as pbar:
                for future in as_completed(futures):
                    if future.result():
                        passed.append(futures[future])
                        pbar.update(1)
                    if len(passed) >= n_samples:
                        for f in futures:
                            f.cancel()
                        break
        logging.disable(logging.NOTSET)

        logger.info(f"Compile check: {len(passed)} / {len(static_passed)} passed")

        # Phase 3: write out up to n_samples, preserving shuffled order for reproducibility
        passed_set = set(passed)
        ordered = [p for p in static_passed if p in passed_set]

        collected = []
        seen_names = set()

        for path in ordered:
            if len(collected) >= n_samples:
                break

            name = self._sanitize_name(path, seen_names)
            seen_names.add(name)

            content = path.read_text(errors='ignore')
            if 'int main(' not in content and 'void main(' not in content:
                content += '\nint main(void) { return 0; }\n'

            out_path = self.benign_dir / f'{name}.c'
            out_path.write_text(content)
            collected.append(out_path)
            logger.info(f"  [{len(collected)}/{n_samples}] {name}")

        logger.info(f"Collected {len(collected)} AnghaBench samples -> {self.benign_dir}")
        return collected

    def collect_thezoo_samples(self, zoo_dir) -> List[Path]: 
        collected = []

        if not zoo_dir.exists():
            try:
                logger.info("Cloning theZoo repository...")
                subprocess.run([
                    'git', 'clone',
                    'https://github.com/ytisf/theZoo.git',
                    str(zoo_dir)], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone theZoo repo: {e}")
                return collected

        malware_sources = zoo_dir / 'malware' / 'Source'

        if malware_sources.exists():
            # TODO: Go through and unzip malware with password 'infected' for C source code
            pass

        logger.info(f"Collected {len(collected)} malicious C files from theZoo")
        return collected

    def collect_coreutils_samples(self, coreutils_dir) -> List[Path]:
        collected = []

        if not coreutils_dir.exists():
            try:
                logger.info("Cloning GNU coreutils...")
                subprocess.run([
                    'git', 'clone',
                    'https://github.com/coreutils/coreutils.git',
                    str(coreutils_dir)], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone coreutils: {e}")
                return collected

        src_dir = coreutils_dir / 'src'
        
        if src_dir.exists():
            #TODO: Go through and rglob *.c files and shutil.copy to collection folder
            pass

        logger.info(f"Collected {len(collected)} benign C files from coreutils")
        return collected

    def collect_busybox_samples(self, busybox_dir) -> List[Path]:
        collected = []

        if not busybox_dir.exists():
            try:
                logger.info("Cloning BusyBox...")
                subprocess.run([
                    'git', 'clone',
                    'https://github.com/mirror/busybox.git',
                    str(busybox_dir)], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone busybox: {e}")
                return collected

        src_dir = busybox_dir / 'src' 

        if src_dir.exists():
            #TODO: Go through and rglob *.c files and shutil.copy to collection folder
            pass

        logger.info(f"Collected {len(collected)} benign C files from busybox")
        return collected

    def create_simple_benign_samples(self) -> List[Path]:
        samples = []

        # Sample 1: Simple Arithmetic
        sample1 = self.benign_dir / 'simple_arithmetic.c'
        with open(sample1, 'w') as f:
            f.write('''
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

int main() {
    int x = 10;
    int y = 20;

    int sum = add(x, y);
    int product = multiply(x, y);

    printf("Sum: %d\\n", sum);
    printf("Product: %d\\n", product);

    return 0;
}
''')

        samples.append(sample1)

        # Sample 2: Fibonacci (Control flow)
        sample2 = self.benign_dir / 'fibonacci.c'
        with open(sample2, 'w') as f:
            f.write('''
#include <stdio.h>

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n-1) + fibonacci(n-2);
}

int main() {
    for (int i = 0; i < 10; i++) {
        printf("Fib(%d) = %d\\n", i, fibonacci(i));
    }
    return 0;
}
''')

        samples.append(sample2)

        # Sample 3: String reverse
        sample3 = self.benign_dir / 'string_rev.c'
        with open(sample3, 'w') as f:
            f.write('''
#include <stdio.h>
#include <string.h>

void reverse_string(char *str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = temp;
    }
}

int main() {
    char str[] = "Hello World";
    printf("Original: %s\\n", str);

    reverse_string(str);
    printf("Reversed: %s\\n", str);

    return 0;
}
''')
        
        samples.append(sample3)

        # Sample 4: Fibonacci sum (iterative fib + fib_add)
        sample4 = self.benign_dir / 'fib_add.c'
        with open(sample4, 'w') as f:
            f.write('''#include <stdio.h>

int fib(int n) {
    int a = 0, b = 1;
    for (int i = 0; i < n; i++) {
        int tmp = a + b;
        a = b;
        b = tmp;
    }
    return a;
}

int fib_add(int n) {
    return fib(n) + fib(n + 1);
}

int main() {
    for (int i = 0; i < 10; i++) {
        printf("fib_add(%d) = %d\\n", i, fib_add(i));
    }
    return 0;
}
''')

        samples.append(sample4)

        logger.info(f"Created {len(samples)} simple benign samples")
        return samples

    def get_all_samples(self) -> List[Tuple[Path, bool]]:
        samples = []

        for f in self.benign_dir.glob('*.c'):
            samples.append((f, False))

        for f in self.malicious_dir.glob('*.c'):
            samples.append((f, True))

        return samples

    def collect_all(
        self,
        repos_dir: Path,
        angha_dir: Path = None,
        angha_samples: int = 100,
        angha_min_lines: int = 10,
        angha_seed: int = 42,
    ) -> List[Tuple[Path, bool]]:
        repos_dir.mkdir(parents=True, exist_ok=True)

        self.create_simple_benign_samples()

        if angha_dir:
            self.collect_anghabench_samples(
                angha_dir,
                n_samples=angha_samples,
                min_lines=angha_min_lines,
                seed=angha_seed,
            )

        # theZoo repo (malicious) — TODO: unzip with password 'infected', filter x86/x64 C files
        #zoo_dir = repos_dir / 'theZoo'
        #self.collect_thezoo_samples(zoo_dir)

        # Coreutils repo (benign) — TODO: copy standalone-compilable utilities
        #coreutils_dir = repos_dir / 'coreutils'
        #self.collect_coreutils_samples(coreutils_dir)

        return self.get_all_samples()

def main():
    parser = argparse.ArgumentParser(description='Perseus Source Collector')
    parser.add_argument('--output-dir', type=Path, default=Path('./data/source'),
                        help='Output directory for collected sources (default: ./data/source)')
    parser.add_argument('--repos-dir', type=Path, default=Path('./repos'),
                        help='Directory for cloned repos (default: ./repos)')
    parser.add_argument('--angha-dir', type=Path, default=None,
                        help='Path to local AnghaBench clone (optional)')
    parser.add_argument('--angha-samples', type=int, default=100,
                        help='Number of AnghaBench functions to sample (default: 100)')
    parser.add_argument('--angha-min-lines', type=int, default=10,
                        help='Minimum lines of code for AnghaBench filter (default: 10)')
    parser.add_argument('--angha-seed', type=int, default=42,
                        help='Random seed for reproducible AnghaBench sampling (default: 42)')
    args = parser.parse_args()

    collector = SourceCollector(args.output_dir)
    samples = collector.collect_all(
        repos_dir=args.repos_dir,
        angha_dir=args.angha_dir,
        angha_samples=args.angha_samples,
        angha_min_lines=args.angha_min_lines,
        angha_seed=args.angha_seed,
    )

    logger.info(f"\nCollection Summary:")
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Benign:    {sum(1 for _, is_mal in samples if not is_mal)}")
    logger.info(f"Malicious: {sum(1 for _, is_mal in samples if is_mal)}")
    logger.info(f"\nFirst 5 samples:")
    for path, is_mal in samples[:5]:
        logger.info(f"  {path.name} ({'malicious' if is_mal else 'benign'})")

if __name__ == '__main__':
    main()

