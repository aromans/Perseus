#!/usr/bin/env python3

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple
import logging

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

    def collect_all(self, repos_dir) -> List[Tuple[Path, bool]]:
        repos_dir.mkdir(parents=True, exist_ok=True)

        self.create_simple_benign_samples()

        # TODO: Finish collection & copy logic and uncomment each function

        # theZoo repo (malicious)
        #zoo_dir = repos_dir / 'theZoo'
        #self.collect_thezoo_samples(zoo_dir)

        # Coreutils repo (benign)
        #coreutils_dir = repos_dir / 'coreutils'
        #self.collect_coreutils_samples(coreutils_dir)

        # Busybox repo (benign)
        #busybox_dir = repos_dir / 'busybox'
        #self.collect_busybox_samples(busybox_dir)

        return self.get_all_samples()

def main():
    
    # Setup directories
    output_dir = Path('./data/source')
    repos_dir  = Path('./repos')

    collector = SourceCollector(output_dir)
    samples   = collector.collect_all(repos_dir)

    logger.info(f"\nCollection Summary:")
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Benign: {sum(1 for _, is_mal in samples if not is_mal)}")
    logger.info(f"Malicious: {sum(1 for _, is_mal in samples if is_mal)}")

    logger.info(f"\nFirst 5 samples:")
    for path, is_mal in samples[:5]:
        label = "malicious" if is_mal else "benign"
        logger.info(f"  {path.name} ({label})")

if __name__ == '__main__':
    main()

