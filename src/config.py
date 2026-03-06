from pathlib import Path
import yaml

SYSTEM_PROMPT = (
    "You are a binary deobfuscation assistant. Given obfuscated x86-64 assembly code, "
    "you produce the equivalent clean, deobfuscated assembly. Preserve the function's "
    "semantics while removing obfuscation patterns such as MBA (mixed boolean-arithmetic), "
    "control flow flattening, and virtualization."
)


def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


_cfg = load_config()
OBF_TYPES: list = _cfg.get('pipeline', {}).get('obf_types', ['mba', 'control_flow', 'virtualization'])
