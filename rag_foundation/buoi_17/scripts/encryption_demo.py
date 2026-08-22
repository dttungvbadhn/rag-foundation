from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(key_path: Path) -> tuple[bool, Path]:
    from cryptography.fernet import Fernet

    source = ROOT / "outputs" / "audit_log.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.touch(exist_ok=True)
    if not key_path.exists():
        key_path.write_bytes(Fernet.generate_key())
    cipher = Fernet(key_path.read_bytes())
    encrypted = source.with_suffix(".jsonl.encrypted")
    encrypted.write_bytes(cipher.encrypt(source.read_bytes()))
    return cipher.decrypt(encrypted.read_bytes()) == source.read_bytes(), encrypted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args(); matched, output = run(args.key_file)
    print(f"ENCRYPT: PASS\nDECRYPT MATCH: {'PASS' if matched else 'FAIL'}\nOUTPUT: {output}\nPRODUCTION READY: NO")

