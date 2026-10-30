"""Unpack and verify the one official static BWE C weight source."""
from pathlib import Path
import gzip
import hashlib
import sys

EXPECTED = "64d95f4ac400a6e2a2ec182cad6eea211c91ebac983a741741beb5957db6e0c3"


def main():
    source, output = map(Path, sys.argv[1:])
    data = gzip.decompress(source.read_bytes())
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise SystemExit("Opus BWE weight SHA256 mismatch")
    if output.exists() and output.read_bytes() == data:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(output)


if __name__ == "__main__":
    main()
