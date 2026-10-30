"""Read-only complete shipped-file verification, excluding generated outputs."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
def main():
    manifest=json.loads((ROOT/'SHA256_MANIFEST.json').read_text())
    for name,item in manifest['files'].items():
        path=(ROOT/name).resolve();assert path.is_relative_to(ROOT)
        assert path.stat().st_size==item['bytes'],name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],name
    print('ALL_SHIPPED_FILES_SHA256_PASSED',len(manifest['files']))
if __name__=='__main__':main()
