#!/usr/bin/env python3
"""Check that the external inputs listed in DEPENDENCIES.json are reachable.

Usage:
  WORK_ROOT=/path/to/work python3 scripts/bootstrap_check.py [--root <snapshot dir>]

Exit code 0 only when every external-internal/external-public item referenced by
the build recipe exists at its adapted location. Absolute reference paths are
shown so they can be mapped to the local workstation.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def expand(value, work_root):
    return value.replace('$WORK', str(work_root))


def check_file(path):
    return Path(path).is_file()


def check_dir(path):
    return Path(path).is_dir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work-root', default=os.environ.get('WORK_ROOT',
        str(Path(__file__).resolve().parents[4])))
    args = parser.parse_args()
    root = args.root.resolve()
    work = Path(args.work_root).resolve()
    doc = json.loads((root / 'DEPENDENCIES.json').read_text())
    print(f'snapshot: {root}')
    print(f'WORK_ROOT: {work}\n')
    missing = []
    ok = 0
    for item in doc['items']:
        ident = item['id']
        if item.get('kind') == 'included':
            path = root / item['path']
            state = 'OK' if path.exists() else 'MISSING'
            if state == 'MISSING':
                missing.append(ident)
            print(f'[{state:7}] {ident:24s} {path.relative_to(root)}')
            ok += state == 'OK'
            continue
        ref = expand(item.get('reference_path', ''), work) if item.get('reference_path') else ''
        present = bool(ref) and Path(ref).exists()
        state = 'OK' if present else 'MISSING'
        if not present and item.get('kind') == 'external-internal':
            missing.append(ident)
        elif not present and item.get('kind') == 'external-public':
            missing.append(ident)
        print(f'[{state:7}] {ident:24s} {ref}')
        if item.get('check', {}).get('version_hint') and present:
            pass
        ok += state == 'OK'
    # tool version probes
    print('\nversion probes')
    gcc = shutil.which('arm-none-eabi-gcc')
    if gcc:
        version = subprocess.run([gcc, '--version'], capture_output=True, text=True).stdout.splitlines()[0]
        print(f'  arm-none-eabi-gcc: {version}')
    qemu = shutil.which('qemu-system-arm')
    if qemu:
        machines = subprocess.run([qemu, '-M', 'help'], capture_output=True, text=True).stdout
        print(f'  qemu-system-arm: mps3-an547 {"present" if "mps3-an547" in machines else "MISSING"}')
    print(f'\nsummary: {ok}/{len(doc["items"])} items present, {len(missing)} missing')
    if missing:
        print('missing: ' + ', '.join(missing))
        print('see BUILD_AND_DOWNLOAD.md section 4 for how to obtain each item')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
