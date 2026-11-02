"""Pin accepted native sources and current frontend resources without changing them."""
from pathlib import Path
import hashlib
import json
import lzma
import shutil
import struct
import zlib

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
BASE = WORK/'streaming_accel_publish_20261030/repo/native/fargan_streaming_accel_20261030/resident_runtime_candidate_opt300e'
FRONTEND = WORK/'new25_clarity_repair_20261030/repo/pipelines/prosody1000_rndvoc_opus24'
VENDOR = FRONTEND/'frontend_bundle/frontend/vendor'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, document):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(document, ensure_ascii=False, indent=2)+'\n')


def main():
    receipt = ROOT/'SOURCE_IDENTITY.json'
    if receipt.exists():
        document = json.loads(receipt.read_text())
        for rel, digest in document['copied_files'].items():
            expected = document.get('recovered_files', {}).get(rel, digest)
            assert sha(ROOT/rel) == expected, rel
        print('Pinned source snapshot verified')
        return
    assert sha(BASE/'firmware.elf') == '2118790e4ce4facc632c694344d41a493ae5f2f87f25bb063d666fb746c52535'
    native = ROOT/'native'
    for name in ('src', 'frontend_source', 'bwe_fast', 'upstream', 'third_party', 'assets'):
        shutil.copytree(BASE/name, native/name)
    tools = WORK/'m55_fargan_stream_20261030/opt300c_20261030/opt300e_resident_20261030'
    for name in ('bin', 'toolchain'):
        (native/name).symlink_to(tools/name, target_is_directory=True)
    for name in ('build.py', 'build_native.py', 'build_o3_resident.py'):
        shutil.copy2(tools/name, native/name)
    # The published assembly has absolute references to its source checkout.
    # Pin the bytes into this snapshot and update only the assembly file paths.
    for name in ('weights.S', 'zh_rom.S'):
        p = native/'assets'/name
        source = p.read_text()
        import re
        for path in re.findall(r'\.incbin "([^"]+)"', source):
            old = Path(path)
            if name == 'weights.S':
                dest = native/'assets'/old.name
            else:
                dest = native/'frontend_source/assets'/old.name
            assert old.read_bytes() == dest.read_bytes(), old
            source = source.replace(path, str(dest))
        p.write_text(source)
    package_dir = ROOT/'resources/packages'
    shutil.copytree(VENDOR/'frontend/packages', package_dir)
    shutil.copytree(VENDOR/'extra_resources', ROOT/'resources/extra_resources')
    shutil.copy2(VENDOR/'frontend/MANIFEST.json', ROOT/'resources/UPSTREAM_MANIFEST.json')
    manifest = json.loads((VENDOR/'frontend/MANIFEST.json').read_text())
    sections = []
    packages = []
    for p in sorted(package_dir.glob('*.fep')):
        assert sha(p) == manifest['packages'][p.name], p
        blob = p.read_bytes()
        magic, version, n, index, payload, size, reserved = struct.unpack_from('<8s6I', blob)
        assert magic == b'FEPACK1\0' and version == 1 and index == 32 and size == len(blob) and not reserved
        assert payload == 32+56*n
        for i in range(n):
            name, codec, start, packed, raw, crc, window = struct.unpack_from('<32s6I', blob, 32+56*i)
            name = name.split(b'\0')[0].decode()
            value = blob[start:start+packed]
            assert window == 262144 and codec in (0, 1)
            if codec:
                value = lzma.decompress(value, format=lzma.FORMAT_RAW,
                    filters=[{'id': lzma.FILTER_LZMA2, 'dict_size': window}])
            assert len(value) == raw and zlib.crc32(value) == crc, (p, name)
            sections.append(dict(package=p.name, name=name, raw_bytes=raw,
                packed_bytes=packed, crc32=crc, sha256=hashlib.sha256(value).hexdigest()))
        packages.append(dict(name=p.name, bytes=len(blob), sha256=sha(p), sections=n))
    copied = {str(p.relative_to(ROOT)): sha(p) for folder in (native, ROOT/'resources')
        for p in sorted(folder.rglob('*')) if p.is_file() and not p.is_symlink()
        and 'bin' not in p.relative_to(folder).parts and 'toolchain' not in p.relative_to(folder).parts}
    write(ROOT/'resources/SECTIONS.json', dict(packages=packages, sections=sections,
        total_package_bytes=sum(p['bytes'] for p in packages),
        max_output_section_bytes=max(s['raw_bytes'] for s in sections),
        python_all_sections_verified=True, firmware_reader_verified=False))
    write(receipt, dict(baseline_elf_sha256=sha(BASE/'firmware.elf'),
        frontend_commit='cea342e5f3ee6d2efd228da78b2f3cf895062118',
        copied_files=copied, defaults_changed=False, training_updates=0))
    # The published checkout contains a later field-cache experiment in two
    # source files although its ELF/BUILD receipt is opt16. Recover only the
    # original field() and linker declarations, then require exact BUILD hashes.
    recover_baseline()
    write(ROOT/'STATUS.json', dict(status='NATIVE_FRONTEND_RESOURCE_PORT_IN_PROGRESS',
        complete=False, complete_frontend_languages=1, all25_same_firmware=False,
        full_system_rom_accepted=False, full_system_ram_accepted=False,
        all25_request_instruction_limit_accepted=False, training_updates=0))
    print(json.dumps(dict(pinned_files=len(copied), packages=len(packages),
        sections=len(sections), package_bytes=sum(p['bytes'] for p in packages))))


def recover_baseline():
    expected = {
        'native/src/quality.c': 'bea7615d110a6a03bce89f6c8a6dbeea35375e4fc6ff1686be8343581f531b8b',
        'native/src/m55.ld': '6aa6f7aa595f51f1d44c0a682a79ebde7071fdaaab62db110912be092267fd11',
    }
    p = ROOT/'native/src/quality.c'
    text = p.read_text()
    if '/* Direct-mapped resolved-tensor cache.' in text:
        start = text.index('/* Direct-mapped resolved-tensor cache.')
        end = text.index('static void acoustic_gru_cache_init', start)
        text = text[:start]+'static const n25_tensor *field(const char *prefix,const char *suffix){char name[160];namefmt(name,sizeof(name),"%s.%s",prefix,suffix);return find(name);}\n'+text[end:]
        p.write_text(text)
    p = ROOT/'native/src/m55.ld'
    text = p.read_text()
    if '  .fieldcache (NOLOAD)' in text:
        start = text.index('  .fieldcache (NOLOAD)')
        end = text.index('  /DISCARD/', start)
        text = text[:start]+text[end:]
        text = '\n'.join(line for line in text.split('\n') if not line.startswith('ASSERT(__fieldcache_end'))
        p.write_text(text)
    for rel, digest in expected.items():
        assert sha(ROOT/rel) == digest, ('Failed to recover exact opt16 source', rel)
    document = json.loads((ROOT/'SOURCE_IDENTITY.json').read_text())
    document['recovered_files'] = expected
    document['source_receipt_mismatch'] = 'Published source included later field-cache experiment; opt16 ELF unchanged. This isolated snapshot restores both exact source SHA values from opt16 BUILD.json.'
    write(ROOT/'SOURCE_IDENTITY.json', document)


if __name__ == '__main__':
    main()
