# Download, run and rebuild

This page is the practical companion to `README.md`. It covers three levels:

1. download and run the verified firmware (no build),
2. inspect the verification evidence,
3. full rebuild (requires an internal dependency bundle that is not fully public).

Everything here is QEMU-only (Cortex-M55 `mps3-an547`); no hardware board was used.

---

## 1. Download

**Repository visibility:** `a-bit-brain-burning/25-Lang-Model` is currently a
**private** repository. Unauthenticated `raw.githubusercontent.com` and
`api.github.com` requests return 404 even for valid files. Use an authenticated
SSH clone (as below), an authenticated HTTPS token, or the GitHub contents API
with a token. If you need anonymous access, make the repository public or ask the
owner for a token/archive.

### 1.1 Clone the whole snapshot (28 MB)

```bash
git clone --branch release/full25-qemu-60m-20261031 \
  https://github.com/a-bit-brain-burning/25-Lang-Model.git 25-lang-model
cd 25-lang-model/native/full25_firmware_20261030
```

### 1.2 Download only what you need (sparse checkout)

```bash
git clone --filter=blob:none --sparse \
  --branch release/full25-qemu-60m-20261031 \
  https://github.com/a-bit-brain-burning/25-Lang-Model.git 25-lang-model
cd 25-lang-model
# firmware + receipts + per-PC profiles only (~19 MB)
git sparse-checkout set \
  native/full25_firmware_20261030/firmware \
  native/full25_firmware_20261030/receipts \
  native/full25_firmware_20261030/profiles
```

### 1.3 Single file download (private repo: token required)

```bash
REF=release/full25-qemu-60m-20261031
REPO=a-bit-brain-burning/25-Lang-Model
PREFIX=native/full25_firmware_20261030
TOKEN=<your GitHub token with repo read access>

curl -sL -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github.raw" \
  "https://api.github.com/repos/$REPO/contents/$PREFIX/firmware/firmware_native25_dictcache_v9.elf?ref=$REF" \
  -o firmware_native25_dictcache_v9.elf

curl -sL -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github.raw" \
  "https://api.github.com/repos/$REPO/contents/$PREFIX/SOURCE_SHA256SUMS.txt?ref=$REF" \
  -o SOURCE_SHA256SUMS.txt

sha256sum firmware_native25_dictcache_v9.elf
# expected: see firmware/firmware_native25_dictcache_v9.elf.sha256
```

---

## 2. Run the verified firmware

Requirements: QEMU >= 8.2 with the `mps3-an547` machine (Cortex-M55).

```bash
mkdir -p run && cd run
cp ../firmware/firmware_native25_dictcache_v9.elf firmware.elf
qemu-system-arm -M mps3-an547 -cpu cortex-m55 -accel tcg,thread=single \
  -nographic -serial none -monitor none \
  -semihosting-config enable=on,target=native \
  -kernel firmware.elf
```

The firmware exercises all 25 languages in sequence and writes
`request_<language_id>_*.pcm/f32/u32` files into the current directory.
A successful run ends with:

```
N25_NATIVE_MULTILANG_SUBSET_PASS_NOT_FULL25
N25_COMBINED_COMPONENTS_PASS_NOT_FULL25_TTS
```

Expected image identity and metrics (from `receipts/v9/qemu.log` and
`receipts/v9/RESULT.json`):

- ROM load bytes: 9,887,216
- reserved physical RAM: 11,512,272
- 125/125 core output files byte-identical to frozen baselines
- 15/25 languages within 60,000,000 instruction events

The per-PC instruction profile CSVs were produced with a GPL QEMU TCG plugin;
the CSVs are included in `profiles/`, the plugin source is not part of this
snapshot.

---

## 3. Inspect the evidence

```bash
jq '.status, .identical_output_files, .elf_sha256,
    .memory.rom_load_bytes, .memory.reserved_physical_ram_bytes' \
  receipts/v9/RESULT.json
jq '.full_request_delivered_packet_events[] | {language,events}' \
  receipts/v9/RESULT.json
sha256sum firmware/firmware_native25_dictcache_v9.elf
gzip -dc profiles/native25_dictcache_v9.csv.gz | head
```

`receipts/v1/` contains the same set for the pre-optimization baseline
(`native25_text_paths_v1`). `receipts/v9/TIMING_v9.json` is the side-by-side
v1/v9 comparison.

---

## 4. Full rebuild

**Important:** a complete rebuild from this snapshot additionally needs an
internal dependency bundle (about 150 MB of backend source trees and ~18 MB of
derived model binary assets) that is not fully contained in this branch. The
snapshot includes everything that is project-specific and publicly meaningful:
scripts, source trees for the two firmware builds, the diff, all receipts and
profiles, the verified ELF, FEP resource packages, and the small host-side
tooling needed to regenerate resources. The missing pieces and their expected
locations are listed in `DEPENDENCIES.json` and checked by
`scripts/bootstrap_check.py`.

### 4.1 Host prerequisites

- Linux x86_64, Python 3.10 (the reference environment used 3.10.13)
- pip packages: `numpy`, `espeakng-loader` (the vendored host codec imports it;
  additional packages may be required by the optional `prepare_*.py` stages)
- QEMU system emulator >= 8.2 in `PATH`
- Arm GNU Toolchain 13.2.rel1 (`arm-none-eabi`), GCC 13.2.1:
  <https://developer.arm.com/downloads/-/arm-gnu-toolchain-13.2.rel1>
- XZ Utils **5.4.5** source: <https://tukaani.org/xz/xz-5.4.5.tar.xz>
- eSpeak NG **1.52.0** source, commit `4870adfa25b1a32b4361592f1be8a40337c58d6c`:
  `git clone https://github.com/espeak-ng/espeak-ng.git`

### 4.2 Workspace layout expected by the scripts

The scripts use absolute paths from the reference workstation that must be
adapted. The constants are:

| Script | Constant | Expected value in reference run |
|---|---|---|
| `build_resources.py` | `XZ` | `$WORK/m4_frontend_rom_accounting_20261023/sources/xz-5.4.5` |
| `build_espeak.py` | `VENDOR` | `$WORK/new25_clarity_repair_20261030/repo/pipelines/prosody1000_rndvoc_opus24/frontend_bundle/frontend/vendor` (`deps/host_vendor` here) |
| `build_espeak.py` | `SOURCE` | `$WORK/m4_frontend_4mb_20261022/upstream/espeak-ng` |
| `build_espeak.py` | `AUDIT` | `$WORK/m4_frontend_rom_accounting_20261023` (`deps/espeak_build_config` here) |
| `build_espeak.py` | `PATCHED` | `$WORK/m4_frontend_phonemes_20261023/patched` (`deps/espeak_patches` here) |
| `build_espeak.py` | `UNICODE` | `$WORK/m4_frontend_exact_20261023` (`deps/unicode_compact` here) |
| `build_resources.py` | `NATIVE` | `<full25 project>/native` (the 150 MB backend tree, see 4.3) |
| `build_resources.py` | `CC` | `native/bin/arm-none-eabi-gcc` (`deps/toolchain_wrapper/arm-none-eabi-gcc`) |

`$WORK` is the parent of the project directory; the reference value was
`/home/fomak/ffb`.

### 4.3 Backend tree (`native/`, not fully included)

`build_combined.py` reads the backend compile recipe from
`native/build/compact_ring_v2/BUILD.json`. A copy of that receipt is included at
`deps/backend_build/compact_ring_v2/BUILD.json`; it lists every backend compile
command. The backend source tree itself (150 MB including its own assets and
build receipts, inherited through symlinks from earlier optimization
worktrees) is required for a real rebuild and is not included here.

Expected layout of `native/` (as used by the reference build):

```
native/
  bin/            -> wrapper directory containing arm-none-eabi-gcc
  toolchain/      -> arm-none-eabi 13.2.1 sysroot (includes newlib)
  build/compact_ring_v2/BUILD.json
  src/  assets/  bwe_fast/  upstream/  third_party/  frontend_source/
```

The `native_final25/` and `native_dictcache/` trees in `source/` are overlays:
they contain `src/`, `dictionary_runtime/`, `frontend_patched/`, `resources/`,
small `assets/` inputs and `resource_profile.json`, and inherit the immutable
backend directories through relative symlinks (see `source/native_dictcache/README`
note and `RECOVERY.txt`).

### 4.4 Model binary assets

The following files are derived from the model weights tracked in this
repository under `models/` and were regenerated by the `prepare_*.py` scripts of
the earlier backend milestones:

```
native_*/assets/model.blocks.bin      native_*/assets/model.banked_blocks.bin
native_*/assets/model.bin             native_*/assets/model.banked_rom.bin
native_*/assets/model.banked_ram.bin  native_*/assets/model.lzma2
```

Their expected SHA256 identities are recorded in the included asset identity
JSONs (`MODEL_IDENTITY.json`, `MERGED_IDENTITY.json`,
`LOSSLESS_ASSETS_VERIFIED.json`, ...). The small build inputs (`weights.S`,
`zh_rom.S`, `model_index.c`, `language_norms.bin`, `metadata.deflate`, ...) are
included.

### 4.5 Build command

After restoring the layout above and editing the path constants:

```bash
cd <full25 project>
python3 build_combined.py \
  --tag REPRO_V1 \
  --backend-tag compact_ring_v2 \
  --arena-bytes 4080000 \
  --multilang --profile \
  --native-root <full25 project>/native_dictcache \
  --prewarm-g2p \
  --case-manifest FINAL25_SOURCE_IDENTITY.json
```

This regenerates FEP assets/ROMFS inputs, links the image, runs QEMU with the
instruction-profile plugin, and asserts that the 125 core output files match the
frozen per-language baselines. Without the profiling plugin, omit `--profile`
(byte-identity checks still run; the event CSV is not produced).

### 4.6 Reproducibility checks

```bash
sha256sum firmware/firmware_native25_dictcache_v9.elf
# expect the value in firmware/firmware_native25_dictcache_v9.elf.sha256
jq '.identical_output_files, .status' reports/<tag>/RESULT.json
```

---

## 5. Known gaps in this snapshot

- The backend `native/` tree and derived model binary assets are not included
  (see 4.3 / 4.4).
- The GPL QEMU instruction-profile plugin source and the host virtualenv are not
  included; the produced CSVs are included.
- The frontend/eSpeak host sources are public (eSpeak NG 1.52.0, XZ 5.4.5) but
  the three patched eSpeak files, the ARM object list, and the two compact
  Unicode tables are provided in `deps/` because they differ from upstream.
- Scripts contain reference-workstation absolute paths; search for
  `$WORK`-like constants listed in 4.2 and adapt them.
- Instruction events are not cycles; physical M55 latency is higher than the
  1-IPC 200 MHz figure. Hardware execution has not been verified.
