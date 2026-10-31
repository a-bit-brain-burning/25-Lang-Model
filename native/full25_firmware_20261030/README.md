# full25_firmware_20261030 — same-image 25-language native text frontend on QEMU Cortex-M55

Status: intermediate milestone, verified on QEMU only (no hardware board executed).
All numbers are QEMU dynamic instruction events, not physical M55 cycles.

**Download / run / rebuild:** see [`BUILD_AND_DOWNLOAD.md`](BUILD_AND_DOWNLOAD.md).
Dependency inventory: [`DEPENDENCIES.json`](DEPENDENCIES.json); run
`python3 scripts/bootstrap_check.py` (after adapting paths) to see which
external inputs are present. A complete rebuild additionally needs the internal
backend tree (see BUILD_AND_DOWNLOAD.md section 4.3).

## 1. What is verified

One firmware image accepts raw ordinary NFC text for each of 25 languages
(bg cs da de el en es et fi fr hr hu it lt lv nl no pl pt ro sk sl sv tr zh),
runs the native text frontend, acoustic model, adapter, FARGAN and BWE inside the
image, and emits complete 24 kHz audio.

- 125/125 core output files (`condition.f32`, `durations.u32`, `output.pcm`,
  `output24.pcm`, `fargan.f32`) are byte-identical to the frozen per-language
  native INT8 baselines; 20 ms packet concatenations are also identical.
- Strict ROI: `tts_begin_language_text_request` entry (included) to
  `tts_first_effective_packet_delivered` entry (excluded); the delivered packet is
  480 PCM24 samples with RMS/peak checks. Boot is separate; no sentence answers
  are cached.
- No training, no new quantization, no weight/arithmetic/default-pointer changes.

## 2. Builds in this snapshot

### v1 baseline — `native25_text_paths_v1`
- ELF SHA256 `77390626db625ac007b76f4ad3dd3d6cb67cc79792a539bc7e90c1743633921e`
- ROM load bytes 9,885,272; reserved physical RAM 9,860,312 (both < 10 MB)
- Languages ≤ 60M instruction events: 15/25
- Over limit: en 1,279.95M, fr 900.15M, ro 658.35M, lt 578.18M, it 356.83M,
  pt 144.99M; slightly over: fi 64.18M, hu 65.31M, no 65.04M, sv 62.03M

### v9 dictionary decode cache — `native25_dictcache_v9`
- ELF SHA256 `$(cat firmware/firmware_native25_dictcache_v9.elf.sha256 | cut -d' ' -f1)`
- ROM load bytes 9,887,216 (unchanged in substance); reserved physical RAM 11,512,272
  (RAM relaxation to ~1.2–1.6 MB over 10 MB was approved for this milestone)
- 125/125 outputs still byte-identical.

| language | v1 instruction events | v9 instruction events | speed-up |
|---|---:|---:|---:|
| ro | 658,352,644 | 138,623,338 | 4.75x |
| en | 1,279,949,118 | 286,714,137 | 4.46x |
| it | 356,829,532 | 97,163,593 | 3.67x |
| fr | 900,148,181 | 257,065,253 | 3.50x |
| pt | 144,986,834 | 63,270,091 | 2.29x |
| lt | 578,175,936 | 439,369,897 | 1.32x |

Exact values are in `receipts/v9/RESULT.json`; the v1/v9 comparison is in
`receipts/v9/TIMING_v9.json`.

### What the v9 change does
`patches/dict_cache_v9.patch` adds a request-scoped decoded-dictionary cache:
the FEP lexicon sections (keys / ids / pool) are LZMA-decoded once per request
into a 1,650,000-byte SRAM arena, exact key and id indexes are built once per
keys block, and subsequent per-word lookups read decoded bytes instead of
re-decoding whole sections. Two block slots cover the 2–4 key blocks of the six
dictionary-heavy languages. Output bytes are unchanged because only the caching
of read-only decoded data changed.

## 3. Layout

```
scripts/            build and preparation scripts (build_combined.py, build_resources.py,
                    prepare_*.py, freeze_build_inputs.py, profiling/accounting tools,
                    bootstrap_check.py)
source/native_final25/   baseline source tree (src/, dictionary_runtime/,
                    frontend_patched/, resources/, small assets/ inputs,
                    resource_profile.json)
source/native_dictcache/ v9 source tree (same layout + dictionary cache)
patches/dict_cache_v9.patch   full diff of v9 against the baseline tree
deps/               host-side tooling and build inputs: fe_codec.py + reader
                    shared libraries, compiler wrapper, patched eSpeak files,
                    ARM object list, compact Unicode tables, backend
                    compact_ring_v2 BUILD.json compile receipt
resources/          FEP resource packages and SECTIONS.json used by the build
receipts/v1/        RESULT.json, qemu.log, BUILD.json, FROZEN_INPUTS.json
receipts/v9/        RESULT.json, qemu.log, BUILD.json, FROZEN_INPUTS.json, TIMING_v9.json
receipts/           STATUS.json, PROGRESS_CHECKPOINT.json, RECOVERY.txt,
                    ALL25_TEXT_PATHS_VERIFIED.json, LATENCY_HOTSPOTS.json,
                    *IDENTITY*.json (source/asset identity), analysis notes
profiles/           per-PC instruction-count CSVs (gzip) for v1 and v9
firmware/           v9 firmware ELF + SHA256
BUILD_AND_DOWNLOAD.md   download, run, verify and full-rebuild instructions
DEPENDENCIES.json       machine-readable dependency inventory and expected paths
```
```

`FROZEN_INPUTS.json` records 366 actual compiler dependencies and incbin inputs
by SHA256 for the v9 build.

## 4. Exclusions (deliberately not uploaded)

- The backend `native/` tree (~150 MB: backend sources, build receipts, derived
  model binaries inherited through symlinks) — required for a real rebuild and
  described in BUILD_AND_DOWNLOAD.md section 4.3; its complete compile-command
  receipt is included at `deps/backend_build/compact_ring_v2/BUILD.json`.
- Derived model binary assets (~18 MB per source tree: `model.blocks.bin`,
  `model.banked_*.bin`, `model.lzma2`, `model.bin`) — regenerated from the model
  weights already present in this repository under `models/` by the
  `prepare_*.py` scripts. Small build inputs and all identity JSONs are included.
- `frozen_inputs/sha256/` content-addressed dependency store (~24 MB) — the
  manifests (`receipts/*/FROZEN_INPUTS.json`) are included; the store is
  regenerated by `freeze_build_inputs.py`.
- `build/`, `reports/*/request_*` raw audio/capture files (~443 MB), QEMU
  binaries, host virtualenv, and third-party upstream source checkouts.
- The GPL QEMU TCG instruction-profile plugin source; its output CSVs are
  included under `profiles/`.

## 5. Build recipe (as executed)

Workspace layout assumed (paths are absolute in the scripts; adapt as needed):

```
/home/fomak/ffb/m55_fargan_stream_20261030/
  full25_firmware_20261030/          <- this project
  native_integer_front/toolchain/    <- arm-none-eabi-gcc 13.2.1 sysroot
  round2_compressed/bin/             <- compiler wrapper
  ...

python3 build_combined.py \
  --tag NEW_TAG --backend-tag compact_ring_v2 --arena-bytes 4080000 \
  --multilang --profile \
  --native-root <workspace>/full25_firmware_20261030/native_dictcache \
  --prewarm-g2p --case-manifest FINAL25_SOURCE_IDENTITY.json
```

The build requires (external to this snapshot): the Cortex-M55 toolchain/sysroot,
espeak-ng source tree, xz-5.4.5 source tree, the compact_ring_v2 backend build
receipt at `native/build/compact_ring_v2/BUILD.json`, the model weight assets
derived from this repository's `models/`, and the host Python environment used by
the `prepare_*.py` scripts. `build_combined.py` regenerates FEP assets and ROMFS
inputs, links the image, then runs QEMU with the instruction profiling plugin and
asserts byte-identical core outputs against the frozen baselines.

## 6. Verification entry points

- `receipts/v9/RESULT.json` — status PASSED, `identical_output_files: 125`,
  `elf_sha256`, memory segments, per-language event list.
- `receipts/v9/qemu.log` — full QEMU console output including every metric.
- `profiles/native25_dictcache_v9.csv.gz` — per-PC per-request instruction events.
- `firmware/firmware_native25_dictcache_v9.elf` — verified image.

QEMU command used:

```
qemu-system-arm -M mps3-an547 -cpu cortex-m55 -accel tcg,thread=single \
  -nographic -serial none -monitor none \
  -semihosting-config enable=on,target=native -kernel firmware.elf
```

with the GPL instruction-profile plugin attached for the ROI CSV.

## 7. Provenance and limits

- Workspace: `/home/fomak/ffb/m55_fargan_stream_20261030/full25_firmware_20261030`,
  snapshot date 2026-10-31.
- Instruction events are architectural events, not cycles; the M55 is in-order and
  MVE operations take multiple beats, so physical latency is higher than the
  1-IPC 200 MHz figure.
- No development board was used. This snapshot is an intermediate milestone:
  en/fr/ro/it/pt/lt and the four slightly-over languages have not reached the
  60M-event target yet.
- Upstream components retain their own licenses (espeak-ng GPL, liblzma public
  domain, Opus BWE BSD, etc.); see the repository for license texts.
