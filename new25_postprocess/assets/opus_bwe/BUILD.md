# Opus 1.6 standalone BWE source subset

This package builds the official BBWENet v2 PCM bandwidth extension demo from
Xiph Opus tag `v1.6`, commit `a8b13e40d751c7b40833b94fc9437c5c3439da89`.
It contains only the BWE inference translation units, required DSP helpers and
headers, the one generated BWE C weight source, and upstream license notices.
The generated weight source is stored as deterministic gzip with `mtime=0`;
building verifies its decompressed SHA256 before compiling it. No download or
PyTorch installation is required.

Requirements: CMake 3.16+, Python 3 standard library, and Clang or GCC with a
POSIX C99 environment and the system math library. Build outside this source
folder:

```sh
cmake -S opus_bwe -B build/opus_bwe -DCMAKE_BUILD_TYPE=Release
cmake --build build/opus_bwe --target bwe_demo --parallel 2
```

An existing interpreter can be selected with `-DPython3_EXECUTABLE=python-path`.
The generic `-DPython_EXECUTABLE=python-path` hint is accepted for the parent
runtime. The only executable target is `bwe_demo`; the resulting executable is
`build/opus_bwe/bwe_demo`.

`-DOPUS_BWE_SIMD=AUTO` is the default. Native AArch64/arm64 builds use the official
NEON implementations, including on macOS arm64 and Linux AArch64. All other
architectures, including Linux x86_64, select portable C implementations. Force
those implementations on any supported architecture with
`-DOPUS_BWE_SIMD=SCALAR`. `NEON` can be selected explicitly on AArch64. The scalar
option also disables the implicit vector helpers in upstream `vec.h`; a compiler
may still automatically vectorize portable C. No Linux-specific runtime CPU
probe, AVX requirement, or Apple-only compiler flag is needed. For cross builds,
provide the normal CMake toolchain and set `SCALAR` if the target is not AArch64.

## Frozen PCM interface

```sh
build/opus_bwe/bwe_demo input.pcm output.pcm
```

Input is headerless mono native-endian signed PCM16 at 16000 Hz. Only complete
320-sample frames are processed. Each complete frame produces exactly 960
native-endian PCM16 samples at 48000 Hz. The executable performs no codec
encode/decode, normalization, WAV parsing, padding, delay trimming, or final
resampling. Model state starts fresh in each invocation.

The parent WAV adapter must preserve the accepted experiment protocol:

1. Keep the original 16 kHz int16 values. Pad the last frame to 320 samples, then
   append two additional zero frames (640 input samples).
2. Run `bwe_demo` and check that the output has three times the padded input
   sample count.
3. Remove 34 samples from the start of the 48 kHz signal: 21 samples from the
   official output buffer plus the official checkpoint's 13-sample upsampler
   delay. Keep the appended post-roll during filtering.
4. Apply `scipy.signal.resample_poly(up=1, down=2, window=("kaiser", 8.6),
   padtype="constant")` to the float64 signal. The reference adapter scales PCM16
   by `1/32768` before filtering, then multiplies by 32768 before quantization.
5. Crop to `(3 * original_input_samples + 1) // 2` samples, use round-to-nearest,
   saturate to `[-32768, 32767]`, and write mono PCM16LE WAV at 24000 Hz.

The official BWE kernel separately saturates its 48 kHz output to ±32767. This
behavior is retained. The model generator has 367,890 trainable parameters and
368,414 state values including fixed DSP/window values; inference uses generated
static C arrays, not the training checkpoint.

## Source changes and validation

`patches/bwe-only.patch` is the complete patch against the pinned upstream files:

- Remove the demo's unused `lpcnet.h` include.
- Remove unrelated model headers from `nnet.c`, constrain recurrent scratch space
  to the only compiled model, and retain a 1024-entry convolution scratch bound.
  This changes workspace bounds, not the arithmetic on valid BWE layers.
- Gate implicit architecture-specific `vec.h` helpers with `OPUS_BWE_SCALAR`, so
  the portable fallback uses upstream generic C routines.

CMake defines the existing upstream `DISABLE_LACE` and `DISABLE_NOLACE` switches.
Function/data sections and the platform linker garbage collector omit unused
codec-facing routines from shared upstream files. No BWE layer, tensor, feature,
window, resampler, activation, or quantizer calculation was rewritten. No
fast-math or floating-point contraction override is added.

The release audit compared 16 real inputs (eight original TTS and eight GTCRN
outputs) with the existing complete official macOS arm64 NEON build. All 48 kHz
PCM outputs and the aligned final 24 kHz outputs matched byte for byte, including
the pre-existing accepted 24 kHz PCM files. macOS arm64 scalar and macOS x86_64
scalar also compiled. Linux AArch64 GCC 14 built and executed both scalar and
NEON variants against the same 16 real inputs and silence, sine, high-amplitude,
and impulse probes. The available Linux container lacked CMake, so those builds
used equivalent direct GCC commands. Linux x86_64 execution was not available;
its portable C path was cross-compiled for x86_64 macOS, whose execution required
unavailable Rosetta. Different compilers, SIMD reduction orders and system math
libraries need not produce identical PCM rounding. `PROVENANCE.json` records
precise validation scope and all original and shipped file hashes.

## License

Retain `licenses/COPYING`, `licenses/LICENSE_PLEASE_READ.txt`, all individual
source notices, and `NOTICE.md` when redistributing this subset or its binary.
