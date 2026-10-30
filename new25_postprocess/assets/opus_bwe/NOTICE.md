This package contains a source subset of Xiph Opus 1.6 and the official generated
BBWENet v2 C weights distributed for that release.

Upstream source: https://github.com/xiph/opus/tree/v1.6
Pinned source commit: a8b13e40d751c7b40833b94fc9437c5c3439da89
Official model distribution: https://media.xiph.org/opus/models/

The upstream BSD-style license is reproduced in licenses/COPYING. Individual
source files retain their applicable two-clause or three-clause BSD notices and
copyright attributions, including those for the DSP/FFT helpers. Upstream patent
licensing references and contribution terms are retained in
licenses/LICENSE_PLEASE_READ.txt.

The generated weight source and header are copied without numerical changes
from the official model archive identified in PROVENANCE.json. The C source is
stored using lossless gzip; its original and decompressed hashes are identical.
Only the BWE source subset and the three documented build-scope patches are
redistributed here. The complete model archive, training checkpoint, and other
models are not included.
