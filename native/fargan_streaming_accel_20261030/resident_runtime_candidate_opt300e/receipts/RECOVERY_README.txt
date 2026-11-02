TERMINAL for the resident full-request work (opt300e_resident_20261030), tag opt16_fast / rebuilt opt16_final.
ELF 2118790e4ce4facc632c694344d41a493ae5f2f87f25bb063d666fb746c52535 (rebuild after reverting the experimental
field-name cache reproduced this exact SHA).
QEMU NEW25_PERSISTENT_RUNTIME_PASS, 4 requests, 456/456 firmware-emitted output files byte-identical to
final_resident/reports/resident_final_v1/run_001, six file identities unchanged.
ROI (architectural instruction events, not cycles):
  request scope  (acoustic_begin -> first active speech packet): 59,747,483  = 298.737 ms @ 1 IPC
  full API scope (API call incl. native text frontend -> first):  59,988,134  = 299.941 ms @ 1 IPC  (< 60M by 11,866)
This round: boot-time whole-dictionary page image (2439 pages / 1,516,448 B in SRAM), t2s first-byte dispatch index,
state-only frontend pipeline without scratch formatting, targeted N25TextFeatures zeroing, inlined dictionary key compare.
A global field-name cache was tried and REJECTED: the linear probe made the full scope slower (61.36M); it was reverted.
Caveats: margin is only 11,866 events; the benchmark is one ordinary Chinese sentence; real M55 cycles are higher than the
1 IPC line (in-order, MVE 64-bit ops are two-beat), so hardware 300 ms is NOT accepted. Added RAM: 1.45 MB dictionary
image (SRAM) plus small tables; complete-system RAM/ROM must be re-verified.
