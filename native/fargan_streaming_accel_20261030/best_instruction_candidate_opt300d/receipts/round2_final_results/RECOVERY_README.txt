TERMINAL for round2 instruction reduction (opt9..opt12). Do not rerun completed tags.
Cold ROI (begin=tensor_views_prepare, single-request release build): opt12 61,984,476 events -> 309.922 ms at 1 IPC.
Warm ROI (begin=acoustic_begin, excludes per-request view prep): opt12 59,966,762 events -> 299.834 ms at 1 IPC.
All candidates keep the six output files byte-identical to integrated and final_native zh references.
WARM IS NOT YET A FIRMWARE FACT: only the profiling begin anchor moved; a resident firmware that prepares views
outside the request must still be built and verified with two requests (A0/A1). Do not quote 299.8ms as hardware.
REAL CYCLES: MVE 64-bit instructions occupy two beats, loads/branches take extra cycles, so hardware latency is
above the naive 1 IPC figure. Needs a cycle model or hardware measurement.
Next: build resident variant with opt12 sources, verify A0/A1 outputs and per-request event count; then continue
cold-scope cuts (n25_unpack f16 fast path, adaconv, compute_linear_c, matrix epilogue).
