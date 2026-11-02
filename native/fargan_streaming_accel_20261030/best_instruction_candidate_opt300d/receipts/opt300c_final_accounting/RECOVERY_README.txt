TERMINAL: dynamic O2/O3 instruction profiling and 300 ms accounting closure completed.
Do not rerun the profiled tags unless the CSVs are missing; both profile repeats were byte-identical.
Measured executed guest instructions for one ordinary Chinese first-audio request:
  release O2 (diagnostic/report-free, byte-identical six files): 72,681,563 events
  release O3 (same source + -O3 + -fno-aggressive-loop-optimizations, byte-identical): 68,816,649 events
Naive single-issue 1-instruction-per-cycle floor at 200 MHz: O2 363.408 ms, O3 344.083 ms.
300 ms at 200 MHz = 60,000,000 cycles. O3 is still 8,816,649 instructions above that budget
(12.81%). MVE 64-bit beats and MOVE/Thumb-2 issue reality only make real latency larger, never
smaller, so 300 ms is NOT met and compiler-level O2/O3 changes alone cannot meet it.
The old 318.58 ms "partial estimate" is a partial priced subtotal, not a whole-request lower
bound; measured instruction events already exceed it. Do not resurrect it as a latency claim.
O3 is byte-identical (six files) and cuts events 5.3176%, so it is a valid future candidate, but
it is NOT adopted: default pointer, delivery numbers and prior receipts remain unchanged.
Next real work: cut >= 8,816,649 instructions from the O3 request, dominated by
packed_matrix_pair/packed_matrix8 (27,926,376 events, 40.58% of O3) plus adaconv/cgemv/
compute_linear_c/ensure_raw_condition_chunk/n25_unpack; then re-profile the same ROI.
Never run the old operation-counter estimator on these report-free builds. Events are not cycles.
No weights, training, math, pause, context or RNG changed; all six output files are unchanged.
