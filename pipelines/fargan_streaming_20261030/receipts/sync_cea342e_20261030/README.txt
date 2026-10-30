cea342e controls synchronized to active FARGAN host entry (no training)

Active: ../synthesize.py; frontend: new25_clarity_repair_20261030/repo/pipelines/prosody1000_rndvoc_opus24 at clean commit cea342e5f3ee6d2efd228da78b2f3cf895062118.
CLI: --identifier-prosody natural (default) / original. Both retain regular duration/U/C/first-digit/flow controls; natural adds exact connection and physical log-F0 controls.

The isolated cea_controls package contains the original identifier control modules with only local import names rewritten to relative package imports. install_flow/join_phrases are exact function AST extracts from source runtime.py. Neither source runtime.py nor its model loader is imported. No RND/HiFi weights load. FARGAN original 20D adapter and 91 tensors, acoustic136 tensors and BWE remain unchanged. RND-specific 80D mel energy offsets are not applied to FARGAN.

Natural identifiers require identifier-wide voiced pitch medians. For correctness, the complete bounded text unit's acoustic result is decoded and corrected before first PCM. This adds real full-acoustic waiting and a condition cache for those units; it is explicitly reported. FARGAN/BWE then stream normally. No identifier or original mode retains persistent acoustic cursor decode. This is sentence-prepared audio streaming, not character-input causal synthesis or a new 300ms M55 result.

SYNC_VERIFIED.json compares the actual source Engine.acoustic_piece AST against active FARGAN controls, with original source control modules loaded in temporary isolated names. All six acoustic arrays and eight control groups are exactly equal for4cases x2modes. No voice/model loader executes in the reference path. CLI WAVs cover ordinary Chinese/CZ2671/MU5108/English numbers and MU original. Real Popen first20ms packet arrives before process exit; concatenated stdout equals WAV, ordinary/pipe WAV identical. Plain Chinese WAV is identical to previous optimized release. All selected outputs have zero native/BWE quantizer clips. No subjective quality acceptance is asserted.

Original active entrypoints are in before/. Frozen new entrypoints are in after/. ENTRYPOINT.patch records active edits. All prior listening artifacts remain in place. Native integration/25-language frontend checks are separately owned by the parent task.

Host listening identity: original FARGAN/adapter FP16 parameters expanded/effective FP32; acoustic original I8 expanded toFP32; these WAVs are not native integer/MVE candidate audio. CZ and MU both have natural/original same-frontend ordinary CLI pairs.
