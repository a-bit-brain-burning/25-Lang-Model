Current acoustic streaming cursor

Import AcousticCursor from acoustic_cursor.py in this workbench. No shared
tts_stream_runtime source, model definition, weights, vocabulary or frontend
file was changed by this component.

The constructor accepts the existing predictor/features/language objects.
It encodes the complete phone unit and predicts durations once. pull(count)
advances only new frames, forwards GRU and all five causal-convolution states,
and returns the original waveform_condition normalized CPU float32 array.
frames/offset/durations/lang/state/dimension/calls match the prior cursor API.

Frame ownership and four position coordinates are computed only for the pull;
the entire audio-length owners/positions arrays are no longer retained. State
tensors are owned compact copies so views do not retain an old chunk's larger
storage. The current decoder state has 96,000 tensor bytes; that figure excludes
the model, encoder/prosody plan, inputs, framework and system memory.

cancel() releases encoder/prosody/decoder tensors and rejects further pulls.
reset(features=None, language=None) reuses the same predictor weights and makes
one new plan; omitted features/language reuse the stored phone-level request.
Configure any per-request encode/prosody control hooks before reset, as with a
new cursor. close() is terminal; create a new cursor to use the predictor again.
At ordinary EOS, subsequent positive pulls return an empty [0,D] float32 array.
report() describes actual frame progress, state shapes, cache bytes and status.

Finite current-model checks are in acoustic_cursor_validation/RESULT.json:
Chinese/English, chunks16/64; new cursor equals the old cursor with the same
chunking byte for byte. English equals complete reference, while Chinese has
existing small floating batching differences (maximum2.1458e-6) versus a full
decode. Frame layouts exactly match the original frame_layout contract.
Cancellation/reset reproduces the same prefix, interleaved requests reproduce
separate runs, compact state storage is checked, and all model state tensors
remain unchanged. No waveform shifting/fitting, training, M55 run or subjective
quality acceptance was performed. This remains complete-text-unit encoding
followed by audio streaming; it is not causal character-by-character input.
