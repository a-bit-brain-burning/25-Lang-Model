.section .model_weights,"a"
.balign 16
.global storage_model_payload
storage_model_payload:
.incbin "/home/fomak/ffb/m55_fargan_stream_20261030/opt300c_20261030/release_trace/assets/model.blocks.bin"
.balign 16
.global storage_model_index
storage_model_index:
.incbin "/home/fomak/ffb/m55_fargan_stream_20261030/opt300c_20261030/release_trace/assets/model.index.bin"
.balign 16
.global storage_metadata_payload
storage_metadata_payload:
.incbin "/home/fomak/ffb/m55_fargan_stream_20261030/opt300c_20261030/release_trace/assets/metadata.deflate"
.balign 16
.global storage_metadata_index
storage_metadata_index:
.incbin "/home/fomak/ffb/m55_fargan_stream_20261030/opt300c_20261030/release_trace/assets/metadata.index.bin"
.section .note.GNU-stack,""
