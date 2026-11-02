 .section .model_weights,"a"
.balign 16
.global storage_model_payload
storage_model_payload:
.incbin "/home/fomak/ffb/m55_fargan_stream_20261030/full25_firmware_20261030/native_final25/assets/model.lzma2"
.balign 16
.global storage_tensor_rom_bank
storage_tensor_rom_bank:
.incbin "/home/fomak/ffb/m55_fargan_stream_20261030/full25_firmware_20261030/native_final25/assets/model.banked_rom.bin"
.section .note.GNU-stack,""
