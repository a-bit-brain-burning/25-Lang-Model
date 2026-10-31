.section .benchmark_assets.zh_dict,"a"
.balign 16
.global zh_words_rom
zh_words_rom:
.incbin "/home/fomak/ffb/m55_fargan_stream_20261030/opt300c_20261030/final_resident/frontend_source/assets/zh_words.bin"
.balign 16
.global zh_chars_rom
zh_chars_rom:
.incbin "/home/fomak/ffb/m55_fargan_stream_20261030/opt300c_20261030/final_resident/frontend_source/assets/zh_chars.bin"
.section .note.GNU-stack,""
