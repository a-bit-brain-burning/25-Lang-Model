ENTRY(reset)
MEMORY {
  CODE (rx) : ORIGIN = 0x10000000, LENGTH = 512K
  DTCM (rw) : ORIGIN = 0x30000000, LENGTH = 512K
  SRAM (rw) : ORIGIN = 0x60000000, LENGTH = 8M
  QSPI (r) : ORIGIN = 0x38000000, LENGTH = 8M
}
__stack_capacity = DEFINED(STACK_BYTES) ? STACK_BYTES : 65536;
__arena_capacity = DEFINED(ARENA_BYTES) ? ARENA_BYTES : 1048576;
SECTIONS {
  .vectors : { __code_start = .; KEEP(*(.vectors)) } > CODE
  .text : { *(.text*) *(EXCLUDE_FILE(*bbwenet_data_rowmajor.o) .rodata*) } > CODE
  .ARM.extab : { *(.ARM.extab*) } > CODE
  .ARM.exidx : { __exidx_start = .; *(.ARM.exidx*) __exidx_end = .; } > CODE
  .data : ALIGN(16) {
    __data_start = .; *(.data*) . = ALIGN(16); __data_end = .;
  } > DTCM AT> CODE
  __data_load = LOADADDR(.data);
  __image_end = LOADADDR(.data) + SIZEOF(.data);
  .bss (NOLOAD) : ALIGN(16) {
    __bss_start = .; *(.bss*) *(.lexcache*) *(COMMON) . = ALIGN(16); __bss_end = .;
  } > DTCM
  .stack ORIGIN(DTCM) + LENGTH(DTCM) - __stack_capacity (NOLOAD) : {
    __stack_bottom = .; . += __stack_capacity; __stack_top = .;
  } > DTCM
  .assets : ALIGN(16) {
    __assets_start = .; *bbwenet_data_rowmajor.o(.rodata*) KEEP(*(.model_weights*)) KEEP(*(.benchmark_assets*))
    . = ALIGN(16); __assets_end = .;
  } > QSPI
  .arena (NOLOAD) : ALIGN(16) {
    __arena_start = .; . += __arena_capacity; __arena_end = .;
  } > SRAM
  .zhprewarm (NOLOAD) : ALIGN(16) {
    __zhprewarm_start = .; *(.zhprewarm*) . = ALIGN(16); __zhprewarm_end = .;
  } > SRAM
  .math_luts (NOLOAD) : ALIGN(16) { *(.math_luts*) . = ALIGN(16); __math_luts_end = .; } > SRAM
  .dictcache (NOLOAD) : ALIGN(16) {
    __dictcache_start = .; *(.dictcache*) . = ALIGN(16); __dictcache_end = .;
  } > SRAM
  /DISCARD/ : { *(.comment) *(.note*) }
}
ASSERT(__bss_end <= __stack_bottom, "DTCM static/stack overlap")
ASSERT(__arena_end <= ORIGIN(SRAM) + LENGTH(SRAM), "Assets/workspace exceed SRAM")
ASSERT(__zhprewarm_end <= ORIGIN(SRAM) + LENGTH(SRAM), "Frontend prewarm image exceeds SRAM")
ASSERT(__stack_capacity >= 4096, "Stack reservation too small")

ASSERT(__math_luts_end <= ORIGIN(SRAM) + LENGTH(SRAM), "Math tables exceed SRAM")
ASSERT(__dictcache_end <= ORIGIN(SRAM) + LENGTH(SRAM), "Dictionary cache exceeds SRAM")
