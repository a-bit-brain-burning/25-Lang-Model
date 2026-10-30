ENTRY(reset)
MEMORY {
 CODE (rx) : ORIGIN = 0x10000000, LENGTH = 512K
 DATA (rw) : ORIGIN = 0x30000000, LENGTH = 512K
 WEIGHTS (r) : ORIGIN = 0x38000000, LENGTH = 8M
}
SECTIONS {
 .vectors : { KEEP(*(.vectors)) } > CODE
 .text : { *(.text*) *(.rodata*) } > CODE
 .data : { *(.data*) } > DATA
 .bss (NOLOAD) : { *(.bss*) *(COMMON) } > DATA
 .model_weights : { KEEP(*(.model_weights)) } > WEIGHTS
 /DISCARD/ : { *(.comment) *(.ARM.exidx*) *(.ARM.extab*) *(.note*) }
}
