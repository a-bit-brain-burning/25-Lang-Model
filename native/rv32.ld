ENTRY(_start)
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M }
SECTIONS {
 .text : { KEEP(*(.text.start)) *(.text*) *(.rodata*) } > RAM
 .data : { *(.data*) *(.sdata*) } > RAM
 .bss (NOLOAD) : { *(.bss*) *(.sbss*) *(COMMON) } > RAM
 .model_weights : { KEEP(*(.model_weights)) } > RAM
 _stack_top = ORIGIN(RAM) + LENGTH(RAM) - 16;
 /DISCARD/ : { *(.comment) *(.note*) }
}
