.syntax unified
.arch armv8.1-m.main
.arch_extension mve.fp
.thumb
.section .vectors,"a"
.word 0x30080000
.word reset
.section .text
.global reset
.thumb_func
reset:
    ldr r0, =0xe000ed88
    ldr r1, [r0]
    orr r1, r1, #0xf00000
    str r1, [r0]
    dsb
    isb
    bl main
    b .
