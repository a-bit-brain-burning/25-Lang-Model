.syntax unified
.arch armv8.1-m.main
.arch_extension mve.fp
.thumb
.section .vectors,"a",%progbits
.word __stack_top
.word reset
.rept 14
.word fault
.endr
.section .text.reset,"ax",%progbits
.global reset
.thumb_func
reset:
    cpsid i
    ldr r0, =0xe000ed08
    ldr r1, =__code_start
    str r1, [r0]
    ldr r0, =0xe000ed88
    ldr r1, [r0]
    orr r1, r1, #0xf00000
    str r1, [r0]
    dsb
    isb
    ldr r0, =__data_load
    ldr r1, =__data_start
    ldr r2, =__data_end
1:  cmp r1, r2
    bhs 2f
    ldr r3, [r0], #4
    str r3, [r1], #4
    b 1b
2:  ldr r0, =__bss_start
    ldr r1, =__bss_end
    movs r2, #0
3:  cmp r0, r1
    bhs 4f
    str r2, [r0], #4
    b 3b
4:  ldr r0, =__stack_bottom
    msr msplim, r0
    ldr r1, =__stack_top
    ldr r2, =0xa5a5a5a5
5:  cmp r0, r1
    bhs 6f
    str r2, [r0], #4
    b 5b
6:  bl main
    bl bench_exit
    b .
.section .text.fault,"ax",%progbits
.global fault
.thumb_func
fault:
    movs r0, #4
    ldr r1, =fault_message
    bkpt #0xab
    movs r0, #0x20
    ldr r1, =fault_status
    bkpt #0xab
    b .
.align 2
fault_status: .word 0x20026, 90
fault_message: .asciz "M55_FATAL_EXCEPTION\n"
