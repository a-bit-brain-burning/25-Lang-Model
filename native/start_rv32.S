.section .text.start
.global _start
_start:
    la sp, _stack_top
    li t0, 0x6000
    csrs mstatus, t0
    csrw fcsr, zero
    call main
1:  j 1b
