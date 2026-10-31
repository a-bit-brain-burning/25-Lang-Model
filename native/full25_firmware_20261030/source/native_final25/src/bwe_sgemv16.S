.syntax unified
.thumb
.text
.align 2
.global bwe_sgemv16
.type bwe_sgemv16,%function
.thumb_func
/* w0,x4,out8,tiles16=12,cols16,stride_bytes20. Exact MUL then ADD. */
.macro SG_TAP reg
 vldmia r1,{s16-s31}
 vmul.f32 q4,q4,\reg
 vmul.f32 q5,q5,\reg
 vmul.f32 q6,q6,\reg
 vmul.f32 q7,q7,\reg
 vadd.f32 q0,q0,q4
 vadd.f32 q1,q1,q5
 vadd.f32 q2,q2,q6
 vadd.f32 q3,q3,q7
 add r1,r1,r6
.endm
bwe_sgemv16:
 push {r4-r11,lr}
 vpush {d8-d15}
 mov r9,r0
 ldr r7,[r9,#0]
 ldr r10,[r9,#4]
 ldr r8,[r9,#8]
 ldr r4,[r9,#12]
 ldr r5,[r9,#16]
 ldr r6,[r9,#20]
 cmp r4,#0
 beq 9f
1:
 veor q0,q0,q0
 veor q1,q1,q1
 veor q2,q2,q2
 veor q3,q3,q3
 mov r1,r7
 mov r2,r10
 lsrs r11,r5,#2
 cmp r11,#0
 beq 3f
2:
 ldmia r2!,{r0,r3,r12,lr}
 SG_TAP r0
 SG_TAP r3
 SG_TAP r12
 SG_TAP lr
 subs r11,#1
 bne 2b
3:
 and r11,r5,#3
 cmp r11,#0
 beq 5f
4:
 ldr r0,[r2],#4
 SG_TAP r0
 subs r11,#1
 bne 4b
5:
 vstmia r8!,{s0-s15}
 add r7,#64
 subs r4,#1
 bne 1b
9:
 vpop {d8-d15}
 pop {r4-r11,pc}
.size bwe_sgemv16,.-bwe_sgemv16
.section .note.GNU-stack,""
