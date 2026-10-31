.syntax unified
.thumb
.text
.align 2
.global bwe_adaconv16
.type bwe_adaconv16,%function
.thumb_func
/* x0,y4,out8,length12,prefix16,blocks16=20. All FP terms keep their order. */
.macro AD_TAP reg
 vldmia r2,{s16-s31}
 vmul.f32 q4,q4,\reg
 vmul.f32 q5,q5,\reg
 vmul.f32 q6,q6,\reg
 vmul.f32 q7,q7,\reg
 vadd.f32 q0,q0,q4
 vadd.f32 q1,q1,q5
 vadd.f32 q2,q2,q6
 vadd.f32 q3,q3,q7
 add r2,#4
.endm
bwe_adaconv16:
 push {r4-r11,lr}
 vpush {d8-d15}
 mov r9,r0
 ldr r10,[r9,#0]
 ldr r11,[r9,#4]
 ldr r8,[r9,#8]
 ldr r4,[r9,#12]
 ldr r6,[r9,#20]
 cmp r6,#0
 beq 9f
1:
 ldr r5,[r9,#16]
 veor q0,q0,q0
 veor q1,q1,q1
 veor q2,q2,q2
 veor q3,q3,q3
 mov r1,r10
 mov r2,r11
 lsrs r7,r5,#2
 cmp r7,#0
 beq 3f
2:
 ldmia r1!,{r0,r3,r12,lr}
 AD_TAP r0
 AD_TAP r3
 AD_TAP r12
 AD_TAP lr
 subs r7,#1
 bne 2b
3:
 and r7,r5,#3
 cbz r7,5f
4:
 ldr r0,[r1],#4
 AD_TAP r0
 subs r7,#1
 bne 4b
5:
 cmp r5,r4
 beq 8f
 /* Skip padded zero terms only for normal finite accumulators. */
 ldr r3,=0x7fffffff
 vdup.32 q7,r3
 vand q4,q0,q7
 vand q5,q1,q7
 vand q6,q2,q7
 vand q7,q3,q7
 ldr r3,=0x00800000
 vsub.i32 q4,q4,r3
 vsub.i32 q5,q5,r3
 vsub.i32 q6,q6,r3
 vsub.i32 q7,q7,r3
 vmax.u32 q4,q4,q5
 vmax.u32 q4,q4,q6
 vmax.u32 q4,q4,q7
 movs r0,#0
 vmaxv.u32 r0,q4
 ldr r3,=0x7effffff
 cmp r0,r3
 bls 8f
 sub r5,r4,r5
 lsrs r7,r5,#2
 cmp r7,#0
 beq 7f
6:
 ldmia r1!,{r0,r3,r12,lr}
 AD_TAP r0
 AD_TAP r3
 AD_TAP r12
 AD_TAP lr
 subs r7,#1
 bne 6b
7:
 and r7,r5,#3
 cbz r7,8f
10:
 ldr r0,[r1],#4
 AD_TAP r0
 subs r7,#1
 bne 10b
8:
 vstmia r8!,{s0-s15}
 add r11,#64
 subs r6,#1
 bne.w 1b
9:
 vpop {d8-d15}
 pop {r4-r11,pc}
.size bwe_adaconv16,.-bwe_adaconv16
.section .note.GNU-stack,""
