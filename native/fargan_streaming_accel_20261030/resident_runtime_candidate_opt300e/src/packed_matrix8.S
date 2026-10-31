.syntax unified
.thumb
.text
.align 2
.global packed_matrix8
.type packed_matrix8,%function
.thumb_func
/* context32: x0,w4,scales8,bias12,y16,blocks20,tiles24,scale28,accumulate32.
 * One AAPCS frame per matrix. Eight even GPR accumulators. q0-q3 are caller
 * saved, and no q4-q7 are touched. No memory accumulator or inner-loop spill.
 * Context is caller-local and scales/bias fields advance across full tiles.
 */
packed_matrix8:
 push {r4-r11,lr}
 vpush {d8-d11}
 mov r9,r0
 ldr r3,[r9,#4]
 ldr r11,[r9,#24]
 ldr r0,[r9,#28]
 vdup.32 q4,r0
1:
 ldr r1,[r9,#0]
 ldr r7,[r9,#20]
 add r5,r3,r7,lsl #6
 lsrs r7,r7,#2
 movs r0,#0
 movs r2,#0
 movs r4,#0
 movs r6,#0
 mov r8,r0
 mov r10,r0
 mov r12,r0
 mov lr,r0
 cmp r7,#0
 beq.w 3f
2:
 .rept 4
 vldrb.8 q0,[r1],#16
 vldrb.8 q1,[r3],#16
 vmladava.s8 r0,q0,q1
 vldrb.8 q1,[r3],#16
 vmladava.s8 r2,q0,q1
 vldrb.8 q1,[r3],#16
 vmladava.s8 r4,q0,q1
 vldrb.8 q1,[r3],#16
 vmladava.s8 r6,q0,q1
 vldrb.8 q1,[r5],#16
 vmladava.s8 r8,q0,q1
 vldrb.8 q1,[r5],#16
 vmladava.s8 r10,q0,q1
 vldrb.8 q1,[r5],#16
 vmladava.s8 r12,q0,q1
 vldrb.8 q1,[r5],#16
 vmladava.s8 lr,q0,q1
 .endr
 subs r7,#1
 bne.w 2b
3:
 ldr r7,[r9,#20]
 and r7,r7,#3
 cbz r7,5f
4:
 vldrb.8 q0,[r1],#16
 vldrb.8 q1,[r3],#16
 vmladava.s8 r0,q0,q1
 vldrb.8 q1,[r3],#16
 vmladava.s8 r2,q0,q1
 vldrb.8 q1,[r3],#16
 vmladava.s8 r4,q0,q1
 vldrb.8 q1,[r3],#16
 vmladava.s8 r6,q0,q1
 vldrb.8 q1,[r5],#16
 vmladava.s8 r8,q0,q1
 vldrb.8 q1,[r5],#16
 vmladava.s8 r10,q0,q1
 vldrb.8 q1,[r5],#16
 vmladava.s8 r12,q0,q1
 vldrb.8 q1,[r5],#16
 vmladava.s8 lr,q0,q1
 subs r7,#1
 bne 4b
5:
 mov r3,r5
 ldr r7,[r9,#16]
 /* Direct scalar accumulators -> two vectors, with no intermediate dots[]. */
 vmov d0,r0,r2
 vmov d1,r4,r6
 vmov d2,r8,r10
 vmov d3,r12,lr
 vcvt.f32.s32 q0,q0
 vcvt.f32.s32 q1,q1
 vmul.f32 q0,q0,q4
 vmul.f32 q1,q1,q4
 ldr r0,[r9,#8]
 vldrw.u32 q2,[r0],#16
 vmul.f32 q0,q0,q2
 vldrw.u32 q2,[r0],#16
 vmul.f32 q1,q1,q2
 str r0,[r9,#8]
 ldr r0,[r9,#12]
 cbz r0,6f
 vldrw.u32 q2,[r0],#16
 vldrw.u32 q3,[r0],#16
 str r0,[r9,#12]
 b 7f
6:
 veor q2,q2,q2
 veor q3,q3,q3
7:
 vadd.f32 q0,q0,q2
 vadd.f32 q1,q1,q3
 ldr r0,[r9,#32]
 cbz r0,9f
 vldrw.u32 q2,[r7]
 add r0,r7,#16
 vldrw.u32 q3,[r0]
 vadd.f32 q0,q0,q2
 vadd.f32 q1,q1,q3
9:
 vstrw.u32 q0,[r7],#16
 vstrw.u32 q1,[r7],#16
 str r7,[r9,#16]
 subs r11,#1
 bne.w 1b
 vpop {d8-d11}
 pop {r4-r11,pc}
.size packed_matrix8,.-packed_matrix8
.section .note.GNU-stack,""
