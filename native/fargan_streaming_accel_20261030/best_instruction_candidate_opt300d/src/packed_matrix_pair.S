.syntax unified
.thumb
.text
.align 2
.global packed_matrix_pair
.type packed_matrix_pair,%function
.thumb_func
/* x0,x1,w,scales,bias,y0,y1,blocks,tiles4,scale0,scale1,accumulate.
 * Eight even scalar accumulators: four outputs for each independent row.
 * The original packed8 layout is retained; alternate four-lane halves.
 */
packed_matrix_pair:
 push {r4-r11,lr}
 vpush {d8-d11}
 mov r9,r0
 ldr r5,[r9,#8]
 ldr r11,[r9,#32]
 ldr r0,[r9,#36]
 vdup.32 q4,r0
 ldr r0,[r9,#40]
 vdup.32 q5,r0
1:
 ldr r1,[r9,#0]
 ldr r3,[r9,#4]
 ldr r7,[r9,#28]
 lsrs r7,#2
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
 vldrb.8 q2,[r5],#16
 vmladava.s8 r0,q0,q2
 vmladava.s8 r8,q1,q2
 vldrb.8 q2,[r5],#16
 vmladava.s8 r2,q0,q2
 vmladava.s8 r10,q1,q2
 vldrb.8 q2,[r5],#16
 vmladava.s8 r4,q0,q2
 vmladava.s8 r12,q1,q2
 vldrb.8 q2,[r5],#16
 vmladava.s8 r6,q0,q2
 vmladava.s8 lr,q1,q2
 .endr
 subs r7,#1
 bne.w 2b
3:
 ldr r7,[r9,#28]
 and r7,r7,#3
 cbz r7,5f
4:
 vldrb.8 q0,[r1],#16
 vldrb.8 q1,[r3],#16
 vldrb.8 q2,[r5],#16
 vmladava.s8 r0,q0,q2
 vmladava.s8 r8,q1,q2
 vldrb.8 q2,[r5],#16
 vmladava.s8 r2,q0,q2
 vmladava.s8 r10,q1,q2
 vldrb.8 q2,[r5],#16
 vmladava.s8 r4,q0,q2
 vmladava.s8 r12,q1,q2
 vldrb.8 q2,[r5],#16
 vmladava.s8 r6,q0,q2
 vmladava.s8 lr,q1,q2
 subs r7,#1
 bne 4b
5:
 vmov d0,r0,r2
 vmov d1,r4,r6
 vmov d2,r8,r10
 vmov d3,r12,lr
 vcvt.f32.s32 q0,q0
 vcvt.f32.s32 q1,q1
 vmul.f32 q0,q0,q4
 vmul.f32 q1,q1,q5
 ldr r0,[r9,#12]
 vldrw.u32 q2,[r0],#16
 str r0,[r9,#12]
 vmul.f32 q0,q0,q2
 vmul.f32 q1,q1,q2
 ldr r0,[r9,#16]
 cbz r0,6f
 vldrw.u32 q2,[r0],#16
 str r0,[r9,#16]
 b 7f
6:
 veor q2,q2,q2
7:
 vadd.f32 q0,q0,q2
 vadd.f32 q1,q1,q2
 ldr r1,[r9,#20]
 ldr r3,[r9,#24]
 ldr r0,[r9,#44]
 cbz r0,9f
 vldrw.u32 q2,[r1]
 vldrw.u32 q3,[r3]
 vadd.f32 q0,q0,q2
 vadd.f32 q1,q1,q3
9:
 vstrw.u32 q0,[r1],#16
 vstrw.u32 q1,[r3],#16
 str r1,[r9,#20]
 str r3,[r9,#24]
 subs r11,#1
 bne.w 1b
 vpop {d8-d11}
 pop {r4-r11,pc}
.size packed_matrix_pair,.-packed_matrix_pair
.section .note.GNU-stack,""
