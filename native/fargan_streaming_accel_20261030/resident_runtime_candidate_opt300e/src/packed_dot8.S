.syntax unified
.thumb
.text
.align 2
.global packed_dot8
.type packed_dot8,%function
.thumb_func
/* r0=x padded to blocks*16, r1=packed weights, r2=blocks, r3=8 I32 outputs.
 * AAPCS: r4-r8,r10 and LR saved once. q0/q1 and r0-r3,r12 are caller saved.
 * Eight scalar accumulators, one x pointer and one w pointer. No loop spills.
 */
packed_dot8:
 push {r4-r11,lr}
 mov r11,r3
 mov r3,r1
 mov r1,r0
 add r5,r3,r2,lsl #6
 and r9,r2,#3
 lsrs r7,r2,#2
 movs r0,#0
 movs r2,#0
 movs r4,#0
 movs r6,#0
 mov r8,r0
 mov r10,r0
 mov r12,r0
 mov lr,r0
 cmp r7,#0
 beq.w 2f
1:
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
 bne 1b
2:
 mov r7,r9
 cbz r7,4f
3:
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
 bne 3b
4:
 stmia r11!,{r0,r2,r4,r6,r8,r10,r12,lr}
 pop {r4-r11,pc}
.size packed_dot8,.-packed_dot8
.section .note.GNU-stack,""
