.syntax unified
.thumb
.text
.align 2
.global packed_dot4_pair
.type packed_dot4_pair,%function
.thumb_func
/* x0,x1,w,blocks ; stack: eight output dots. Layout4 x0 then4 x1.
 * w selects4 lanes in packed8 layout; stride128B per input16 block. */
packed_dot4_pair:
 push {r4-r10,lr}
 mov r7,r3
 mov r5,r2
 mov r3,r1
 mov r1,r0
 movs r0,#0
 movs r2,#0
 movs r4,#0
 movs r6,#0
 mov r8,r0
 mov r10,r0
 mov r12,r0
 mov lr,r0
 and r9,r7,#3
 lsrs r7,r7,#2
 cmp r7,#0
 beq 2f
1:
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
 bne 1b
2:
 mov r7,r9
 cbz r7,4f
3:
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
 bne 3b
4:
 ldr r9,[sp,#32]
 stmia r9!,{r0,r2,r4,r6,r8,r10,r12,lr}
 pop {r4-r10,pc}
.size packed_dot4_pair,.-packed_dot4_pair
.section .note.GNU-stack,""
