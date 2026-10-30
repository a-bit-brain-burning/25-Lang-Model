#include "platform.h"
#include <stdint.h>
#ifdef NEW25_M55
void n25_puts(const char *text) {
    register unsigned r0 __asm__("r0")=4;
    register const char *r1 __asm__("r1")=text;
    __asm__ volatile("bkpt #0xab" : "+r"(r0),"+r"(r1) : : "memory");
}
_Noreturn void n25_exit(unsigned status) {
    unsigned words[2]={0x20026,status};
    register unsigned r0 __asm__("r0")=0x20;
    register const unsigned *r1 __asm__("r1")=words;
    __asm__ volatile("bkpt #0xab" : "+r"(r0),"+r"(r1) : : "memory");
    for(;;){}
}
#else
void n25_puts(const char *text) {
    volatile unsigned char *uart=(void *)0x10000000u;
    while(*text)*uart=(unsigned char)*text++;
}
_Noreturn void n25_exit(unsigned status) {
    *(volatile uint32_t *)0x00100000u=status?(status<<16)|0x3333u:0x5555u;
    for(;;){}
}
#endif
