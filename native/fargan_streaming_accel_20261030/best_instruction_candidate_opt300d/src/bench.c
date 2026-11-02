#define TTS_BENCH_IMPLEMENTATION 1
#include "bench.h"
extern unsigned char __code_start[], __image_end[], __data_start[], __bss_end[];
extern unsigned char __assets_start[], __assets_end[], __arena_start[], __arena_end[];
extern unsigned char __stack_bottom[], __stack_top[];
static size_t arena_used, arena_peak, allocation_failures;

static intptr_t semi(uint32_t operation, const void *arguments) {
    register uintptr_t r0 __asm__("r0") = operation;
    register const void *r1 __asm__("r1") = arguments;
    __asm__ volatile("bkpt #0xab" : "+r"(r0), "+r"(r1) : : "memory");
    return (intptr_t)r0;
}
void bench_puts(const char *text) { (void)semi(4, text); }
_Noreturn void bench_exit(unsigned status) {
    uint32_t args[2] = {0x20026, status};
    (void)semi(0x20, args);
    for (;;) {}
}
uint32_t bench_tick_frequency(void) { return (uint32_t)semi(0x31, 0); }
uint64_t bench_now(void) {
    uint32_t ticks[2] = {0, 0};
    if (semi(0x30, ticks) != 0) {
        bench_puts("M55_TIMER_UNAVAILABLE\n"); bench_exit(91);
    }
    return ((uint64_t)ticks[1] << 32) | ticks[0];
}
void bench_metric(const char *name, uint64_t value) {
    char line[192], digits[24]; size_t pos = 0, count = 0;
    const char *prefix = "{\"kind\":\"metric\",\"name\":\"";
    while (*prefix) line[pos++] = *prefix++;
    while (*name && pos < 130) {
        char c = *name++;
        /* Metric names are identifiers, not arbitrary JSON strings. */
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_')) bench_exit(92);
        line[pos++] = c;
    }
    if (*name) bench_exit(92);
    prefix = "\",\"value\":";
    while (*prefix) line[pos++] = *prefix++;
    do { digits[count++] = (char)('0' + value % 10); value /= 10; } while (value);
    while (count) line[pos++] = digits[--count];
    line[pos++] = '}'; line[pos++] = '\n'; line[pos] = 0;
    bench_puts(line);
}
size_t bench_arena_capacity(void) { return (size_t)(__arena_end - __arena_start); }
void *bench_alloc(size_t bytes, size_t alignment) {
    size_t capacity = bench_arena_capacity();
    if (!alignment || (alignment & (alignment-1)) || alignment > 4096 ||
        bytes > capacity || arena_used > SIZE_MAX-(alignment-1)) {
        allocation_failures++; return 0;
    }
    uintptr_t current = (uintptr_t)__arena_start + arena_used;
    size_t padding = (alignment - (current & (alignment-1))) & (alignment-1);
    if (padding > capacity-arena_used || bytes > capacity-arena_used-padding) {
        allocation_failures++; return 0;
    }
    size_t start = arena_used+padding;
    arena_used = start+bytes;
    if (arena_used > arena_peak) arena_peak = arena_used;
    return __arena_start+start;
}
size_t bench_arena_mark(void) { return arena_used; }
int bench_arena_rewind(size_t mark) {
    if (mark > arena_used) return -1;
    arena_used = mark; return 0;
}
size_t bench_arena_peak(void) { return arena_peak; }
size_t bench_stack_written_high_water(void) {
    const volatile uint32_t *p = (const volatile uint32_t *)__stack_bottom;
    const volatile uint32_t *top = (const volatile uint32_t *)__stack_top;
    while (p < top && *p == 0xa5a5a5a5u) p++;
    return (size_t)((const unsigned char *)top - (const unsigned char *)p);
}
void bench_memory_report(void) {}
int bench_write_file(const char *name, const void *data, size_t bytes) {
    size_t length = 0;
    while (name[length]) length++;
    uintptr_t open_args[3] = {(uintptr_t)name, 5, length}; /* binary write */
    intptr_t handle = semi(1, open_args);
    if (handle < 0) return -1;
    uintptr_t write_args[3] = {(uintptr_t)handle, (uintptr_t)data, bytes};
    intptr_t unwritten = semi(5, write_args);
    uintptr_t close_args[1] = {(uintptr_t)handle};
    intptr_t closed = semi(2, close_args);
    return unwritten == 0 && closed == 0 ? 0 : -1;
}
