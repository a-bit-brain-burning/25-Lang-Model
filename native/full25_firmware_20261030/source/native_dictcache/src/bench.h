#ifndef M55_BENCH_H
#define M55_BENCH_H
#include <stdint.h>
#include <stddef.h>
void bench_puts(const char *text);
void bench_metric(const char *name, uint64_t value);
_Noreturn void bench_exit(unsigned status);
uint64_t bench_now(void); /* QEMU 8.2.2 semihost host elapsed ticks, not cycles. */
uint32_t bench_tick_frequency(void);
void *bench_alloc(size_t bytes, size_t alignment);
size_t bench_arena_mark(void);
int bench_arena_rewind(size_t mark);
size_t bench_arena_peak(void);
size_t bench_arena_capacity(void);
size_t bench_stack_written_high_water(void);
void bench_memory_report(void);
int bench_write_file(const char *name, const void *data, size_t bytes);
int bench_append_file(const char *name, const void *data, size_t bytes);
#endif
