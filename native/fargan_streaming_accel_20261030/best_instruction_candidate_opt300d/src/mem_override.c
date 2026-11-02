/* MVE byte-copy/zero overrides for newlib scalar loops.
 * Semantics are byte-identical to libc; only the instruction count per byte
 * changes. VLDRB.8/VSTRB.8 are byte-element accesses, and 32-bit tails use
 * explicitly-packed accesses (Cortex-M55 supports unaligned LDR/STR). */
#include <stddef.h>
#include <stdint.h>
#include <arm_mve.h>

typedef struct __attribute__((packed)) u32_packed { uint32_t v; } u32_packed;

void *memcpy(void *dst, const void *src, size_t n) {
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  size_t bulk = n & ~(size_t)63;
  while (bulk) {
    vstrbq_u8(d,      vldrbq_u8(s));
    vstrbq_u8(d + 16, vldrbq_u8(s + 16));
    vstrbq_u8(d + 32, vldrbq_u8(s + 32));
    vstrbq_u8(d + 48, vldrbq_u8(s + 48));
    d += 64; s += 64; bulk -= 64;
  }
  n &= 63;
  while (n >= 16) {
    vstrbq_u8(d, vldrbq_u8(s));
    d += 16; s += 16; n -= 16;
  }
  while (n >= 4) {
    ((u32_packed *)d)->v = ((const u32_packed *)s)->v;
    d += 4; s += 4; n -= 4;
  }
  while (n--) *d++ = *s++;
  return dst;
}

void *memset(void *dst, int value, size_t n) {
  unsigned char *d = (unsigned char *)dst;
  unsigned char v = (unsigned char)value;
  uint8x16_t q = vdupq_n_u8(v);
  size_t bulk = n & ~(size_t)63;
  while (bulk) {
    vstrbq_u8(d, q); vstrbq_u8(d + 16, q);
    vstrbq_u8(d + 32, q); vstrbq_u8(d + 48, q);
    d += 64; bulk -= 64;
  }
  n &= 63;
  while (n >= 16) { vstrbq_u8(d, q); d += 16; n -= 16; }
  uint32_t w = (uint32_t)v * 0x01010101u;
  while (n >= 4) { ((u32_packed *)d)->v = w; d += 4; n -= 4; }
  while (n--) *d++ = v;
  return dst;
}

void *memmove(void *dst, const void *src, size_t n) {
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  if (d == s || n == 0) return dst;
  if (d < s || d >= s + n) {
    if (n >= 64) {
      size_t bulk = n & ~(size_t)63;
      while (bulk) {
        vstrbq_u8(d,      vldrbq_u8(s));
        vstrbq_u8(d + 16, vldrbq_u8(s + 16));
        vstrbq_u8(d + 32, vldrbq_u8(s + 32));
        vstrbq_u8(d + 48, vldrbq_u8(s + 48));
        d += 64; s += 64; n -= 64; bulk -= 64;
      }
    }
    while (n >= 16) { vstrbq_u8(d, vldrbq_u8(s)); d += 16; s += 16; n -= 16; }
    while (n >= 4) { ((u32_packed *)d)->v = ((const u32_packed *)s)->v; d += 4; s += 4; n -= 4; }
    while (n--) *d++ = *s++;
  } else {
    size_t tail = n & 15;
    unsigned char *de = d + n;
    const unsigned char *se = s + n;
    size_t m = n - tail;
    while (m) { de -= 16; se -= 16; vstrbq_u8(de, vldrbq_u8(se)); m -= 16; }
    while (tail--) { de--; se--; *de = *se; }
  }
  return dst;
}
