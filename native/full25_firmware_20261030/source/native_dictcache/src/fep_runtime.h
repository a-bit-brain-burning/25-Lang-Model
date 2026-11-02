#ifndef N25_FEP_RUNTIME_H
#define N25_FEP_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
/* All storage is caller-owned. No filesystem, host callbacks or heap allocator. */
typedef struct {
    const unsigned char *data;
    size_t size;
    uint32_t sections;
    size_t payload;
} FepPack;
typedef struct {
    const unsigned char *packed;
    size_t packed_bytes, raw_bytes;
    uint32_t codec, crc32, window;
} FepSection;
enum { FEP_OK=0, FEP_INVALID=-1, FEP_CAPACITY=-2, FEP_DECODE=-3,
       FEP_CRC=-4, FEP_ABSENT=-5 };
int fep_open(FepPack *pack, const void *rom, size_t bytes);
int fep_find(const FepPack *pack, const char *name, FepSection *section);
int fep_decode(const FepSection *section, void *out, size_t output_capacity,
               void *workspace, size_t workspace_capacity, size_t *workspace_peak);
#endif
