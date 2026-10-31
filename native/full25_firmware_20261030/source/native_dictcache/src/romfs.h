#ifndef N25_ROMFS_H
#define N25_ROMFS_H
#include <stddef.h>
#include <stdint.h>
typedef struct {
    const char *name;
    unsigned pack, part, offset, bytes;
    const unsigned char *direct;
} N25RomFile;
extern const N25RomFile n25_rom_files[];
extern const unsigned n25_rom_file_count;
int n25_romfs_initialize(void);
size_t n25_romfs_heap_peak(void);
size_t n25_romfs_decode_workspace_peak(void);
unsigned n25_romfs_decode_count(void);
int n25_romfs_set_workspace(void *buffer,size_t capacity);
#endif

int n25_romfs_prewarm_files(void);
const void*n25_romfs_direct_file(const char*,size_t*);
