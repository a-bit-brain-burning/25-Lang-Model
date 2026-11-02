#ifndef FARGAN_COMPRESSED_STORAGE_H
#define FARGAN_COMPRESSED_STORAGE_H
#include <stdint.h>
#include <stddef.h>
extern uint64_t storage_boot_ticks;
void storage_initialize(uint64_t boot_ticks);
void storage_sha256(const unsigned char*data,size_t size,unsigned char out[32]);
#endif
