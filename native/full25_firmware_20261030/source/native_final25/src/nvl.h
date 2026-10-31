#ifndef N25_NVL_H
#define N25_NVL_H
#include <stddef.h>
#include <stdint.h>
/* Paged exact word -> original residual record table. No vocabulary truncation. */
typedef struct {
    const unsigned char *data,*model;
    size_t bytes,model_bytes;
    unsigned pages,words;
} NvlPack;
typedef struct {
    unsigned char *page;
    size_t capacity,used;
    const unsigned char *owner;
    unsigned index;
} NvlCache;
int nvl_open(NvlPack *pack,const void *rom,size_t bytes);
int nvl_lookup(const NvlPack *pack,NvlCache *cache,const char *word,
    unsigned char *record,size_t capacity,size_t *written);
int nvl_page(const NvlPack *pack,NvlCache *cache,unsigned page);
#endif
