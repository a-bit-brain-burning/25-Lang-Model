#ifndef LEXPACK_H
#define LEXPACK_H
#include <stddef.h>
#include <stdint.h>
/* ROM pointers must remain valid. Reader never allocates heap memory.
 * Raw workspace >= max_raw; key workspace >= max_key; inflate arena >= 65536.
 * Values point into caller raw workspace, valid until the next decompression.
 * Source SHA is a provenance identifier; authenticate the package on install.
 */
typedef struct {
    const unsigned char *rom;
    uint32_t size, sections, section_offset, blocks, block_offset;
    uint32_t key_offset, data_offset, max_raw, max_key;
} lex_view;
enum {LEX_OK=0,LEX_NOT_FOUND=1,LEX_INVALID=-1,LEX_CAPACITY=-2,LEX_CORRUPT=-3};
int lex_open(lex_view *view,const void *rom,size_t bytes);
int lex_section(const lex_view *view,const char *name,size_t name_bytes,uint32_t *section);
int lex_unpack(const lex_view *view,uint32_t block,void *raw,size_t raw_capacity,void *arena,size_t arena_capacity,size_t *raw_bytes);
int lex_lookup(const lex_view *view,uint32_t section,const void *key,size_t key_bytes,
               void *raw,size_t raw_capacity,void *key_buffer,size_t key_capacity,
               void *arena,size_t arena_capacity,const unsigned char **value,size_t *value_bytes);
#endif
