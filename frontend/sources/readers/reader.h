#ifndef FRONTEND_FEP_READER_H
#define FRONTEND_FEP_READER_H
#include <stdint.h>
#include <stddef.h>
/* Output, allocator workspace and ROM must not overlap. The caller supplies
 * their actual capacities. Errors are negative; no allocation escapes a call. */
int fep_read(const unsigned char *rom,size_t size,const char *name,
             unsigned char *out,size_t cap,void *workspace,size_t workcap,
             size_t *allocator_used,size_t *written);
/* A query is a NUL-terminated UTF-8 word without embedded NULs. Exact row index
 * is returned, or -5 for absence. query == NULL enumerates NUL-separated keys. */
int fep_keys(const unsigned char *raw,size_t size,const uint32_t *alphabet,
             size_t alphabet_count,const char *query,unsigned char *output,
             size_t capacity,size_t *written);
int fep_id(const unsigned char *raw,size_t size,uint32_t row,uint32_t *value_id);
/* Returned value is a view into the caller's decoded pool, not an allocation. */
int fep_pool(const unsigned char *raw,size_t size,uint32_t index,
             const unsigned char **value,size_t *length);
#endif
