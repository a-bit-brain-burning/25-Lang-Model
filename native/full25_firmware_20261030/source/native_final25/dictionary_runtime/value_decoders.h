#ifndef VALUE_DECODERS_H
#define VALUE_DECODERS_H
#include <stddef.h>
#include <stdint.h>
int en_apply(const void *prediction,size_t pn,const void *record,size_t rn,
    void *temporary,size_t tn,void *output,size_t cap,size_t *used);
int phrase_apply(const uint16_t *defaults,size_t count,const void *record,size_t rn,
    void *output,size_t cap,size_t *used);
int primary_id(const void *record,size_t rn,const void *translation,size_t tn,uint16_t *result);
int aux_expand(const void *pool,size_t pool_bytes,const void *record,size_t rn,
    void *output,size_t cap,size_t *used);
#endif
