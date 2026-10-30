#ifndef PREDICT_READER_H
#define PREDICT_READER_H
#include <stddef.h>
#include <stdint.h>
/* Read-only direct ROM model. No heap, float or unbounded text buffering.
 * Call only after a successful lookup in the paired finite word table.
 * Maximum 128 Unicode scalars per key. No vocabulary expansion is provided. */
typedef struct {const unsigned char *rom;uint32_t size,count,index,data;} spr_view;
enum {SPR_OK=0,SPR_INVALID=-1,SPR_CAPACITY=-2};
int spr_open(spr_view *view,const void *rom,size_t bytes);
int spr_predict(const spr_view *view,const void *word,size_t bytes,
                void *phones,size_t capacity,size_t *used);
/* Output uses the prior full-record contract: phone IDs, 0xff, exact aux bytes. */
int spr_apply(const void *prediction,size_t prediction_bytes,const void *record,
              size_t record_bytes,void *output,size_t capacity,size_t *used);
#endif
