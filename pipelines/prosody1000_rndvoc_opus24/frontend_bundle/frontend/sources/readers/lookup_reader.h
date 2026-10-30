#ifndef LOOKUP_READER_H
#define LOOKUP_READER_H
#include "predict_reader.h"
#include "table_reader.h"
/* Exact membership precedes spelling prediction. On unknown keys returns
 * LEX_NOT_FOUND, never a guessed pronunciation. All buffers are caller-owned.
 * Output is the original full binary record; auxiliary interpretation is a
 * separate consumer. The model/table pair must be authenticated by manifest. */
int spr_lookup(const lex_view *table,const spr_view *model,
    const void *word,size_t word_bytes,
    void *raw,size_t raw_capacity,void *key,size_t key_capacity,
    void *arena,size_t arena_capacity,void *prediction,size_t prediction_capacity,
    void *output,size_t output_capacity,size_t *used);
#endif
