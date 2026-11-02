#ifndef N25_DICT_CACHE_H
#define N25_DICT_CACHE_H
#include <stddef.h>
/* Request-scoped decoded FEP lexicon working set in SRAM.
 * Avoids the repeated whole-section LZMA decode that dominates EN/FR/IT/LT/PT/RO. */
int n25_dict_cache_query(unsigned lexicon,const char*word,unsigned char*out,size_t cap,size_t*written);
void n25_dict_cache_reset(void);
#endif
