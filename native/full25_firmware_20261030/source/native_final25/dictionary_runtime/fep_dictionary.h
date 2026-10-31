#ifndef N25_FEP_DICTIONARY_H
#define N25_FEP_DICTIONARY_H
#include <stddef.h>
#include <stdint.h>
typedef struct {
 unsigned pack,blocks;
 const char*const*first_keys;
 const uint32_t*alphabet;
 unsigned alphabet_count;
}FepLexiconDesc;
extern const FepLexiconDesc fep_lexicons[];
/* 0 exact record, -5 absent, other negatives malformed input or capacity. */
int fep_lexicon_record(const FepLexiconDesc*,const char*,unsigned char*,size_t,size_t*,void*,size_t);
int fep_lexicon_prewarm_es(void);
const void*fep_lexicon_cached_section(unsigned,const char*,size_t*);
#endif
