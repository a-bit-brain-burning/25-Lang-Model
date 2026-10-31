#ifndef N25_PAGED_LEXICON_H
#define N25_PAGED_LEXICON_H
#include <stddef.h>
typedef struct{const char*first;unsigned offset,packed,raw;}N25LexiconPage;
typedef struct{const char*language;const unsigned char*data;unsigned bytes;const N25LexiconPage*pages;unsigned count;unsigned char sha256[32];}N25PagedLexicon;
extern const N25PagedLexicon n25_paged_lexicons[];
extern const unsigned n25_paged_lexicon_count;
int n25_paged_lexicon_initialize(void);
int n25_paged_lexicon_record(const char*,const char*,unsigned char*,size_t,size_t*);
#endif
