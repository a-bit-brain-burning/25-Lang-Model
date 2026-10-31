#ifndef N25_ROMANCE_RECORD_H
#define N25_ROMANCE_RECORD_H
#include <stdint.h>
#include <stddef.h>
#include "predict_reader.h"
typedef struct{const char*language;unsigned lexicon,model_bytes,aux_bytes,workspace_bytes,symbol_count;
 const char*const*symbols;const uint32_t*alphabet;unsigned alphabet_count;const unsigned char*default_record;unsigned default_bytes;}RomanceDictionary;
typedef struct{const RomanceDictionary*d; spr_view model;void*workspace;size_t workspace_bytes;const void*aux;}RomanceRecordContext;
extern const RomanceDictionary romance_dictionaries[];
extern const unsigned romance_dictionary_count;
int romance_record_init(RomanceRecordContext*,const RomanceDictionary*,void*,size_t,void*,void*);
int romance_record_word(RomanceRecordContext*,const char*,unsigned char*,signed char*,unsigned,unsigned*);
#endif
