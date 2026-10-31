#ifndef N25_SPANISH_RECORD_H
#define N25_SPANISH_RECORD_H
#include "predict_reader.h"
typedef struct {spr_view model;void*workspace;size_t workspace_bytes;}SpanishContext;
int spanish_context_init(SpanishContext*,void*,size_t,void*,size_t);
int spanish_word(SpanishContext*,const char*,unsigned char*,signed char*,unsigned,unsigned*);
#endif
