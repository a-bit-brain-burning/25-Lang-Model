#include "lookup_reader.h"
int spr_lookup(const lex_view *table,const spr_view *model,
    const void *word,size_t word_bytes,
    void *raw,size_t raw_capacity,void *key,size_t key_capacity,
    void *arena,size_t arena_capacity,void *prediction,size_t prediction_capacity,
    void *output,size_t output_capacity,size_t *used){
    uint32_t section,n=0;const unsigned char *value;size_t vn,pos=1,pn;unsigned int shift;int status;
    if(used)*used=0;
    if(!used)return LEX_INVALID;
    status=lex_section(table,"entries",7,&section);if(status)return status;
    status=lex_lookup(table,section,word,word_bytes,raw,raw_capacity,key,key_capacity,
        arena,arena_capacity,&value,&vn);if(status)return status;
    if(vn<2 || value[0]!=8)return LEX_CORRUPT;
    for(shift=0;shift<35;shift+=7){unsigned char b;
        if(pos>=vn)return LEX_CORRUPT;
        b=value[pos++];if(shift==28 && (b&0xf0))return LEX_CORRUPT;
        n|=(uint32_t)(b&127)<<shift;
        if(!(b&128))break;
    }
    if(shift==35 || n!=vn-pos)return LEX_CORRUPT;
    status=spr_predict(model,word,word_bytes,prediction,prediction_capacity,&pn);if(status)return status;
    return spr_apply(prediction,pn,value+pos,n,output,output_capacity,used);
}
