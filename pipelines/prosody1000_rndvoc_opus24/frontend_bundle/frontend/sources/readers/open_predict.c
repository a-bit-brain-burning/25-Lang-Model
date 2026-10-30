/* EXPERIMENTAL open-vocabulary use of the existing bounded spelling predictor.
 * The older finite-dictionary contract remains unchanged in its own package.
 * This wrapper explicitly opts into changed OOV semantics and rejects words
 * outside a stored lower-case alphabet or the 3..64 Unicode-scalar limit.
 * No heap allocation, Unicode locale tables or unbounded buffers are used.
 */
#include <stddef.h>
#include <stdint.h>
#include "predict_reader.h"

int open_word_allowed(const void *word,size_t bytes,const uint32_t *alphabet,size_t count) {
    const unsigned char *s=(const unsigned char *)word;
    size_t i=0,n=0;
    if(!s||!alphabet||!count||bytes>256)return 0;
    while(i<bytes) {
        uint32_t cp=s[i++],minimum=0;unsigned extra=0;
        if(cp<0x80) { if(!cp)return 0; }
        else if(cp>=0xc2&&cp<=0xdf) {cp&=31;extra=1;minimum=0x80;}
        else if(cp>=0xe0&&cp<=0xef) {cp&=15;extra=2;minimum=0x800;}
        else if(cp>=0xf0&&cp<=0xf4) {cp&=7;extra=3;minimum=0x10000;}
        else return 0;
        if(extra>bytes-i)return 0;
        for(unsigned j=0;j<extra;j++){unsigned c=s[i++];if((c&0xc0)!=0x80)return 0;cp=(cp<<6)|(c&63);}
        if(cp<minimum||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff))return 0;
        size_t lo=0,hi=count;
        while(lo<hi){size_t mid=lo+(hi-lo)/2;if(alphabet[mid]<cp)lo=mid+1;else hi=mid;}
        if(lo==count||alphabet[lo]!=cp||++n>64)return 0;
    }
    return n>=3;
}

int open_predict(const spr_view *model,const void *word,size_t bytes,
                 const uint32_t *alphabet,size_t count,
                 const void *record,size_t record_bytes,
                 void *prediction,size_t prediction_capacity,
                 void *output,size_t output_capacity,size_t *used) {
    size_t n=0;
    if(!used)return SPR_INVALID;
    *used=0;
    if(!open_word_allowed(word,bytes,alphabet,count))return -3;
    if(!record||record_bytes<2||((const unsigned char *)record)[0]||((const unsigned char *)record)[1])return SPR_INVALID;
    int status=spr_predict(model,word,bytes,prediction,prediction_capacity,&n);
    if(status)return status;
    if(!n)return SPR_INVALID;
    return spr_apply(prediction,n,record,record_bytes,output,output_capacity,used);
}
