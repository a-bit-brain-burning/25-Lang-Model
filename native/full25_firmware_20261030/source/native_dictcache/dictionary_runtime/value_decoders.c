#include "value_decoders.h"
#include "predict_reader.h"
#include <string.h>
static int vint(const unsigned char *r,size_t n,size_t *p,uint32_t *value){
    unsigned int shift;uint32_t result=0;
    for(shift=0;shift<35;shift+=7){unsigned char b;if(*p>=n)return 0;b=r[(*p)++];if(shift==28&&(b&0xf0))return 0;result|=(uint32_t)(b&127)<<shift;if(!(b&128)){*value=result;return 1;}}
    return 0;
}
static size_t put(unsigned char *out,uint32_t value){size_t n=0;do{out[n++]=(unsigned char)((value&127)|(value>=128?128:0));value>>=7;}while(value);return n;}
static uint32_t rd32(const unsigned char *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
int aux_expand(const void *pool,size_t pool_bytes,const void *record,size_t rn,
    void *output,size_t cap,size_t *used){
    const unsigned char *p=pool,*r=record;unsigned char *out=output;size_t end=0,pos;uint32_t count,data,identifier,a,b;
    if(used)*used=0;
    if(!p || pool_bytes<24 || !r || !out || !used || memcmp(p,"AUXP1\0\0\0",8))return -1;
    count=rd32(p+8);data=rd32(p+12);
    if(data>pool_bytes || count>(data-20)/4 || data!=20+4*((uint64_t)count+1) || rd32(p+16)!=pool_bytes)return -1;
    while(end<rn && r[end]!=255)end++;
    if(end==rn)return -1;
    pos=end+1;if(!vint(r,rn,&pos,&identifier) || pos!=rn || identifier>=count)return -1;
    a=rd32(p+20+(size_t)identifier*4);b=rd32(p+20+(size_t)(identifier+1)*4);
    if(b<a || b>pool_bytes-data)return -1;
    if(end+1>cap || b-a>cap-end-1)return -2;
    memcpy(out,r,end+1);memcpy(out+end+1,p+data+a,b-a);*used=end+1+b-a;return 0;
}
int en_apply(const void *prediction,size_t pn,const void *record,size_t rn,
    void *temporary,size_t tn,void *output,size_t cap,size_t *used){
    const unsigned char *r=record;unsigned char *temp=temporary,*out=output;size_t n,phones=0,stress;int status;
    if(used)*used=0;
    if(!r || !rn || !temporary || !out || !used)return -1;
    if(r[0]==1){if(rn>cap)return -2;memcpy(out,r,rn);*used=rn;return 0;}
    if(r[0]!=0)return -1;
    status=spr_apply(prediction,pn,r+1,rn-1,temp,tn,&n);if(status)return status;
    while(phones<n && temp[phones]!=255)phones++;
    if(phones==n || phones>255)return -1;
    stress=n-phones-1;if(stress!=(phones+3)/4)return -1;
    if(n+1>cap)return -2;
    out[0]=0;out[1]=(unsigned char)phones;memcpy(out+2,temp,phones);memcpy(out+2+phones,temp+phones+1,stress);*used=n+1;return 0;
}
int primary_id(const void *record,size_t rn,const void *translation,size_t tn,uint16_t *result){
    const unsigned char *r=record,*t=translation;size_t pos=0;uint32_t count,id;
    if(!r || !t || !result || !vint(r,rn,&pos,&count) || !count || !vint(r,rn,&pos,&id) || tn<2 || id>(tn-2)/2)return -1;
    *result=(uint16_t)((uint16_t)t[2*id]|(uint16_t)t[2*id+1]<<8);return 0;
}
int phrase_apply(const uint16_t *defaults,size_t count,const void *record,size_t rn,
    void *output,size_t cap,size_t *used){
    const unsigned char *r=record;unsigned char *out=output;size_t pos=1,total=0,i;uint32_t mask;
    if(used)*used=0;
    if(!r || !rn || !out || !used || !defaults)return -1;
    if(r[0]==1){if(rn-1>cap)return -2;memcpy(out,r+1,rn-1);*used=rn-1;return 0;}
    if(r[0]!=0 || count>32 || !vint(r,rn,&pos,&mask) || (count<32 && (mask>>count)))return -1;
    for(i=0;i<count;i++){
        if(mask&((uint32_t)1<<i)){
            size_t start=pos,j;uint32_t readings,id;
            if(!vint(r,rn,&pos,&readings) || readings>rn-pos)return -1;
            for(j=0;j<readings;j++)if(!vint(r,rn,&pos,&id))return -1;
            if(pos-start>cap-total)return -2;
            memcpy(out+total,r+start,pos-start);total+=pos-start;
        }else{
            unsigned char bytes[5];size_t n;if(defaults[i]==65535)return -1;
            n=put(bytes,defaults[i]);if(1+n>cap-total)return -2;
            out[total++]=1;memcpy(out+total,bytes,n);total+=n;
        }
    }
    if(pos!=rn)return -1;
    *used=total;return 0;
}
