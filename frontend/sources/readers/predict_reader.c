#include "predict_reader.h"
#include <limits.h>
#include <string.h>
#define HEAD 28u
#define MAX_WORD 128u
static const unsigned char lefts[9]={0,0,1,1,0,2,1,2,2};
static const unsigned char rights[9]={0,1,0,1,2,0,2,1,2};
static uint32_t rd32(const unsigned char *p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static int compare(const unsigned char *a,size_t na,const unsigned char *b,size_t nb){size_t n=na<nb?na:nb;int c=memcmp(a,b,n);return c?c:(na>nb)-(na<nb);}
static const unsigned char *row(const spr_view *v,uint32_t i){return v->rom+v->data+rd32(v->rom+v->index+(size_t)i*4);}
int spr_open(spr_view *view,const void *rom,size_t bytes){
    spr_view v;const unsigned char *p=rom,*previous=NULL;size_t pn=0;uint32_t i,last=0;
    if(!view)return SPR_INVALID;
    memset(view,0,sizeof(*view));
    if(!p || bytes<HEAD || bytes>UINT32_MAX || memcmp(p,"SPRP1\0\0\0",8) || rd32(p+8)!=1)return SPR_INVALID;
    v.rom=p;v.count=rd32(p+12);v.index=rd32(p+16);v.data=rd32(p+20);v.size=rd32(p+24);
    if(v.size!=bytes || v.index!=HEAD || v.data<HEAD || v.data>bytes ||
       v.count>(UINT32_MAX-HEAD-4)/4 || v.data!=HEAD+4*(v.count+1))return SPR_INVALID;
    for(i=0;i<v.count;i++){
        uint32_t a=rd32(p+v.index+(size_t)i*4),b=rd32(p+v.index+(size_t)(i+1)*4);const unsigned char *r;size_t n,j;
        if(a!=last || b<a || b>bytes-v.data || b-a<3)return SPR_INVALID;
        r=p+v.data+a;n=r[0];
        if(!n || n>22 || n+2>b-a || r[1]>8 || r[n+1]>3 || n+2+r[n+1]!=b-a)return SPR_INVALID;
        for(j=n+2;j<b-a;j++)if(r[j]==255)return SPR_INVALID;
        if(previous && compare(previous,pn,r+1,n)>=0)return SPR_INVALID;
        previous=r+1;pn=n;last=b;
    }
    if(rd32(p+v.index+(size_t)v.count*4)!=bytes-v.data)return SPR_INVALID;
    *view=v;return SPR_OK;
}
static const unsigned char *find(const spr_view *v,const unsigned char *key,size_t n,size_t *used){
    uint32_t lo=0,hi=v->count;
    while(lo<hi){uint32_t m=lo+(hi-lo)/2;const unsigned char *r=row(v,m);int c=compare(r+1,r[0],key,n);
        if(c<0)lo=m+1;else hi=m;
    }
    if(lo<v->count){const unsigned char *r=row(v,lo);if(!compare(r+1,r[0],key,n)){*used=r[r[0]+1];return r+r[0]+2;}}
    return NULL;
}
/* Strict UTF-8: prevents scalar slicing inconsistencies with Python. */
static int scalar_offsets(const unsigned char *s,size_t n,uint32_t *at,size_t *count){
    size_t i=0,k=0;
    if(n>UINT32_MAX)return 0;
    while(i<n){unsigned char c=s[i];size_t z,j;uint32_t cp;
        if(k==MAX_WORD)return 0;
        at[k++]=(uint32_t)i;
        if(c<128){if(c==1 || c==2)return 0;i++;continue;}
        if(c>=0xc2 && c<=0xdf){z=2;cp=c&31;}
        else if(c>=0xe0 && c<=0xef){z=3;cp=c&15;}
        else if(c>=0xf0 && c<=0xf4){z=4;cp=c&7;}else return 0;
        if(z>n-i)return 0;
        for(j=1;j<z;j++){if((s[i+j]&0xc0)!=0x80)return 0;cp=(cp<<6)|(s[i+j]&63);}
        if((z==3 && cp<0x800) || (z==4 && cp<0x10000) || cp>0x10ffff || (cp>=0xd800 && cp<=0xdfff))return 0;
        i+=z;
    }
    at[k]=(uint32_t)n;*count=k;return 1;
}
int spr_predict(const spr_view *v,const void *word,size_t bytes,void *phones,size_t capacity,size_t *used){
    const unsigned char *s=word;unsigned char *out=phones;uint32_t at[MAX_WORD+1];size_t n,i,total=0;
    if(used)*used=0;
    if(!v || !v->rom || !word || !phones || !used || !scalar_offsets(s,bytes,at,&n))return SPR_INVALID;
    for(i=0;i<n;i++){
        const unsigned char *value=NULL;size_t vn=0;unsigned int shape;
        for(shape=0;shape<9;shape++){
            unsigned char key[22];size_t kn=1,got=0;int relative;
            key[0]=(unsigned char)shape;
            for(relative=-(int)lefts[shape];relative<=(int)rights[shape];relative++){
                int pos=(int)i+relative;
                if(pos<0)key[kn++]=1;
                else if((size_t)pos>=n)key[kn++]=2;
                else {size_t z=at[pos+1]-at[pos];memcpy(key+kn,s+at[pos],z);kn+=z;}
            }
            {const unsigned char *found=find(v,key,kn,&got);if(found){value=found;vn=got;}}
        }
        if(vn>capacity-total)return SPR_CAPACITY;
        if(vn)memcpy(out+total,value,vn);
        total+=vn;
    }
    *used=total;return SPR_OK;
}
int spr_apply(const void *prediction,size_t pn,const void *record,size_t rn,void *output,size_t cap,size_t *used){
    const unsigned char *pred=prediction,*r=record,*p;unsigned char *out=output;size_t n,i=0,j=0,total=0,aux;unsigned char mode;
    if(used)*used=0;
    if(!pred || !r || !out || !used || rn<2)return SPR_INVALID;
    mode=r[0];n=r[1];if(n>rn-2)return SPR_INVALID;p=r+2;aux=rn-2-n;
    if(mode==0){if(n)return SPR_INVALID;if(pn>cap)return SPR_CAPACITY;memcpy(out,pred,pn);total=pn;}
    else if(mode==1){if(n>cap)return SPR_CAPACITY;memcpy(out,p,n);total=n;}
    else if(mode==3){size_t a,b,m;if(n<2)return SPR_INVALID;a=p[0];b=p[1];m=n-2;
        if(a>pn || b>pn-a)return SPR_INVALID;
        if(a+b>cap || m>cap-a-b)return SPR_CAPACITY;
        memcpy(out,pred,a);memcpy(out+a,p+2,m);memcpy(out+a+m,pred+pn-b,b);total=a+b+m;
    }
    else if(mode==2){
        while(i<n){unsigned char code=p[i++];size_t run=(code>>2)+1;unsigned char op=code&3;
            if(op!=2 && run>pn-j)return SPR_INVALID;
            if(op!=1){if(run>cap-total)return SPR_CAPACITY;
                if(op==0)memcpy(out+total,pred+j,run);
                else {if(run>n-i)return SPR_INVALID;memcpy(out+total,p+i,run);i+=run;}
                total+=run;
            }
            if(op!=2)j+=run;
        }
        if(j!=pn)return SPR_INVALID;
    }else return SPR_INVALID;
    for(i=0;i<total;i++)if(out[i]==255)return SPR_INVALID;
    if(total==cap || aux>cap-total-1)return SPR_CAPACITY;
    out[total++]=255;memcpy(out+total,r+2+n,aux);total+=aux;*used=total;return SPR_OK;
}
