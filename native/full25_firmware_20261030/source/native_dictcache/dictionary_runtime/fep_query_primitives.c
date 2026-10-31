/* Bounded FEPACK1 section decoder and exact lookup primitives.
 * Host tests use liblzma. This source does not assert an ARM link/code budget.
 * Caller owns the ROM, output buffers and allocator arena. No system malloc.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include "reader.h"
static uint32_t u32(const unsigned char *p) {return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static int span(size_t start,size_t n,size_t cap) {return start<=cap && n<=cap-start;}
static int vi(const unsigned char **p,const unsigned char *end,uint32_t *out) {
    uint32_t v=0;
    for(unsigned s=0;s<=28;s+=7) {
        if(*p>=end)return -1;unsigned c=*(*p)++;
        if(s==28 && (c&0xf0))return -1;v|=(uint32_t)(c&127)<<s;
        if(!(c&128)){*out=v;return 0;}
    }return -1;
}
static int utf8(const unsigned char **p,const unsigned char *end,uint32_t *out) {
    if(*p>=end)return -1;unsigned c=*(*p)++;unsigned n;uint32_t v,min;
    if(c<128){if(!c)return -1;*out=c;return 0;}
    if(c>=194&&c<224){n=1;v=c&31;min=128;}
    else if(c>=224&&c<240){n=2;v=c&15;min=2048;}
    else if(c>=240&&c<245){n=3;v=c&7;min=65536;}
    else return -1;
    for(unsigned i=0;i<n;i++){if(*p>=end)return -1;c=*(*p)++;if((c&192)!=128)return -1;v=v<<6|(c&63);}
    if(v<min||v>0x10ffff||(v>=0xd800&&v<=0xdfff))return -1;*out=v;return 0;
}
static int emit(uint32_t cp,unsigned char *out,size_t *n,size_t cap) {
    unsigned count=cp<128?1:cp<2048?2:cp<65536?3:4;
    if(!cp||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff)||count>cap-*n)return -1;
    if(count==1)out[(*n)++]=(unsigned char)cp;
    else {
        out[(*n)++]=(unsigned char)((count==2?192:count==3?224:240)|(cp>>(6*(count-1))));
        for(unsigned i=count-1;i;i--)out[(*n)++]=(unsigned char)(128|((cp>>(6*(i-1)))&63));
    }return 0;
}

/* query != NULL: exact row index, -5 absent. Otherwise NUL-separated enumeration.
 * The returned row index is within this block, not a hash or probabilistic hit.
 */
typedef struct {uint32_t p,s,f,t,key;}FepCheckpoint;
static int fep_keys_worker(const unsigned char *raw,size_t size,const uint32_t *alphabet,size_t an,const char *query,
             unsigned char *output,size_t cap,size_t *written,const FepKeyIndex*seek,FepKeyIndex*build) {
    if(!raw||!written||size<32||memcmp(raw,"FEKEY1\0\0",8))return -1;*written=0;
    uint32_t mode=u32(raw+8),n=u32(raw+12),alpha=u32(raw+16),pn=u32(raw+20),sn=u32(raw+24),fn=u32(raw+28);
    if(mode>2||n>65536||(mode==2?(alpha!=an||!alphabet):alpha!=0))return -1;
    if(!span(32,pn,size)||!span(32+(size_t)pn,sn,size)||!span(32+(size_t)pn+sn,fn,size))return -1;
    const unsigned char *p=raw+32,*pe=p+pn,*s=pe,*se=s+sn,*f=se,*fe=f+fn,*t=fe,*end=raw+size;
    unsigned char prev[1024]={0},key[1024];uint32_t prev_ids[256];size_t prev_n=0,prev_c=0,written_n=0;
    int found=-5;uint32_t start=0;
    if(build){
        unsigned groups=(n+63)/64;
        if(!groups||groups>build->capacity/sizeof(FepCheckpoint))return -2;
        build->groups=groups;build->used=(size_t)groups*sizeof(FepCheckpoint);
    }
    if(seek){
        if(!query||seek->raw!=raw||seek->bytes!=size||!seek->groups)return -1;
        const FepCheckpoint*entries=seek->data;unsigned lo=0,hi=seek->groups;
        while(lo<hi){unsigned m=lo+(hi-lo)/2;const char*name=(const char*)seek->data+entries[m].key;
            if(strcmp(name,query)<=0)lo=m+1;else hi=m;}
        if(!lo)return -5;unsigned group=lo-1;const FepCheckpoint*e=entries+group;
        const char*previous=(const char*)seek->data+e->key;
        if(!strcmp(previous,query))return (int)(group*64);
        prev_n=strlen(previous);if(prev_n>=sizeof(prev))return -1;memcpy(prev,previous,prev_n+1);
        p=raw+e->p;s=raw+e->s;f=raw+e->f;t=raw+e->t;start=group*64+1;
        if(p<raw+32||p>pe||s<pe||s>se||f<se||f>fe||t<fe||t>end)return -1;
        if(mode==2){const unsigned char*q=prev;uint32_t cp;
            while(q<prev+prev_n){if(prev_c==256||utf8(&q,prev+prev_n,&cp))return -1;
                size_t a=0,b=an;while(a<b){size_t m=a+(b-a)/2;if(alphabet[m]<cp)a=m+1;else b=m;}
                if(a==an||alphabet[a]!=cp)return -1;prev_ids[prev_c++]=(uint32_t)a;
            }
        }
    }
    for(uint32_t i=start;i<n;i++) {
        uint32_t prefix,suffix;size_t key_n=0;
        if(vi(&p,pe,&prefix)||vi(&s,se,&suffix)||!suffix)return -1;
        if(mode==2) {
            uint32_t delta;
            if(prefix>prev_c||suffix>256-prefix||vi(&f,fe,&delta))return -1;
            uint32_t base=prefix<prev_c?prev_ids[prefix]+1:0;
            if(delta>UINT32_MAX-base)return -1;prev_ids[prefix]=base+delta;
            for(uint32_t j=1;j<suffix;j++)if(vi(&t,end,&prev_ids[prefix+j]))return -1;
            prev_c=prefix+suffix;
            for(size_t j=0;j<prev_c;j++)if(prev_ids[j]>=an||emit(alphabet[prev_ids[j]],key,&key_n,sizeof(key)-1))return -1;
        } else {
            size_t prefix_bytes=prefix;
            if(mode==1) {
                const unsigned char *q=prev;uint32_t cp;
                for(uint32_t j=0;j<prefix;j++)if(utf8(&q,prev+prev_n,&cp))return -1;
                prefix_bytes=(size_t)(q-prev);
            }
            if(prefix_bytes>prev_n)return -1;
            memcpy(key,prev,prefix_bytes);key_n=prefix_bytes;
            const unsigned char *start=t;
            if(mode==1){uint32_t cp;for(uint32_t j=0;j<suffix;j++)if(utf8(&t,end,&cp))return -1;}
            else {if(suffix>(size_t)(end-t))return -1;t+=suffix;}
            size_t tail=(size_t)(t-start);
            if(tail>sizeof(key)-1-key_n)return -2;memcpy(key+key_n,start,tail);key_n+=tail;
            const unsigned char *q=key;uint32_t cp;while(q<key+key_n)if(utf8(&q,key+key_n,&cp))return -1;
        }
        key[key_n]=0;
        if(build&&i%64==0){
            if(key_n+1>build->capacity-build->used)return -2;
            FepCheckpoint*e=(FepCheckpoint*)build->data+i/64;
            *e=(FepCheckpoint){(uint32_t)(p-raw),(uint32_t)(s-raw),(uint32_t)(f-raw),(uint32_t)(t-raw),(uint32_t)build->used};
            memcpy((unsigned char*)build->data+build->used,key,key_n+1);build->used+=key_n+1;
        }
        if(i&&strcmp((char *)prev,(char *)key)>=0)return -1;
        if(query){int comparison=strcmp(query,(char*)key);
            if(!comparison){if(seek)return (int)i;found=(int)i;}
            else if(seek&&comparison<0)return -5;
        }
        if(!query&&!build){if(!output||key_n+1>cap-written_n)return -2;memcpy(output+written_n,key,key_n+1);written_n+=key_n+1;}
        memcpy(prev,key,key_n+1);prev_n=key_n;
    }
    if(p!=pe||s!=se||f!=fe||t!=end)return -1;
    *written=written_n;return query?found:0;
}

int fep_id(const unsigned char *raw,size_t size,uint32_t index,uint32_t *out) {
    if(!raw||!out||index>=65536)return -1;const unsigned char *p=raw,*end=raw+size;
    for(uint32_t i=0;i<=index;i++)if(vi(&p,end,out))return -1;return 0;
}
int fep_pool(const unsigned char *raw,size_t size,uint32_t index,const unsigned char **value,size_t *length) {
    if(!raw||!value||!length||size<12)return -1;
    uint32_t n=u32(raw),group=u32(raw+4),ng=u32(raw+8);
    if(group!=256||ng!=n/256u+(n%256u!=0)||ng>(size-12)/4||index>=n)return -1;
    size_t base=12+(size_t)ng*4;uint32_t g=index/256,a=u32(raw+12+4*g),b=g+1<ng?u32(raw+12+4*(g+1)):(uint32_t)(size-base);
    if(a>b||b>size-base)return -1;const unsigned char *p=raw+base+a,*end=raw+base+b;uint32_t len;
    for(uint32_t i=g*256;i<=index;i++){
        if(vi(&p,end,&len)||len>(size_t)(end-p))return -1;
        if(i==index){*value=p;*length=len;return 0;}p+=len;
    }return -1;
}

int fep_keys(const unsigned char*raw,size_t size,const uint32_t*alphabet,size_t an,const char*query,unsigned char*out,size_t cap,size_t*written){
 return fep_keys_worker(raw,size,alphabet,an,query,out,cap,written,NULL,NULL);
}
int fep_keys_index_build(FepKeyIndex*index,const unsigned char*raw,size_t size,const uint32_t*alphabet,size_t an,void*memory,size_t capacity){
 if(!index||!memory)return -1;*index=(FepKeyIndex){NULL,size,alphabet,an,memory,capacity,0,0};size_t written=0;
 int rc=fep_keys_worker(raw,size,alphabet,an,NULL,NULL,0,&written,NULL,index);
 if(!rc)index->raw=raw;return rc;
}
int fep_keys_index_query(const FepKeyIndex*index,const char*word){
 if(!index||!index->raw||!word)return -1;size_t written=0;
 return fep_keys_worker(index->raw,index->bytes,index->alphabet,index->alphabet_count,word,NULL,0,&written,index,NULL);
}

int fep_ids_index_build(const unsigned char*raw,size_t size,unsigned count,uint32_t*index,size_t capacity){
 if(!raw||!index||(count+63)/64>capacity)return -1;
 const unsigned char*p=raw,*end=raw+size;uint32_t value;
 for(unsigned i=0;i<count;i++){if(i%64==0)index[i/64]=(uint32_t)(p-raw);if(vi(&p,end,&value))return -1;}
 return p==end?0:-1;
}
int fep_id_index_query(const unsigned char*raw,size_t size,const uint32_t*index,unsigned count,unsigned row,uint32_t*out){
 if(!raw||!index||!out||row>=count||index[row/64]>size)return -1;
 const unsigned char*p=raw+index[row/64],*end=raw+size;
 for(unsigned i=row/64*64;i<=row;i++)if(vi(&p,end,out))return -1;
 return 0;
}
