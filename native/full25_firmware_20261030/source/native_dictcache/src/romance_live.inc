/* Whole-utterance native G2P repair, preserving Romance 36/22 coordinates. */
#include "romance_constants.h"
static const RomanceConstant*romance_constant(const char*name){
 unsigned lo=0,hi=sizeof(romance_constants)/sizeof(romance_constants[0]);
 while(lo<hi){unsigned m=lo+(hi-lo)/2;if(strcmp(romance_constants[m].name,name)<0)lo=m+1;else hi=m;}
 return lo<sizeof(romance_constants)/sizeof(romance_constants[0])&&!strcmp(romance_constants[lo].name,name)?romance_constants+lo:NULL;
}
static void romance_features(const RomanceConstant*p,float*out){
 if(p->base!=65535){memcpy(out,n25_four_phones[p->base].features,32*4);memcpy(out+32,n25_four_phones[p->base].extra,4*4);}
 else memset(out,0,36*4);
 for(unsigned i=0;i<p->count;i++){const unsigned char*patch=romance_patches[p->patch+i];out[patch[0]]=romance_values[patch[1]];}
}
static int romance_nucleus(const RomanceConstant*p){
 float value=p->base==65535?0:n25_four_phones[p->base].features[31];
 for(unsigned i=0;i<p->count;i++)if(romance_patches[p->patch+i][0]==31)value=romance_values[romance_patches[p->patch+i][1]];
 return value>.5f;
}
static int romance_split(char*token,const RomanceConstant**phones,signed char*stresses,unsigned*count,unsigned capacity,int*pending){
 unsigned bytes=strlen(token);if(!bytes)return 0;
 const RomanceConstant*phone=romance_constant(token);
 if(phone){if(*count>=capacity)return -6;phones[*count]=phone;stresses[*count]=romance_nucleus(phone)?*pending:-1;
  if(romance_nucleus(phone))*pending=0;(*count)++;return 0;}
 for(unsigned cut=bytes;cut-->1;){
  if(((unsigned char)token[cut]&192)==128)continue;
  char saved=token[cut];token[cut]=0;phone=romance_constant(token);token[cut]=saved;if(!phone)continue;
  unsigned before=*count;int old=*pending;
  token[cut]=0;int rc=romance_split(token,phones,stresses,count,capacity,pending);token[cut]=saved;
  if(!rc)rc=romance_split(token+cut,phones,stresses,count,capacity,pending);
  if(!rc)return 0;*count=before;*pending=old;
 }return -4;
}
static int romance_chunk(char*raw,const char*word,int portuguese,const RomanceConstant**phones,signed char*stress,unsigned capacity,unsigned*count){
 unsigned n=0;char*p=raw;
 while(*p){while(*p=='_')p++;if(!*p)break;char token[128];unsigned used=0;int pending=0;
  while(*p&&*p!='_'){
   const unsigned char*q=(const unsigned char*)p;uint32_t cp;if(get_cp(&q,&cp))return -3;p=(char*)q;
   if(cp==0x2c8||cp==0x2cc){if(cp==0x2c8||!pending)pending=cp==0x2c8?1:2;continue;}
   if(cp=='g')cp=0x261;
   if(cp==0x303&&used){unsigned char last=(unsigned char)token[used-1];uint32_t combined=last=='a'?0xe3:last=='e'?0x1ebd:last=='i'?0x129:last=='o'?0xf5:last=='u'?0x169:0;
    if(combined){used--;used+=put_cp(token+used,combined);continue;}}
   if(used+4>=sizeof(token))return -3;used+=put_cp(token+used,cp);
  }token[used]=0;
  int rc=romance_split(token,phones,stress,&n,capacity,&pending);if(rc)return rc;
 }
 if(portuguese){
  const char*first[] = {"l","n"},*replacement[]={"ʎ","ɲ"},*spelling[]={"lh","nh"};
  for(unsigned k=0;k<2;k++)if(strstr(word,spelling[k]))for(unsigned i=n;i-->1;){
   if(!strcmp(phones[i-1]->name,first[k])&&!strcmp(phones[i]->name,"j")){
    const RomanceConstant*r=romance_constant(replacement[k]);if(!r)return -4;phones[i-1]=r;stress[i-1]=-1;
    memmove(phones+i,phones+i+1,(n-i-1)*sizeof(*phones));memmove(stress+i,stress+i+1,n-i-1);n--;
   }
  }
  unsigned bytes=strlen(word);int ending=bytes&&word[bytes-1]=='o'?-1:bytes>1&&word[bytes-2]=='o'&&word[bytes-1]=='s'&&n>=2?-2:0;
  if(ending&&!strcmp(phones[(int)n+ending]->name,"w")){phones[(int)n+ending]=romance_constant("u");stress[(int)n+ending]=0;}
 }
 if(!n)return -4;*count=n;return 0;
}
