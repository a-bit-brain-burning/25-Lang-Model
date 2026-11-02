/* Frozen Romance lexical-stress 36/22 contract over the complete ES FEP. */
#include "bench.h"
#include "spanish_record.h"
#include "spanish_tables.h"
static int spanish_gap(uint32_t cp){
 if(cp==' '||cp=='\t'||cp=='\r'||cp=='\n'||cp==0xa0)return 1;
 return cp=='.'||cp==','||cp=='!'||cp=='?'||cp==';'||cp==':'||cp==0xbf||cp==0xa1||cp==0x2014||cp==0x2013||cp=='-'||cp=='('||cp==')'||cp==0xab||cp==0xbb||cp=='"'||cp=='\''||cp==0x2026;
}
static float spanish_strength(const char*text,unsigned length){
 float strength=0;const unsigned char*p=(const unsigned char*)text,*end=p+length;
 while(p<end){uint32_t cp;if(get_cp(&p,&cp))return -1;
  if(cp=='.'||cp=='!'||cp=='?'||cp==0x2026)return 1;
  if(cp==';'||cp==':'){if(strength<.65f)strength=.65f;}
  else if(cp==','){if(strength<.4f)strength=.4f;}
  else if(cp==0x2014||cp==0x2013){if(strength<.25f)strength=.25f;}
 }return strength;
}
static int spanish_frontend_impl(const char*text,N25FourOutput*out,SpanishContext*context){
 char normalized[4096];unsigned length=0;const unsigned char*p=(const unsigned char*)text;
 while(*p){uint32_t cp;if(get_cp(&p,&cp))return -3;
  if(cp==0x2019||cp==0x2bc)cp='\'';
  const N25FourCharacter*ch=character(cp);if(ch)cp=ch->lower;
  if(!ch&&!spanish_gap(cp))return -3;
  if(length+4>=sizeof(normalized))return -3;
  if(cp==0xdf){normalized[length++]='s';normalized[length++]='s';}
  else length+=put_cp(normalized+length,cp);
 }normalized[length]=0;
 FourWord words[64];unsigned can_split[64],wn=0,at=0;
 while(at<length){
  const unsigned char*q=(const unsigned char*)normalized+at;uint32_t cp;if(get_cp(&q,&cp))return -3;
  if(!character(cp)){at=(unsigned)(q-(const unsigned char*)normalized);continue;}
  if(wn==64)return -3;unsigned begin=at;
  while(at<length){
   q=(const unsigned char*)normalized+at;if(get_cp(&q,&cp))return -3;
   if(character(cp)){at=(unsigned)(q-(const unsigned char*)normalized);continue;}
   if(cp=='\''){
    at=(unsigned)(q-(const unsigned char*)normalized);const unsigned char*next=q;uint32_t nc;
    if(*next&&!get_cp(&next,&nc)&&character(nc))continue;
   }break;
  }
  words[wn]=(FourWord){begin,at,0,0,0,0};can_split[wn++]=1;
 }
 if(!wn)return -3;
 unsigned char phones[256];signed char stresses[256];unsigned used=0;
 for(unsigned wi=0;wi<wn;wi++){
  FourWord*w=words+wi;char word[512];unsigned bytes=w->hi-w->lo,n=0;
  if(bytes>=sizeof(word))return -3;memcpy(word,normalized+w->lo,bytes);word[bytes]=0;
  int rc=spanish_word(context,word,phones+used,stresses+used,256-used,&n);
  if(rc==-5&&can_split[wi]&&strchr(word,'\'')){
   unsigned prefix=(unsigned)(strchr(word,'\'')-word)+1;
   if(prefix==bytes||wn==64)return -5;
   unsigned old_end=w->hi;
   memmove(words+wi+2,words+wi+1,(wn-wi-1)*sizeof(*words));
   memmove(can_split+wi+2,can_split+wi+1,(wn-wi-1)*sizeof(*can_split));
   w->hi=w->lo+prefix;words[wi+1]=(FourWord){w->hi,old_end,0,0,0,0};can_split[wi]=can_split[wi+1]=0;wn++;
   word[prefix]=0;rc=spanish_word(context,word,phones+used,stresses+used,256-used,&n);
  }
  if(rc)return rc;w->phone_start=used;used+=n;w->phone_end=used;
 }
 unsigned total=used+wn+1;if(total>256)return -6;
 for(unsigned wi=0;wi<wn;wi++){
  unsigned before=wi?words[wi-1].hi:0,after=wi+1<wn?words[wi+1].lo:length;
  words[wi].before=spanish_strength(normalized+before,words[wi].lo-before);
  words[wi].after=spanish_strength(normalized+words[wi].hi,after-words[wi].hi);
 }
 memset(out->pause,0,total*4);memset(out->phonetic,0,(size_t)total*36*4);memset(out->context,0,(size_t)total*22*4);
 unsigned count=0;
 for(unsigned wi=0;wi<=wn;wi++){
  out->phones[count]=spanish_phones[SPANISH_SYMBOL_COUNT].id;out->pause[count]=1;
  memcpy(out->phonetic+count*36,spanish_phones[SPANISH_SYMBOL_COUNT].features,36*4);
  float*row=out->context+count*22;
  row[5]=(float)((double)count/(total-1));row[6]=(float)((double)total/256.);row[7]=(float)((double)wn/64.>1?1:(double)wn/64.);
  if(!wi)row[10]=1;else if(wi==wn){row[11]=row[8]=1;count++;break;}else row[8]=row[9]=.25f;
  count++;const FourWord*w=words+wi;unsigned size=w->phone_end-w->phone_start;
  for(unsigned i=0;i<size;i++,count++){
   unsigned index=w->phone_start+i,symbol=phones[index];int stress=stresses[index];
   out->phones[count]=spanish_phones[symbol].id;memcpy(out->phonetic+count*36,spanish_phones[symbol].features,36*4);
   row=out->context+count*22;
   row[0]=i==0;row[1]=i+1==size;row[2]=(float)((double)i/(size>1?size-1:1));row[3]=(float)((double)size/20.>1?1:(double)size/20.);
   row[4]=(float)((double)wi/(wn>1?wn-1:1));row[5]=(float)((double)count/(total-1));row[6]=(float)((double)total/256.);row[7]=(float)((double)wn/64.>1?1:(double)wn/64.);
   row[8]=i+1==size?w->after:0;row[9]=i==0?w->before:0;
   if(stress>=0){row[12]=stress==1?1.f:stress==0?0.f:.5f;row[13]=1;}
  }
 }
 out->count=count;out->phonetic_dim=36;out->context_dim=22;return count==total?0:-6;
}
static int n25_spanish_to_view(const char*text,N25FourOutput*out){
 if(!text||!out)return -1;
 SpanishContext ready;
 if(!spanish_context_init(&ready,NULL,0,NULL,0))return spanish_frontend_impl(text,out,&ready);
 size_t mark=bench_arena_mark();
 void*workspace=bench_alloc(664098,16),*model=bench_alloc(SPANISH_MODEL_BYTES,16);int rc=-98;
 if(workspace&&model){SpanishContext context;rc=spanish_context_init(&context,model,SPANISH_MODEL_BYTES,workspace,664098);
  if(!rc)rc=spanish_frontend_impl(text,out,&context);}
 if(bench_arena_rewind(mark))return -96;return rc;
}
