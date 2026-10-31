#include "romance_record.h"
#include "french_words.h"
static int romance_word_in(const char*text,const FourWord*w,const char*const*names,unsigned count,int strip){
 char word[512];unsigned bytes=w->hi-w->lo;if(bytes>=sizeof(word))return 0;memcpy(word,text+w->lo,bytes);
 if(strip)while(bytes&&word[bytes-1]=='\'')bytes--;word[bytes]=0;
 for(unsigned i=0;i<count;i++)if(!strcmp(word,names[i]))return 1;return 0;
}
static void french_group(const char*normalized,const FourWord*words,unsigned first,unsigned last,const RomanceConstant*const*phones,signed char*stresses){
 int content=0;unsigned target=0;int found=0;
 for(unsigned wi=first;wi<=last;wi++){
  if(!romance_word_in(normalized,words+wi,french_functions,sizeof(french_functions)/sizeof(french_functions[0]),1))content=1;
  for(unsigned i=words[wi].phone_start;i<words[wi].phone_end;i++)if(romance_nucleus(phones[i])&&strncmp(phones[i]->name,"ə",2)){target=i;found=1;}
 }if(content&&found)stresses[target]=1;
}

static int romance_dictionary_impl(const char*language,const char*text,N25FourOutput*out,RomanceRecordContext*context){
 char normalized[4096];unsigned length=0;const unsigned char*p=(const unsigned char*)text;
 while(*p){uint32_t cp;if(get_cp(&p,&cp))return -3;if(cp==0x2019||cp==0x2bc)cp='\'';
  const N25FourCharacter*ch=character(cp);if(ch)cp=ch->lower;
  if(!ch&&!spanish_gap(cp))return -3;if(length+4>=sizeof(normalized))return -3;
  if(cp==0xdf){normalized[length++]='s';normalized[length++]='s';}else length+=put_cp(normalized+length,cp);
 }normalized[length]=0;
 FourWord words[64];unsigned wn=0,at=0;
 while(at<length){const unsigned char*q=(const unsigned char*)normalized+at;uint32_t cp;if(get_cp(&q,&cp))return -3;
  if(!character(cp)){at=(unsigned)(q-(const unsigned char*)normalized);continue;}if(wn==64)return -3;unsigned begin=at;
  while(at<length){q=(const unsigned char*)normalized+at;if(get_cp(&q,&cp))return -3;
   if(character(cp)){at=(unsigned)(q-(const unsigned char*)normalized);continue;}
   if(cp=='\''){at=(unsigned)(q-(const unsigned char*)normalized);const unsigned char*next=q;uint32_t nc;if(*next&&!get_cp(&next,&nc)&&character(nc))continue;}break;
  }words[wn++]=(FourWord){begin,at,0,0,0,0};
 }if(!wn)return -3;
 const RomanceConstant*phones[256];signed char stresses[256];unsigned count=0;
 for(unsigned wi=0;wi<wn;wi++){
  FourWord*w=words+wi;char word[512];unsigned bytes=w->hi-w->lo,n=0;if(bytes>=sizeof(word))return -3;
  memcpy(word,normalized+w->lo,bytes);word[bytes]=0;w->phone_start=count;
  unsigned char symbols[256];
  int rc=romance_record_word(context,word,symbols,stresses+count,256-count,&n);if(rc)return rc;
  for(unsigned i=0;i<n;i++){phones[count+i]=romance_constant(context->d->symbols[symbols[i]]);if(!phones[count+i])return -4;}
  count+=n;w->phone_end=count;
  unsigned before=wi?words[wi-1].hi:0,after=wi+1<wn?words[wi+1].lo:length;
  w->before=spanish_strength(normalized+before,w->lo-before);w->after=spanish_strength(normalized+w->hi,after-w->hi);
 }
 if(!strcmp(language,"fr")){
  memset(stresses,-1,count);unsigned first=0;
  for(unsigned wi=0;wi<wn;wi++){
   if(wi-first>=2&&romance_word_in(normalized,words+wi,french_clause_starts,sizeof(french_clause_starts)/sizeof(french_clause_starts[0]),0)){
    french_group(normalized,words,first,wi-1,phones,stresses);first=wi;
   }
   if(words[wi].after>0){french_group(normalized,words,first,wi,phones,stresses);first=wi+1;}
  }if(first<wn)french_group(normalized,words,first,wn-1,phones,stresses);
 }
 unsigned total=count+wn+1;if(total>256)return -6;
 memset(out->pause,0,total*4);memset(out->context,0,(size_t)total*22*4);count=0;
 const RomanceConstant*sil=romance_constant("sil");if(!sil)return -4;
 for(unsigned wi=0;wi<=wn;wi++){
  out->phones[count]=sil->id;out->pause[count]=1;romance_features(sil,out->phonetic+count*36);
  float*row=out->context+count*22;row[5]=(float)((double)count/(total-1));row[6]=(float)((double)total/256.);row[7]=(float)((double)wn/64.>1?1:(double)wn/64.);
  if(!wi)row[10]=1;else if(wi==wn){row[11]=row[8]=1;count++;break;}else row[8]=row[9]=.25f;
  count++;const FourWord*w=words+wi;unsigned size=w->phone_end-w->phone_start;
  for(unsigned i=0;i<size;i++,count++){
   unsigned index=w->phone_start+i;const RomanceConstant*phone=phones[index];int stress=stresses[index];
   out->phones[count]=phone->id;romance_features(phone,out->phonetic+count*36);row=out->context+count*22;
   row[0]=i==0;row[1]=i+1==size;row[2]=(float)((double)i/(size>1?size-1:1));row[3]=(float)((double)size/20.>1?1:(double)size/20.);
   row[4]=(float)((double)wi/(wn>1?wn-1:1));row[5]=(float)((double)count/(total-1));row[6]=(float)((double)total/256.);row[7]=(float)((double)wn/64.>1?1:(double)wn/64.);
   row[8]=i+1==size?w->after:0;row[9]=i==0?w->before:0;if(stress>=0){row[12]=stress==1?1.f:stress==0?0.f:.5f;row[13]=1;}
  }
 }out->count=count;out->phonetic_dim=36;out->context_dim=22;return count==total?0:-6;
}

static int n25_romance_dictionary(const char*language,const char*text,N25FourOutput*out){
 const RomanceDictionary*d=NULL;
 for(unsigned i=0;i<romance_dictionary_count;i++)if(!strcmp(language,romance_dictionaries[i].language))d=romance_dictionaries+i;
 if(!d)return -2;size_t mark=bench_arena_mark();
 void*work=bench_alloc(d->workspace_bytes,16),*model=bench_alloc(d->model_bytes,16),*aux=d->aux_bytes?bench_alloc(d->aux_bytes,16):NULL;
 int rc=-98;if(work&&model&&(!d->aux_bytes||aux)){RomanceRecordContext context;rc=romance_record_init(&context,d,work,d->workspace_bytes,model,aux);
  if(!rc)rc=romance_dictionary_impl(language,text,out,&context);}
 if(bench_arena_rewind(mark))return -96;return rc;
}
