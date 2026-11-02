/* Original CMU residuals, eSpeak predictor selector and English pause topology. */
#include "english_tables.h"
#include "romance_record.h"
#include "fep_dictionary.h"
#include "value_decoders.h"
extern const RomanceDictionary final_dictionaries[2];
static int english_prediction(const char*word,unsigned char*out,unsigned capacity,size_t*length){
 if(espeak_SetVoiceByName("en-us")!=EE_OK)return -5;const void*cursor=word;unsigned n=0,clauses=0;
 while(cursor&&clauses++<100){const void*old=cursor;const unsigned char*p=(const unsigned char*)espeak_TextToPhonemes(&cursor,espeakCHARS_UTF8,2|('_'<<8));if(old==cursor)return -5;if(!p)p=(const unsigned char*)"";
  while(*p){while(*p=='_'||*p==' '||*p=='\t'||*p=='\n')p++;if(!*p)break;char name[64];unsigned used=0,stress=1;
   while(*p&&*p!='_'&&*p!=' '&&*p!='\t'&&*p!='\n'){uint32_t cp;if(get_cp(&p,&cp))return -3;
    if(cp==0x2c8||cp==0x2cc){if(cp==0x2c8||stress!=2)stress=cp==0x2c8?2:3;continue;}if(used+4>=sizeof(name))return -3;used+=put_cp(name+used,cp);
   }name[used]=0;unsigned code=254;
   for(unsigned i=0;i<sizeof(english_prediction_map)/sizeof(english_prediction_map[0]);i++)if(!strcmp(name,english_prediction_map[i].ipa)){
    unsigned symbol=english_prediction_map[i].code;code=symbol*4+(strchr("AEIOU",english_symbols[symbol][0])?stress:0);break;}
   if(n>=capacity)return -6;out[n++]=(unsigned char)code;
  }
 }*length=n;return 0;
}
static int english_vint(const unsigned char**p,const unsigned char*end,unsigned*out){
 unsigned n=0;for(unsigned shift=0;shift<=28;shift+=7){if(*p>=end)return -1;unsigned b=*(*p)++;if(shift==28&&(b&240))return -1;n|=(b&127)<<shift;if(!(b&128)){*out=n;return 0;}}return -1;
}
static int english_string(const unsigned char**p,const unsigned char*end,char*out,unsigned cap){
 if(*p>=end)return -1;unsigned tag=*(*p)++;if(!tag){*out=0;return 0;}unsigned n;
 if(tag!=9||english_vint(p,end,&n)||n>=cap||n>(size_t)(end-*p))return -1;memcpy(out,*p,n);out[n]=0;*p+=n;return 0;
}
static int english_ascii_pronunciation(const char*text,const N25FourPhone**phones,signed char*stress,unsigned cap,unsigned*count){
 unsigned n=0;while(*text){while(*text==' ')text++;if(!*text)break;char name[8];unsigned used=0;
  while(*text&&*text!=' '){if(used+1==sizeof(name))return -4;name[used++]=*text++;}name[used]=0;int s=-1;
  if(used&&name[used-1]>='0'&&name[used-1]<='2')s=name[--used]-'0';name[used]=0;unsigned symbol=39;
  for(unsigned i=0;i<39;i++)if(!strcmp(name,english_symbols[i])){symbol=i;break;}
  if(symbol==39||n>=cap)return -4;phones[n]=phone_by_name(english_ipa[symbol]);if(!phones[n])return -4;stress[n++]=s;
 }if(!n)return -4;*count=n;return 0;
}
static int english_dictionary_word(RomanceRecordContext*c,const char*word,const N25FourPhone**phones,signed char*stress,unsigned cap,unsigned*count){
 unsigned char record[4096],prediction[1024],raw[4096],temporary[4096];size_t rn=0,pn=0,n=0;*count=0;
 int rc=fep_lexicon_record(fep_lexicons,word,record,sizeof(record),&rn,c->workspace,c->workspace_bytes);if(rc)return rc;if(!rn)return -4;
 if(record[0]==1){
  const unsigned char*p=record+1,*end=record+rn;unsigned fields;char base[1024],marked[1024];
  if(p>=end||*p++!=5||english_vint(&p,end,&fields)||fields!=2||english_string(&p,end,base,sizeof(base))||english_string(&p,end,marked,sizeof(marked))||p!=end)return -4;
  /* Productive dictionary policy also accepts a retained stress-only entry. */
  return english_ascii_pronunciation(*marked?marked:base,phones,stress,cap,count);
 }
 if(record[0]==2){
  rc=english_prediction(word,prediction,sizeof(prediction),&pn);
  if(!rc)rc=spr_apply(prediction,pn,record+1,rn-1,raw,sizeof(raw),&n);if(rc)return rc;
  unsigned length=0;while(length<n&&raw[length]!=255){unsigned symbol=raw[length]/4,code=raw[length]&3;
   if(length>=cap||symbol>=39)return -4;phones[length]=phone_by_name(english_ipa[symbol]);if(!phones[length])return -4;stress[length]=code?(int)code-1:-1;length++;
  }if(!length||length>=n)return -4;*count=length;return 0;
 }
 if(record[0])return -4;
 rc=spr_predict(&c->model,word,strlen(word),prediction,sizeof(prediction),&pn);
 if(!rc)rc=en_apply(prediction,pn,record,rn,temporary,sizeof(temporary),raw,sizeof(raw),&n);if(rc)return rc;
 if(n<2||raw[0]||!raw[1]||raw[1]>cap||n!=2u+raw[1]+(raw[1]+3u)/4)return -4;
 unsigned length=raw[1];for(unsigned i=0;i<length;i++){
  unsigned symbol=raw[2+i],code=(raw[2+length+i/4]>>(2*(i%4)))&3;if(symbol>=39)return -4;
  phones[i]=phone_by_name(english_ipa[symbol]);if(!phones[i])return -4;stress[i]=code?(int)code-1:-1;
 }*count=length;return 0;
}
static float english_strength(const char*p,unsigned count){
 float value=0;for(unsigned i=0;i<count;i++){if(p[i]=='.'||p[i]=='!'||p[i]=='?')return 1;if(p[i]==';'||p[i]==':')value=.66f;else if(p[i]==','&&value<.33f)value=.33f;}return value;
}
static int english_impl(const char*text,N25FourOutput*out,RomanceRecordContext*context){
 char normalized[4096];unsigned length=0;
 for(const unsigned char*p=(const unsigned char*)text;*p;p++){
  unsigned c=*p;if(c>=128||length+1>=sizeof(normalized))return -3;
  if(c>='A'&&c<='Z')c+=32;
  if(!(c>='a'&&c<='z')&&c!=' '&&c!='\t'&&c!='\r'&&c!='\n'&&!strchr(".,!?;:'\"()-",c))return -3;
  normalized[length++]=(char)c;
 }normalized[length]=0;FourWord words[64];unsigned wn=0,at=0;
 while(at<length){if(normalized[at]<'a'||normalized[at]>'z'){at++;continue;}if(wn==64)return -3;unsigned begin=at++;
  while(at<length){if(normalized[at]>='a'&&normalized[at]<='z'){at++;continue;}if(normalized[at]=='\''&&at+1<length&&normalized[at+1]>='a'&&normalized[at+1]<='z'){at++;continue;}break;}
  words[wn++]=(FourWord){begin,at,0,0,0,0};
 }if(!wn)return -3;
 const N25FourPhone*phones[512];signed char stresses[512];unsigned lexical_count=0;
 for(unsigned wi=0;wi<wn;wi++){
  FourWord*w=words+wi;char word[512];unsigned bytes=w->hi-w->lo,n=0;if(bytes>=sizeof(word))return -3;memcpy(word,normalized+w->lo,bytes);word[bytes]=0;
  int rc=english_dictionary_word(context,word,phones+lexical_count,stresses+lexical_count,512-lexical_count,&n);if(rc)return rc;
  w->phone_start=lexical_count;lexical_count+=n;w->phone_end=lexical_count;
  unsigned before=wi?words[wi-1].hi:0,after=wi+1<wn?words[wi+1].lo:length;
  w->before=english_strength(normalized+before,w->lo-before);w->after=english_strength(normalized+w->hi,after-w->hi);
 }
 unsigned total=lexical_count+wn+1;if(total>N25_FOUR_PHONE_CAP)return -6;memset(out->pause,0,total*4);memset(out->context,0,(size_t)total*22*4);
 const N25FourPhone*sil=phone_by_name("sil");unsigned count=0,position=0;
 float plen=(float)((double)lexical_count/200.>2?2:(double)lexical_count/200.),wlen=(float)((double)wn/50.>2?2:(double)wn/50.);
 for(unsigned wi=0;wi<=wn;wi++){
  out->phones[count]=sil->id;out->pause[count]=1;memcpy(out->phonetic+count*32,sil->features,32*4);float*row=out->context+count*22;row[6]=plen;row[7]=wlen;
  if(!wi)row[10]=1;else if(wi==wn){row[4]=row[5]=row[11]=1;count++;break;}
  else{row[4]=(float)((double)(wi-1)/(wn>1?wn-1:1));float left=(float)((double)(position-1)/(lexical_count>1?lexical_count-1:1));float right=(float)((double)position/(lexical_count>1?lexical_count-1:1));row[5]=.5f*(left+right);row[8]=row[9]=words[wi-1].after;}
  count++;const FourWord*w=words+wi;unsigned size=w->phone_end-w->phone_start;
  for(unsigned i=0;i<size;i++,count++,position++){
   unsigned index=w->phone_start+i;const N25FourPhone*phone=phones[index];int stress=stresses[index];out->phones[count]=phone->id;memcpy(out->phonetic+count*32,phone->features,32*4);row=out->context+count*22;
   row[0]=i==0;row[1]=i+1==size;row[2]=(float)((double)i/(size>1?size-1:1));row[3]=(float)((double)size/12.>2?2:(double)size/12.);
   row[4]=(float)((double)wi/(wn>1?wn-1:1));row[5]=(float)((double)position/(lexical_count>1?lexical_count-1:1));row[6]=plen;row[7]=wlen;row[8]=i+1==size?w->after:0;row[9]=i==0?w->before:0;
   if(stress>=0){row[12]=stress==1?1.f:stress==2?.5f:0.f;row[13]=1;}
  }
 }out->count=count;out->phonetic_dim=32;out->context_dim=22;return count==total?0:-6;
}
static int n25_english_to_view(const char*text,N25FourOutput*out){
 const RomanceDictionary*d=final_dictionaries;size_t mark=bench_arena_mark();void*work=bench_alloc(d->workspace_bytes,16),*model=bench_alloc(d->model_bytes,16);int rc=-98;
 if(work&&model){RomanceRecordContext context;rc=romance_record_init(&context,d,work,d->workspace_bytes,model,NULL);if(!rc)rc=english_impl(text,out,&context);}
 if(bench_arena_rewind(mark))return -96;return rc;
}
