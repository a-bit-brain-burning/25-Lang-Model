/* Native port of the pinned eu_four phone/stress/context contract.
 * Word pronunciation comes from the embedded original eSpeak resources.
 */
#include "four_frontend.h"
#include "four_tables.h"
#include <espeak-ng/speak_lib.h>
#include <string.h>
#include <stdint.h>
typedef struct {unsigned lo,hi,phone_start,phone_end;float before,after;} FourWord;
typedef struct {const N25FourPhone *phone;float stress;unsigned known;} FourPhone;
static int get_cp(const unsigned char **p,uint32_t *cp){
 unsigned c=*(*p)++,n;uint32_t value,min;
 if(c<128){*cp=c;return c?0:-1;}
 if(c>=194&&c<224){n=1;value=c&31;min=128;}
 else if(c>=224&&c<240){n=2;value=c&15;min=2048;}
 else if(c>=240&&c<=244){n=3;value=c&7;min=65536;}
 else return -1;
 for(unsigned i=0;i<n;i++){c=*(*p)++;if((c&192)!=128)return -1;value=(value<<6)|(c&63);}
 if(value<min||value>0x10ffff||(value>=0xd800&&value<=0xdfff))return -1;
 *cp=value;return 0;
}
static unsigned put_cp(char *out,uint32_t cp){
 if(cp<128){out[0]=(char)cp;return 1;}
 if(cp<2048){out[0]=192|(cp>>6);out[1]=128|(cp&63);return 2;}
 if(cp<65536){out[0]=224|(cp>>12);out[1]=128|((cp>>6)&63);out[2]=128|(cp&63);return 3;}
 out[0]=240|(cp>>18);out[1]=128|((cp>>12)&63);out[2]=128|((cp>>6)&63);out[3]=128|(cp&63);return 4;
}
static const N25FourCharacter *character(uint32_t cp){
 unsigned lo=0,hi=N25_FOUR_CHARACTER_COUNT;
 while(lo<hi){unsigned m=lo+(hi-lo)/2;if(n25_four_characters[m].cp<cp)lo=m+1;else hi=m;}
 return lo<N25_FOUR_CHARACTER_COUNT&&n25_four_characters[lo].cp==cp?n25_four_characters+lo:0;
}
static const N25FourPhone *phone_by_name(const char *name){
 unsigned lo=0,hi=N25_FOUR_PHONE_COUNT;
 while(lo<hi){unsigned m=lo+(hi-lo)/2;if(strcmp(n25_four_phones[m].name,name)<0)lo=m+1;else hi=m;}
 return lo<N25_FOUR_PHONE_COUNT&&!strcmp(n25_four_phones[lo].name,name)?n25_four_phones+lo:0;
}
static float boundary(const char *p,unsigned n,int greek,int croatian){
 float value=0;const unsigned char *at=(const unsigned char*)p,*end=at+n;
 while(at<end){uint32_t cp;if(get_cp(&at,&cp))return -1;
  if(cp=='.'||cp=='!'||cp=='?'||cp==0x2026||(greek&&cp==';')){value=1.f;continue;}
  if(cp==';'||cp==':'||cp==0x2014||cp==0x2013||cp==0xb7||cp==0x387){if(value<.7f)value=.7f;}
  else if(croatian&&(cp=='['||cp==']'||cp==0xab||cp==0xbb||cp==0x2039||cp==0x203a))continue;
  else if(cp==','){if(value<.4f)value=.4f;}
  else if(cp!=' '&&cp!='\t'&&cp!='\r'&&cp!='\n'&&cp!='\''&&cp!='"'&&cp!='-'&&cp!='('&&cp!=')'&&cp!=0xa0)return -1;
 }
 return value;
}
static int append_token(FourPhone *phones,unsigned *count,unsigned capacity,const char *token,float *pending,int hungarian,int greek){
 if(!*token)return 0;
 const N25FourPhone *p=phone_by_name(token);if(!p||*count>=capacity)return -4;
 if(*count&&(hungarian||greek)&&!p->features[31]&&phones[*count-1].phone==p){
  if(hungarian){char name[64];size_t n=strlen(p->name);if(n+3>=sizeof(name))return -4;memcpy(name,p->name,n);memcpy(name+n,"ː",3);
   p=phone_by_name(name);if(!p)return -4;phones[*count-1].phone=p;}
  return 0;
 }
 float stress=p->features[31]?*pending:0.f;if(p->features[31])*pending=0;
 phones[*count]=(FourPhone){p,stress,1};(*count)++;return 0;
}
static int parse_ipa(const char *raw,FourPhone *phones,unsigned *count,unsigned capacity,int danish,int hungarian,int greek){
 char token[64];unsigned used=0;float pending=0;*count=0;
 const unsigned char*p=(const unsigned char*)raw;
 while(*p){uint32_t cp;if(get_cp(&p,&cp))return -3;
  if(cp==0x2c8||cp==0x2cc){pending=cp==0x2c8?1.f:.5f;continue;}
  if(cp==0x361||cp==0x35c)continue;
  if(cp=='g')cp=0x261;
  if(cp==0x30a)cp=0x325;
  if(danish&&cp==0x3b5)cp=0x25b;
  if(cp=='_'||cp==' '||cp=='\t'||cp=='\n'||(danish&&cp==0x294)){
   token[used]=0;int rc=append_token(phones,count,capacity,token,&pending,hungarian,greek);if(rc)return rc;used=0;
   if(danish&&cp==0x294){rc=append_token(phones,count,capacity,"ʔ",&pending,0,0);if(rc)return rc;}
   continue;
  }
  if(used+4>=sizeof(token))return -3;used+=put_cp(token+used,cp);
 }
 token[used]=0;int rc=append_token(phones,count,capacity,token,&pending,hungarian,greek);
 if(rc||pending||!*count)return -4;return 0;
}
#include "finnic_rules.inc"
#include "croatian_rules.inc"
#include "south_rules.inc"
#include "latvian_rules.inc"
#include "spanish_frontend.inc"
#include "romance_live.inc"
#include "romance_dictionary.inc"
#include "english_frontend.inc"
#include "lithuanian_rules.inc"
static int n25_four_core(const char *language,const char *text,N25FourOutput*out,RomanceRecordContext*lt_context){
 if(!language||!text||!out)return -1;
 if(!strcmp(language,"en"))return n25_english_to_view(text,out);
 if(!strcmp(language,"es"))return n25_spanish_to_view(text,out);
 if(!strcmp(language,"it")||!strcmp(language,"pt")||!strcmp(language,"fr")||!strcmp(language,"ro"))return n25_romance_dictionary(language,text,out);
 const N25FourLanguage *lang=NULL;
 for(unsigned i=0;i<N25_FOUR_LANGUAGE_COUNT;i++)if(!strcmp(language,n25_four_languages[i].language)){lang=n25_four_languages+i;break;}
 if(!lang)return -2;
 int greek=!strcmp(language,"el"),hungarian=!strcmp(language,"hu"),danish=!strcmp(language,"da"),turkish=!strcmp(language,"tr");
 int finnic=!strcmp(language,"fi")||!strcmp(language,"et"),estonian=!strcmp(language,"et");
 int croatian=!strcmp(language,"hr"),latvian=!strcmp(language,"lv"),lithuanian=!strcmp(language,"lt");
 int slovenian=!strcmp(language,"sl"),bulgarian=!strcmp(language,"bg");int south=croatian||slovenian||bulgarian;
 unsigned context_dim=finnic||south||latvian||lithuanian?34:22;
 unsigned cap_letters=0,cap_all=1;
 char normalized[4096];unsigned length=0;const unsigned char *p=(const unsigned char*)text;
 while(*p){uint32_t cp;if(get_cp(&p,&cp))return -3;
  if(cp==0x2019||cp==0x2018||cp==0x2bc)cp='\'';
  if(cp==0x201c||cp==0x201d||cp==0x201e||cp==0x201f)cp='"';
  if(cp==0x2010||cp==0x2011)cp='-';
  const N25FourCharacter *ch=character(cp);
  if(south){
   if(ch){cap_letters++;if(ch->lower==cp)cap_all=0;}
   else if(cp!='\''&&cp!='-'){if(cap_letters>1&&cap_all)return -3;cap_letters=0;cap_all=1;}
  }
  if(ch){if(ch->script!=(bulgarian?3u:greek?2u:1u))return -3;cp=ch->lower;}
  if(turkish&&cp==0x130)cp='i';else if(turkish&&cp=='i'&&*(p-1)=='I')cp=0x131;
  if(croatian&&ch&&!(cp>='a'&&cp<='z')&&cp!=0x10d&&cp!=0x107&&cp!=0x111&&cp!=0x161&&cp!=0x17e)return -3;
  if((slovenian||bulgarian)&&ch){const uint32_t*alphabet=bulgarian?bg_alphabet:sl_alphabet;unsigned n=bulgarian?sizeof(bg_alphabet)/4:sizeof(sl_alphabet)/4;int allowed=0;for(unsigned j=0;j<n;j++)if(cp==alphabet[j])allowed=1;if(!allowed)return -3;}
  if(lithuanian&&ch){int allowed=0;for(unsigned j=0;j<sizeof(lithuanian_alphabet)/4;j++)if(cp==lithuanian_alphabet[j])allowed=1;if(!allowed)return -3;}
  if(latvian&&ch){int allowed=0;for(unsigned j=0;j<sizeof(latvian_alphabet)/sizeof(latvian_alphabet[0]);j++)if(cp==latvian_alphabet[j])allowed=1;if(!allowed)return -3;}
  if((finnic||south)&&cp=='-'&&*p=='-'){while(*p=='-')p++;cp=0x2014;}
  if(finnic&&ch&&!(cp>='a'&&cp<='z')&&cp!=0x161&&cp!=0x17e&&cp!=0xe4&&cp!=0xf6&&!(estonian?(cp==0xf5||cp==0xfc):(cp==0xe5)))return -3;
  if(length+4>=sizeof(normalized))return -3;length+=put_cp(normalized+length,cp);
 }
 if(south&&cap_letters>1&&cap_all)return -3;
 normalized[length]=0;
 FourWord words[64];LithuanianMeta lt_meta[64];unsigned wn=0;size_t at=0;
 while(at<length){
  const unsigned char *q=(const unsigned char*)normalized+at;uint32_t cp;if(get_cp(&q,&cp))return -3;
  if(!character(cp)){at=(size_t)(q-(const unsigned char*)normalized);continue;}
  if(wn>=64)return -3;unsigned begin=at,end;
  for(;;){
   q=(const unsigned char*)normalized+at;if(!*q)break;const unsigned char*old=q;if(get_cp(&q,&cp))return -3;
   if(character(cp)){at=(size_t)(q-(const unsigned char*)normalized);continue;}
   if(cp=='\''||cp=='-'){
    const unsigned char *next=q;uint32_t following;
    if(*next&&!get_cp(&next,&following)&&character(following)){at=(size_t)(q-(const unsigned char*)normalized);continue;}
   }
   (void)old;break;
  }
  end=at;words[wn++]=(FourWord){begin,end,0,0,0,0};
 }
 if(!wn)return -3;
 for(unsigned i=0;i<wn;i++){
  unsigned before=i?words[i-1].hi:0,after=i+1<wn?words[i+1].lo:length;
  words[i].before=boundary(normalized+before,words[i].lo-before,greek,south);
  words[i].after=boundary(normalized+words[i].hi,after-words[i].hi,greek,south);
  if(words[i].before<0||words[i].after<0)return -3;
 }
 if(espeak_SetVoiceByName(lang->voice)!=EE_OK)return -5;
 FourPhone lexical[N25_FOUR_PHONE_CAP];unsigned lexical_count=0;
 for(unsigned wi=0;wi<wn;wi++){
  FourWord*w=words+wi;unsigned bytes=w->hi-w->lo;char word[512],raw[4096];
  if(bytes>=sizeof(word))return -3;memcpy(word,normalized+w->lo,bytes);word[bytes]=0;
  if(lithuanian){unsigned n=0;w->phone_start=lexical_count;int rc=lt_word(lt_context,word,lexical+lexical_count,&n,N25_FOUR_PHONE_CAP-lexical_count,lt_meta+wi);if(rc)return rc;lexical_count+=n;w->phone_end=lexical_count;continue;}
  unsigned used=0;
  const char *single=hungarian&&!strcmp(word,"a")?"ɑ":hungarian&&!strcmp(word,"s")?"ʃ":
      greek&&(!strcmp(word,"η")||!strcmp(word,"ή"))?"i":greek&&!strcmp(word,"ο")?"o":croatian&&(!strcmp(word,"s")||!strcmp(word,"k"))?word:
      slovenian&&(!strcmp(word,"v")||!strcmp(word,"s")||!strcmp(word,"z")||!strcmp(word,"k")||!strcmp(word,"h"))?(!strcmp(word,"h")?"x":word):
      bulgarian&&!strcmp(word,"в")?"v":bulgarian&&!strcmp(word,"с")?"s":NULL;
  if(single){strcpy(raw,single);used=strlen(raw);}
  else{
   const void*cursor=word;unsigned clauses=0;
   while(cursor&&clauses++<100){
    const void*old=cursor;const char*value=espeak_TextToPhonemes(&cursor,espeakCHARS_UTF8,((finnic||south)?0:2)|('_'<<8));
    if(cursor==old)return -5;if(!value)value="";size_t n=strlen(value);
    if(used+n+2>=sizeof(raw))return -5;if(used)raw[used++]=' ';memcpy(raw+used,value,n);used+=n;
   }
   raw[used]=0;
  }
  unsigned n=0;w->phone_start=lexical_count;
  int rc=(slovenian||bulgarian)?parse_south(raw,word,lexical+lexical_count,&n,N25_FOUR_PHONE_CAP-lexical_count,bulgarian):
   latvian?parse_latvian(raw,word,lexical+lexical_count,&n,N25_FOUR_PHONE_CAP-lexical_count):
   croatian?parse_croatian(raw,word,lexical+lexical_count,&n,N25_FOUR_PHONE_CAP-lexical_count):
   finnic?parse_finnic(raw,word,lexical+lexical_count,&n,N25_FOUR_PHONE_CAP-lexical_count,estonian):
   parse_ipa(raw,lexical+lexical_count,&n,N25_FOUR_PHONE_CAP-lexical_count,danish,hungarian,greek);if(rc)return rc;
  if(n>N25_FOUR_PHONE_CAP-lexical_count)return -6;lexical_count+=n;w->phone_end=lexical_count;
 }
 unsigned total=lexical_count+wn+1;if(total>N25_FOUR_PHONE_CAP)return -6;
 unsigned dim=lang->phonetic_dim;
 memset(out->phones,0,total*4);memset(out->pause,0,total*4);
 memset(out->phonetic,0,(size_t)total*dim*4);memset(out->context,0,(size_t)total*context_dim*4);
 const N25FourPhone *sil=phone_by_name("sil");if(!sil)return -6;
 unsigned count=0,position=0;
 for(unsigned wi=0;wi<=wn;wi++){
  unsigned begin=count++;out->phones[begin]=sil->id;out->pause[begin]=1;memcpy(out->phonetic+begin*dim,sil->features,32*4);
  float*c=out->context+begin*context_dim;
  c[6]=(float)((double)lexical_count/200.>2?2:(double)lexical_count/200.);
  c[7]=(float)((double)wn/50.>2?2:(double)wn/50.);
  if(!wi)c[10]=1;else if(wi==wn){c[11]=1;c[4]=c[5]=1;break;}
  else{c[4]=(float)((double)(wi-1)/(wn>1?wn-1:1));c[5]=(float)((double)position/(lexical_count>1?lexical_count-1:1));c[8]=c[9]=words[wi-1].after;}
  FourWord*w=words+wi;unsigned n=w->phone_end-w->phone_start;
  unsigned nuclei[128],nn=0;
  for(unsigned i=0;i<n;i++)if(lexical[w->phone_start+i].phone->features[31]){if(nn>=128)return -6;nuclei[nn++]=i;}
  unsigned starts[129],feet[129],fn=0;starts[0]=0;
  for(unsigned si=1;si<nn;si++)starts[si]=nuclei[si]==nuclei[si-1]+1?nuclei[si]:nuclei[si]-1;
  starts[nn]=n;feet[fn++]=0;
  for(unsigned si=1;si<nn;si++)if(lexical[w->phone_start+nuclei[si]].stress>0)feet[fn++]=si;
  feet[fn]=nn;unsigned foot=0;
  int question=0;
  if(finnic||south||latvian||lithuanian){unsigned end=wi;while(end+1<wn&&words[end].after<.7f)end++;unsigned stop=end+1<wn?words[end+1].lo:length;
   for(unsigned j=words[end].hi;j<stop;j++)if(normalized[j]=='?')question=1;}
  unsigned stressed_count=0,stressed_index=0;
  if(south||latvian||lithuanian)for(unsigned si=0;si<nn;si++)if(lexical[w->phone_start+nuclei[si]].stress==1){stressed_count++;stressed_index=si;}
  unsigned syllable=0,syllable_start=0;
  for(unsigned i=0;i<n;i++,count++,position++){
   FourPhone*p=lexical+w->phone_start+i;float*x=out->phonetic+count*dim,*row=out->context+count*context_dim;
   out->phones[count]=p->phone->id;memcpy(x,p->phone->features,32*4);
   if(dim==36){memcpy(x+32,p->phone->extra,4*4);if(x[32]||x[33])x[6]=0;}
   row[0]=i==0;row[1]=i+1==n;row[2]=(float)((double)i/(n>1?n-1:1));row[3]=(float)((double)n/12.>2?2:(double)n/12.);
   row[4]=(float)((double)wi/(wn>1?wn-1:1));row[5]=(float)((double)position/(lexical_count>1?lexical_count-1:1));
   row[6]=out->context[6];row[7]=out->context[7];row[8]=i+1==n?w->after:0;row[9]=i==0?w->before:0;
   unsigned syllable_end=n;
   if(nn&&syllable+1<nn){unsigned left=nuclei[syllable],right=nuclei[syllable+1];syllable_end=right==left+1?right:right-1;}
   if(nn){row[12]=lexical[w->phone_start+nuclei[syllable]].stress;row[13]=lexical[w->phone_start+nuclei[syllable]].known&1;}
   if(finnic&&nn){
    while(foot+1<fn&&syllable>=feet[foot+1])foot++;
    unsigned fs=feet[foot],fe=feet[foot+1];
    row[22]=(float)((double)syllable/(nn>1?nn-1:1));row[23]=(float)((double)nn/8.>2?2:(double)nn/8.);
    row[24]=(float)((double)(i-syllable_start)/(syllable_end-syllable_start>1?syllable_end-syllable_start-1:1));
    row[25]=syllable==fs;row[26]=(float)((double)(syllable-fs)/(fe-fs>1?fe-fs-1:1));row[27]=(float)((double)(fe-fs)/4.>2?2:(double)(fe-fs)/4.);
    row[32]=syllable+1==nn&&(w->after>0||wi+1==wn);row[33]=question;
   }
   if(south&&nn){
    row[22]=(float)((double)syllable/(nn>1?nn-1:1));row[23]=(float)((double)nn/8.>2?2:(double)nn/8.);
    row[24]=(float)((double)(i-syllable_start)/(syllable_end-syllable_start>1?syllable_end-syllable_start-1:1));
    row[25]=syllable+1==nn&&(w->after>0||wi+1==wn);row[26]=question;
    if(i==nuclei[syllable]){char core[64];if(finnic_core(core,p->phone->name))return -6;if(croatian)row[28]=!strcmp(core,"a")||!strcmp(core,"e")||!strcmp(core,"i")||!strcmp(core,"o")||!strcmp(core,"u");else{row[27]=(p->known&2)!=0;row[28]=(p->known&4)!=0;}}
    if(stressed_count==1)row[33]=(float)(((double)syllable-stressed_index)/(nn>1?nn-1:1));
   }
   if((latvian||lithuanian)&&nn){
    const N25FourPhone*nucleus=lexical[w->phone_start+nuclei[syllable]].phone;
    int heavy=nucleus->features[22]!=0||nucleus->features[30]!=0;
    for(unsigned j=nuclei[syllable]+1;j<syllable_end;j++)if(latvian_sonorant(lexical[w->phone_start+j].phone->name))heavy=1;
    row[22]=(float)((double)syllable/(nn>1?nn-1:1));row[23]=(float)((double)nn/8.>2?2:(double)nn/8.);
    row[24]=(float)((double)(i-syllable_start)/(syllable_end-syllable_start>1?syllable_end-syllable_start-1:1));
    row[25]=heavy;row[26]=nucleus->features[22];row[27]=nucleus->features[30];
    if(stressed_count==1){double before=syllable<stressed_index?(double)(stressed_index-syllable)/4.:0,after=syllable>stressed_index?(double)(syllable-stressed_index)/4.:0;row[28]=(float)(before>2?2:before);row[29]=(float)(after>2?2:after);}
    row[30]=lithuanian?lt_meta[wi].reliability:lexical[w->phone_start+nuclei[0]].stress==1?.5f:.25f;
    if(lithuanian&&lt_meta[wi].accent_syllable==(int)syllable){row[18]=lt_meta[wi].accent;row[19]=1;}row[31]=syllable+1==nn;
    row[32]=syllable+1==nn&&(w->after>0||wi+1==wn);row[33]=question;
   }
   if(!south||nn){row[20]=i==syllable_start;row[21]=i+1==syllable_end;}
   if(i+1==syllable_end){syllable++;syllable_start=syllable_end;}
  }
 }
 out->count=count;out->phonetic_dim=dim;out->context_dim=context_dim;return count==total?0:-6;
}

int n25_four_frontend_to_view(const char*language,const char*text,N25FourOutput*out){
 if(!language||!text||!out)return -1;
 if(strcmp(language,"lt"))return n25_four_core(language,text,out,NULL);
 const RomanceDictionary*d=final_dictionaries+1;size_t mark=bench_arena_mark();
 void*work=bench_alloc(d->workspace_bytes,16),*model=bench_alloc(d->model_bytes,16),*aux=bench_alloc(d->aux_bytes,16);int rc=-98;
 if(work&&model&&aux){RomanceRecordContext context;rc=romance_record_init(&context,d,work,d->workspace_bytes,model,aux);if(!rc)rc=n25_four_core(language,text,out,&context);}
 if(bench_arena_rewind(mark))return -96;return rc;
}

int n25_four_frontend(const char*language,const char*text,N25FourFeatures*out){
 if(!out)return -1;
 N25FourOutput view={0,0,0,out->phones,out->pause,out->phonetic,out->context};
 int rc=n25_four_frontend_to_view(language,text,&view);
 if(!rc){out->count=view.count;out->phonetic_dim=view.phonetic_dim;out->context_dim=view.context_dim;}
 return rc;
}
