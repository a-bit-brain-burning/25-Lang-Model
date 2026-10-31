/* Frozen Lithuanian native-name, lexical stress and Baltic numeric contract. */
#include "lithuanian_tables.h"
#include "lithuanian_extra.h"
typedef struct{float reliability,accent;int accent_syllable;}LithuanianMeta;
static const N25FourPhone*lithuanian_phone(const char*name){
 const N25FourPhone*p=phone_by_name(name);if(p)return p;
 for(unsigned i=0;i<sizeof(lithuanian_extra)/sizeof(lithuanian_extra[0]);i++)if(!strcmp(name,lithuanian_extra[i].name))return lithuanian_extra+i;return NULL;
}
static const char*lithuanian_name(const char*name){
 for(unsigned i=0;i<sizeof(lithuanian_names)/sizeof(lithuanian_names[0]);i++)if(!strcmp(name,lithuanian_names[i].name))return lithuanian_names[i].ipa;return NULL;
}
static int lt_family(uint32_t cp){
 if(cp=='a'||cp==0x105||cp==0x251||cp==0x250||cp==0x28c||cp==0x252)return 0;
 if(cp=='e'||cp==0x119||cp==0x117||cp==0x25b||cp==0xe6)return 1;
 if(cp=='i'||cp=='y'||cp==0x12f||cp==0x26a)return 2;
 if(cp=='o'||cp==0x254)return 3;
 if(cp=='u'||cp==0x173||cp==0x16b||cp==0x28a)return 4;return -1;
}
typedef struct{unsigned groups,nuclei,paths;unsigned char expected[64][2],width[64],actual[64][2],actual_width[64];unsigned indices[64];uint64_t selected;}LTAlignment;
static void lt_align_visit(LTAlignment*a,unsigned gi,unsigned vi,uint64_t joins){
 if(a->paths>1)return;if(gi==a->groups){if(vi==a->nuclei){a->paths++;a->selected=joins;}return;}if(vi>=a->nuclei)return;
 unsigned width=a->width[gi];
 if(a->actual_width[vi]==width&&a->actual[vi][0]==a->expected[gi][0]&&(width==1||a->actual[vi][1]==a->expected[gi][1]))lt_align_visit(a,gi+1,vi+1,joins);
 if(width==2&&vi+1<a->nuclei&&a->indices[vi+1]==a->indices[vi]+1&&a->actual_width[vi]==1&&a->actual_width[vi+1]==1&&a->actual[vi][0]==a->expected[gi][0]&&a->actual[vi+1][0]==a->expected[gi][1])lt_align_visit(a,gi+1,vi+2,joins|((uint64_t)1<<vi));
}
static int lt_merge_diphthongs(FourPhone*phones,unsigned*count,const char*word){
 uint32_t letters[128];unsigned ln=0;const unsigned char*p=(const unsigned char*)word;
 while(*p){if(ln==128||get_cp(&p,letters+ln))return -4;ln++;}
 LTAlignment a={0};
 for(unsigned i=0;i<ln;i++){
  int family=lt_family(letters[i]);if(family<0)continue;int soft=0;
  if(letters[i]=='i'&&i&&lt_family(letters[i-1])<0&&i+1<ln&&(lt_family(letters[i+1])==0||lt_family(letters[i+1])>=3)){i++;family=lt_family(letters[i]);soft=1;}
  if(a.groups==64)return -4;unsigned gi=a.groups++;a.expected[gi][0]=soft&&family==0?1:family;a.width[gi]=1;
  if(i+1<ln){uint32_t x=letters[i],y=letters[i+1];int pair=(x=='a'&&(y=='i'||y=='u'))||(x=='e'&&(y=='i'||y=='u'))||(x=='i'&&y=='e')||(x=='u'&&(y=='i'||y=='o'))||(x=='o'&&(y=='i'||y=='u'));
   if(pair){a.width[gi]=2;a.expected[gi][1]=lt_family(y);i++;}}
 }
 for(unsigned i=0;i<*count;i++)if(phones[i].phone->features[31]){
  if(a.nuclei==64)return -4;unsigned ni=a.nuclei++;a.indices[ni]=i;char core[64];if(latvian_core(core,phones[i].phone->name))return -4;
  const unsigned char*q=(const unsigned char*)core;unsigned width=0;while(*q){uint32_t cp;if(get_cp(&q,&cp)||width==2){width=3;break;}int family=lt_family(cp);if(family<0){width=3;break;}a.actual[ni][width++]=family;}a.actual_width[ni]=width;
 }
 lt_align_visit(&a,0,0,0);if(a.paths!=1)return 0;unsigned ni=0,out=0;
 for(unsigned i=0;i<*count;i++){
  if(ni<a.nuclei&&i==a.indices[ni]){
   if(a.selected&((uint64_t)1<<ni)){
    char first[64],second[64];if(latvian_core(first,phones[i].phone->name)||latvian_core(second,phones[i+1].phone->name)||strlen(first)+strlen(second)>=64)return -4;strcat(first,second);
    const N25FourPhone*phone=lithuanian_phone(first);if(!phone)return -4;phones[out++]=(FourPhone){phone,phones[i].stress>phones[i+1].stress?phones[i].stress:phones[i+1].stress,0};i++;ni+=2;continue;
   }ni++;
  }phones[out++]=phones[i];
 }*count=out;return 0;
}
static int lt_backend(const char*word,FourPhone*phones,unsigned*count,unsigned capacity){
 if(espeak_SetVoiceByName("lt")!=EE_OK)return -5;char raw[4096];unsigned used=0,clauses=0;const void*cursor=word;
 while(cursor&&clauses++<100){const void*old=cursor;const char*value=espeak_TextToPhonemes(&cursor,espeakCHARS_UTF8,'_'<<8);if(old==cursor)return -5;if(!value)value="";unsigned bytes=strlen(value);if(used+bytes+2>=sizeof(raw))return -5;if(used)raw[used++]=' ';memcpy(raw+used,value,bytes);used+=bytes;}raw[used]=0;
 char*tokens[512];unsigned tn=0;char*p=raw;while(*p){while(*p=='_'||*p==' '||*p=='\t'||*p=='\n')p++;if(!*p)break;if(tn==512)return -4;tokens[tn++]=p;while(*p&&*p!='_'&&*p!=' '&&*p!='\t'&&*p!='\n')p++;if(*p)*p++=0;}
 unsigned n=0;float pending=0;
 for(unsigned i=0;i<tn;i++){
  char*token=tokens[i],*q=token,*out=token;if(strchr(token,'\''))pending=1;else if(strchr(token,','))pending=.5f;
  while(*q){if(*q!='\''&&*q!=',')*out++=*q;q++;}*out=0;if(!*token)continue;
  if(!strcmp(token,"@-")&&i+1<tn){const char*next=tokens[i+1];while(*next=='\''||*next==',')next++;if(!strcmp(next,"r")||!strcmp(next,"*"))continue;}
  if(!strcmp(token,";")){
   if(!n||phones[n-1].phone->features[31])return -4;const char*old=phones[n-1].phone->name;char target[64];
   if(!strcmp(old,"ɫ"))old="l";if(strlen(old)+4>=64)return -4;strcpy(target,old);if(!strstr(target,"ʲ"))strcat(target,"ʲ");phones[n-1].phone=lithuanian_phone(target);if(!phones[n-1].phone)return -4;continue;
  }
  const char*ipa=lithuanian_name(token);char target[64];
  unsigned bytes=strlen(token);if(!ipa&&bytes&&token[bytes-1]==';'){
   token[bytes-1]=0;ipa=lithuanian_name(token);if(ipa){const N25FourPhone*base=lithuanian_phone(ipa);if(!base||base->features[31])return -4;if(!strcmp(ipa,"ɫ"))ipa="l";strcpy(target,ipa);strcat(target,"ʲ");ipa=target;}
  }
  if(!ipa||n>=capacity)return -4;const N25FourPhone*phone=lithuanian_phone(ipa);if(!phone)return -4;phones[n++]=(FourPhone){phone,phone->features[31]?pending:0,0};if(phone->features[31])pending=0;
 }if(!n||pending)return -4;
 for(unsigned i=n-1;i-->0;){FourPhone*phone=phones+i;const N25FourPhone*next=phones[i+1].phone;char core[64];if(latvian_core(core,phone->phone->name))return -4;if(phone->phone->features[31]||!strcmp(core,"j"))continue;
  const unsigned char*q=(const unsigned char*)next->name;uint32_t cp;if(get_cp(&q,&cp))return -4;
  int direct=next->features[31]&&(cp=='i'||cp==0x26a||cp=='e'||cp==0x25b||cp==0xe6);
  int cluster=!next->features[31]&&(strstr(next->name,"ʲ")||!strcmp(next->name,"j"))&&strcmp(core,"k")&&strcmp(core,"ɡ");
  if((direct||cluster)&&!strstr(phone->phone->name,"ʲ")){char target[64];const char*name=phone->phone->name;if(!strcmp(name,"ɫ"))name="l";if(strlen(name)+4>=64)return -4;strcpy(target,name);strcat(target,"ʲ");phone->phone=lithuanian_phone(target);if(!phone->phone)return -4;}
 }
 int rc=lt_merge_diphthongs(phones,&n,word);if(rc)return rc;for(unsigned i=0;i<n;i++)if(phones[i].phone->features[31]){*count=n;return 0;}return -4;
}
static int lt_integer(const unsigned char**p,const unsigned char*end,int*out){
 unsigned n;if(*p>=end||*(*p)++!=3||english_vint(p,end,&n))return -1;*out=n&1?-(int)(n/2)-1:(int)(n/2);return 0;
}
static int lt_word(RomanceRecordContext*c,const char*word,FourPhone*phones,unsigned*count,unsigned capacity,LithuanianMeta*meta){
 *meta=(LithuanianMeta){.25f,0,-1};unsigned char record[4096],prediction[1024],raw[4096],expanded[4096];size_t rn=0,pn=0,n=0;
 int rc=fep_lexicon_record(fep_lexicons+4,word,record,sizeof(record),&rn,c->workspace,c->workspace_bytes);
 if(rc==-5)return lt_backend(word,phones,count,capacity);if(rc)return rc;if(!rn)return -4;unsigned mode=record[0]>>4;
 if(mode){
  if(mode>2)return -4;FourPhone candidate[256];unsigned size=0;rc=lt_backend(word,candidate,&size,256);if(rc)return rc;
  for(unsigned i=0;i<size;i++){const char*name=candidate[i].phone->name;char marked[64];if(mode==1&&candidate[i].stress==1){if(strlen(name)+3>=64)return -4;strcpy(marked,"ˈ");strcat(marked,name);name=marked;}
   unsigned code=254;for(unsigned j=0;j<c->d->symbol_count;j++)if(!strcmp(name,c->d->symbols[j])){code=j;break;}
   if(code==254&&name==marked)for(unsigned j=0;j<c->d->symbol_count;j++)if(!strcmp(candidate[i].phone->name,c->d->symbols[j])){code=j;break;}
   prediction[pn++]=(unsigned char)code;
  }record[0]&=15;
 }else{rc=spr_predict(&c->model,word,strlen(word),prediction,sizeof(prediction),&pn);if(rc)return rc;}
 rc=spr_apply(prediction,pn,record,rn,raw,sizeof(raw),&n);if(rc)return rc;rc=aux_expand(c->aux,c->d->aux_bytes,raw,n,expanded,sizeof(expanded),&n);if(rc)return rc;
 unsigned length=0;float pending=0;while(length<n&&expanded[length]!=255){if(length>=capacity||expanded[length]>=c->d->symbol_count)return -4;
  const char*name=c->d->symbols[expanded[length]];if(!strncmp(name,"ˈ",2)||!strncmp(name,"ˌ",2)){pending=(unsigned char)name[1]==0x88?1:.5f;name+=2;}
  const N25FourPhone*phone=lithuanian_phone(name);if(!phone)return -4;phones[length++]=(FourPhone){phone,phone->features[31]?pending:0,1};if(phone->features[31])pending=0;
 }if(!length||length>=n||pending)return -4;
 const unsigned char*p=expanded+length+1,*end=expanded+n;unsigned fields,aux_count=0;int tid,mask,known=1,fallback=0;
 if(p>=end||*p++!=5||english_vint(&p,end,&fields)||fields!=3||lt_integer(&p,end,&tid)||tid<0||tid>3||p>=end)return -4;
 if(!tid){if(*p++!=0)return -4;}
 else{
  if(*p++!=5||english_vint(&p,end,&aux_count))return -4;
  if(tid==1){unsigned value;if(aux_count!=2||p>=end||*p++!=4||english_vint(&p,end,&value)||value!=4)return -4;fallback=1;}
  if(tid==2){unsigned pairs;if(aux_count!=2||p>=end||*p++!=6||english_vint(&p,end,&pairs)||pairs!=2)return -4;
   for(unsigned i=0;i<pairs;i++){unsigned key;if(p>=end||*p++!=4||english_vint(&p,end,&key))return -4;
    if(!key){unsigned kind;if(p>=end||*p++!=4||english_vint(&p,end,&kind)||(kind!=2&&kind!=3))return -4;meta->accent=kind==2?.25f:.75f;}
    else if(key==1){if(lt_integer(&p,end,&meta->accent_syllable))return -4;}else return -4;
   }
  }
  if(tid==3&&aux_count!=1)return -4;if(p>=end||(*p!=1&&*p!=2))return -4;known=*p++==2;
 }
 if(lt_integer(&p,end,&mask)||mask||p!=end)return -4;unsigned stressed=0;
 for(unsigned i=0;i<length;i++)if(phones[i].stress)stressed++;
 for(unsigned i=0;i<length;i++)phones[i].known=known&&stressed;
 meta->reliability=known&&stressed?1.f:0.f;
 if(fallback){FourPhone candidate[256];unsigned size=0,left[256],ln=0,rnuc=0;
  rc=lt_backend(word,candidate,&size,256);if(rc)return rc;for(unsigned i=0;i<length;i++)if(phones[i].phone->features[31])left[ln++]=i;
  for(unsigned i=0;i<size;i++)if(candidate[i].phone->features[31])rnuc++;
  if(ln==rnuc){unsigned at=0;for(unsigned i=0;i<size;i++)if(candidate[i].phone->features[31]){phones[left[at]].stress=candidate[i].stress;phones[left[at++]].known=0;}meta->reliability=.25f;}
 }*count=length;return 0;
}
