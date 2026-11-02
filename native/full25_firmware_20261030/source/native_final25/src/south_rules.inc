/* Frozen native-name BG/SL adapters and full source-bound lexical prosody. */
#include "south_names.h"
#include "paged_lexicon.h"
static const char*south_name(unsigned language,const char*token){
 for(unsigned i=0;i<south_name_counts[language];i++)if(!strcmp(token,south_names[language][i].name))return south_names[language][i].ipa;
 return NULL;
}
static int south_family(const char*name,unsigned family){
 static const char*const families[]={"a|ɐ","e|ɛ|ə","i|ɪ","o|ɔ","u|ʊ","ə|ɤ","ə|r̩"};
 if(family>=7)return 0;char core[64];if(finnic_core(core,name))return 0;const char*p=families[family];size_t bytes=strlen(core);
 while(*p){const char*end=strchr(p,'|');size_t n=end?(size_t)(end-p):strlen(p);if(n==bytes&&!memcmp(p,core,n))return 1;if(!end)break;p=end+1;}return 0;
}
static int south_prosody(const char*language,const char*word,FourPhone*phones,unsigned count){
 unsigned char record[256];size_t bytes=0;int rc=n25_paged_lexicon_record(language,word,record,sizeof(record),&bytes);if(rc==-5)return 0;if(rc)return rc;
 if(bytes<3||record[0]>32||bytes<3u+record[0])return -4;unsigned nuclei[32],n=0;
 for(unsigned i=0;i<count;i++)if(phones[i].phone->features[31]){if(n==32)return -4;nuclei[n++]=i;}
 if(n!=record[0])return 0;for(unsigned i=0;i<n;i++)if(!south_family(phones[nuclei[i]].phone->name,record[2+i]))return 0;
 if(record[1]){if(record[1]>n)return -4;for(unsigned i=0;i<n;i++){phones[nuclei[i]].stress=i+1==record[1];phones[nuclei[i]].known=1;}}
 const unsigned char*p=record+2+n,*end=record+bytes;unsigned vowels=*p++;
 for(unsigned i=0;i<vowels;i++){
  if(end-p<3)return -4;unsigned si=*p++,flags=*p++,length=*p++;if(si>=n||length>(size_t)(end-p)||length>=64)return -4;
  char name[64],core[64],target[64];memcpy(name,p,length);name[length]=0;p+=length;
  FourPhone*phone=phones+nuclei[si];if(finnic_core(core,flags&2?name:phone->phone->name))return -4;
  strcpy(target,core);int is_long=strstr(flags&1?name:phone->phone->name,"ː")!=NULL;if(is_long)strcat(target,"ː");
  const N25FourPhone*replacement=phone_by_name(target);if(!replacement)return -4;phone->phone=replacement;
  if(flags&1)phone->known|=2;if(flags&2)phone->known|=4;
 }return p==end?0:-4;
}
static int parse_south(char*raw,const char*word,FourPhone*phones,unsigned*count,unsigned capacity,int bulgarian){
 char*tokens[512];unsigned tn=0;char*p=raw;
 while(*p){while(*p=='_'||*p==' '||*p=='\t'||*p=='\n')p++;if(!*p)break;if(tn==512)return -4;tokens[tn++]=p;
  while(*p&&*p!='_'&&*p!=' '&&*p!='\t'&&*p!='\n')p++;if(*p)*p++=0;
 }
 unsigned n=0;float pending=0;
 for(unsigned i=0;i<tn;i++){
  char*token=tokens[i],*q=token,*v=token;if(strchr(token,'\''))pending=1;else if(strchr(token,','))pending=.5f;
  while(*q){if(*q!='\''&&*q!=',')*v++=*q;q++;}*v=0;if(!*token)continue;
  if(!bulgarian&&!strcmp(token,"|"))continue;
  if(!strcmp(token,";")){
   char core[64];if(!n||finnic_core(core,phones[n-1].phone->name))return -4;
   if(phones[n-1].phone->features[31]&&!strcmp(core,"i"))continue;
   if(bulgarian&&!phones[n-1].phone->features[31]){phones[n-1].phone=finnic_suffix(phones[n-1].phone,"ʲ");if(!phones[n-1].phone)return -4;continue;}return -4;
  }
  if(!strcmp(token,":")){if(!n||!phones[n-1].phone->features[31])return -4;if(!strstr(phones[n-1].phone->name,"ː"))phones[n-1].phone=finnic_suffix(phones[n-1].phone,"ː");if(!phones[n-1].phone)return -4;continue;}
  unsigned length=strlen(token);int is_long=length&&token[length-1]==':';if(is_long)token[--length]=0;
  const char*name=south_name(bulgarian,token);char palatal[64];
  if(!name&&bulgarian&&length&&token[length-1]==';'){token[--length]=0;name=south_name(1,token);if(name){strcpy(palatal,name);strcat(palatal,"ʲ");name=palatal;}}
  if(!name)return -4;
  if(!bulgarian&&(!strcmp(token,"l^")||!strcmp(token,"n^"))){
   name=token[0]=='l'?"l":"n";
   if(i+1<tn){char next[64];unsigned used=0;for(const char*at=tokens[i+1];*at&&used+1<sizeof(next);at++)if(*at!='\''&&*at!=','&&*at!=':')next[used++]=*at;next[used]=0;
    const char*following=south_name(0,next);const N25FourPhone*f=following?phone_by_name(following):NULL;
    if(f&&f->features[31]){if(n>=capacity)return -4;phones[n++]=(FourPhone){phone_by_name(name),0,0};name="j";}
   }
  }
  if(n>=capacity)return -4;const N25FourPhone*phone=phone_by_name(name);if(phone&&is_long)phone=finnic_suffix(phone,"ː");if(!phone)return -4;
  phones[n++]=(FourPhone){phone,phone->features[31]?pending:0,0};if(phone->features[31])pending=0;
 }
 if(!n||pending)return -4;
 if(bulgarian){
  unsigned spellings[128],letters=0,targets[128],found=0;const unsigned char*at=(const unsigned char*)word;
  while(*at){uint32_t cp;if(get_cp(&at,&cp))return -4;if(cp==0x43b){const unsigned char*next=at;uint32_t following=0;if(*next&&get_cp(&next,&following))return -4;
   if(letters==128)return -4;spellings[letters++]=following==0x44f||following==0x44e||following==0x44c;}}
  for(unsigned i=0;i<n;i++){char core[64];if(finnic_core(core,phones[i].phone->name))return -4;if(!strcmp(core,"l")||!strcmp(core,"ɫ")){if(found==128)return -4;targets[found++]=i;}}
  if(letters==found){for(unsigned i=0;i<found;i++)if(spellings[i]&&!strstr(phones[targets[i]].phone->name,"ʲ")){phones[targets[i]].phone=phone_by_name("lʲ");if(!phones[targets[i]].phone)return -4;}}
  else for(unsigned i=0;i<letters;i++)if(spellings[i])return -4;
 }
 int rc=south_prosody(bulgarian?"bg":"sl",word,phones,n);if(rc)return rc;*count=n;return 0;
}
