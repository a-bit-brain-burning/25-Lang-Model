/* Frozen eu_south_slavic Croatian native-name parsing. */
#include "croatian_names.h"
static int parse_croatian(char*raw,const char*word,FourPhone*phones,unsigned*count,unsigned capacity){
 char*tokens[512];unsigned tn=0;char*p=raw;
 while(*p){while(*p=='_'||*p==' '||*p=='\t'||*p=='\n')p++;if(!*p)break;
  if(tn==512)return -4;tokens[tn++]=p;
  while(*p&&*p!='_'&&*p!=' '&&*p!='\t'&&*p!='\n')p++;if(*p)*p++=0;
 }
 unsigned n=0,nuclei=0;float pending=0;
 for(unsigned i=0;i<tn;i++){
  char*token=tokens[i],*q=token,*v=token;
  if(strchr(token,'\''))pending=1;else if(strchr(token,','))pending=.5f;
  while(*q){if(*q!='\''&&*q!=',')*v++=*q;q++;}*v=0;if(!*token)continue;
  if(!strcmp(token,"r*")){if(i+1>=tn||strcmp(tokens[i+1],"*"))return -4;continue;}
  if(!strcmp(token,";")){
   char core[64];if(!n||finnic_core(core,phones[n-1].phone->name)||!phones[n-1].phone->features[31]||strcmp(core,"i"))return -4;
   continue;
  }
  if(!strcmp(token,":")){
   if(!n||!phones[n-1].phone->features[31])return -4;
   if(!strstr(phones[n-1].phone->name,"ː"))phones[n-1].phone=finnic_suffix(phones[n-1].phone,"ː");
   if(!phones[n-1].phone)return -4;continue;
  }
  unsigned len=strlen(token);int is_long=len&&token[len-1]==':';if(is_long)token[--len]=0;
  const char*name=NULL;
  for(unsigned j=0;j<sizeof(croatian_names)/sizeof(croatian_names[0]);j++)if(!strcmp(token,croatian_names[j].name)){name=croatian_names[j].ipa;break;}
  if(!name||n>=capacity)return -4;const N25FourPhone*phone=phone_by_name(name);
  if(phone&&is_long)phone=finnic_suffix(phone,"ː");if(!phone)return -4;
  phones[n++]=(FourPhone){phone,phone->features[31]?pending:0,0};
  if(phone->features[31]){pending=0;nuclei++;}
 }
 if(!n||pending||(!nuclei&&strcmp(word,"s")&&strcmp(word,"k")))return -4;
 *count=n;return 0;
}
