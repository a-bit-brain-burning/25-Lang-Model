/* Port of frozen eu_finnic._parse_native; quantity remains explicitly unknown. */
#include "finnic_names.h"
static int finnic_core(char*out,const char*name){
 const unsigned char*p=(const unsigned char*)name;unsigned n=0;
 while(*p){uint32_t cp;if(get_cp(&p,&cp))return -1;
  if(cp==0x2d0||cp==0x2d1||cp==0x2b2)continue;
  if(n+4>=64)return -1;n+=put_cp(out+n,cp);
 }out[n]=0;return 0;
}
static const N25FourPhone* finnic_suffix(const N25FourPhone*p,const char*suffix){
 char name[64];size_t n=strlen(p->name),m=strlen(suffix);
 if(n+m>=sizeof(name))return NULL;memcpy(name,p->name,n);memcpy(name+n,suffix,m+1);return phone_by_name(name);
}
static int parse_finnic(char*raw,const char*word,FourPhone*phones,unsigned*count,unsigned capacity,int estonian){
 char*tokens[512];unsigned tn=0;char*p=raw;
 while(*p){while(*p=='_'||*p==' '||*p=='\t'||*p=='\n')p++;if(!*p)break;
  if(tn==512)return -4;tokens[tn++]=p;
  while(*p&&*p!='_'&&*p!=' '&&*p!='\t'&&*p!='\n')p++;if(*p)*p++=0;
 }
 const char*original[512];unsigned n=0;float pending=0;
 for(unsigned i=0;i<tn;i++){
  char*token=tokens[i],*q=token,*v=token;
  if(strchr(token,'\''))pending=1;else if(strchr(token,','))pending=.5f;
  while(*q){if(*q!='\''&&*q!=',')*v++=*q;q++;}*v=0;
  if(!*token)continue;
  if(!estonian&&!strcmp(token,";")){if(!n||!phones[n-1].phone->features[31])return -4;continue;}
  if(estonian&&!strcmp(token,"i1")){
   if(i+1>=tn)return -4;const char*next=tokens[i+1];unsigned len=strlen(next);
   if(len&&next[len-1]==':')len--;if(!len||next[len-1]!='^')return -4;continue;
  }
  if(!strcmp(token,":")){
   if(!n||!phones[n-1].phone->features[31])return -4;
   if(!strstr(phones[n-1].phone->name,"ː"))phones[n-1].phone=finnic_suffix(phones[n-1].phone,"ː");
   if(!phones[n-1].phone)return -4;continue;
  }
  unsigned len=strlen(token);int is_long=len&&token[len-1]==':';if(is_long)token[--len]=0;
  const char*name=NULL;
  for(unsigned j=0;j<sizeof(finnic_names)/sizeof(finnic_names[0]);j++)if(!strcmp(token,finnic_names[j].name)){
   name=estonian?finnic_names[j].et:finnic_names[j].fi;break;
  }
  if(!name||n>=capacity)return -4;const N25FourPhone*phone=phone_by_name(name);
  if(phone&&is_long)phone=finnic_suffix(phone,"ː");if(!phone)return -4;
  phones[n]=(FourPhone){phone,phone->features[31]?pending:0,0};original[n++]=token;
  if(phone->features[31])pending=0;
 }
 if(!n||pending)return -4;
 if(estonian)for(unsigned i=1;i+1<n;i++){
  const char*name=original[i];char previous[64];if(finnic_core(previous,phones[i-1].phone->name))return -4;
  if((!strcmp(name,"p")||!strcmp(name,"t")||!strcmp(name,"k")||!strcmp(name,"t^")||!strcmp(name,"f")||!strcmp(name,"S"))&&
     !strstr(phones[i].phone->name,"ː")&&phones[i+1].phone->features[31]&&
     (phones[i-1].phone->features[31]||!strcmp(previous,"m")||!strcmp(previous,"n")||!strcmp(previous,"ŋ")||!strcmp(previous,"l")||!strcmp(previous,"r")||!strcmp(previous,"v"))){
   phones[i].phone=finnic_suffix(phones[i].phone,"ˑ");if(!phones[i].phone)return -4;
  }
 }
 unsigned used=0;
 for(unsigned i=0;i<n;i++){
  FourPhone a=phones[i];
  if(i+1<n){FourPhone b=phones[i+1];char ac[64],bc[64],target[64];
   if(finnic_core(ac,a.phone->name)||finnic_core(bc,b.phone->name))return -4;
   int same=!strcmp(ac,bc);target[0]=0;int merge=0;
   if(same&&!a.phone->features[31]&&!b.phone->features[31]){
    if(strlen(ac)+6>=sizeof(target))return -4;strcpy(target,ac);
    if(strstr(a.phone->name,"ʲ")||strstr(b.phone->name,"ʲ"))strcat(target,"ʲ");strcat(target,"ː");merge=1;
   }else if(estonian&&a.phone->features[31]&&b.phone->features[31]){
    const unsigned char*cur=(const unsigned char*)ac;uint32_t cp;if(get_cp(&cur,&cp))return -4;
    if(same&&!*cur&&(a.phone->features[22]||b.phone->features[22])){
     strcpy(target,ac);strcat(target,"ː");if(b.stress>a.stress)a.stress=b.stress;merge=1;
    }else if(strlen(ac)+strlen(bc)<sizeof(target)){
     strcpy(target,ac);strcat(target,bc);int allowed=0;
     if(a.stress==1)for(unsigned j=0;j<sizeof(et_diphthongs)/sizeof(et_diphthongs[0]);j++)if(!strcmp(target,et_diphthongs[j]))allowed=1;
     else (void)0;
     if(a.stress!=1)allowed=!strcmp(target,"ɑi")||!strcmp(target,"ei")||!strcmp(target,"ui");
     char spelling[64];unsigned k=0;cur=(const unsigned char*)target;
     while(*cur){if(get_cp(&cur,&cp))return -4;if(cp==0x251)cp='a';else if(cp==0x264)cp=0xf5;else if(cp==0xe6)cp=0xe4;else if(cp==0xf8)cp=0xf6;else if(cp=='y')cp=0xfc;k+=put_cp(spelling+k,cp);}spelling[k]=0;
     merge=allowed&&!b.stress&&strstr(word,spelling);
    }
   }
   if(merge){a.phone=phone_by_name(target);if(!a.phone)return -4;i++;}
  }
  phones[used++]=a;
 }
 unsigned first=used;
 for(unsigned i=0;i<used;i++)if(phones[i].phone->features[31]){first=i;break;}
 if(first==used)return -4;
 if(!estonian){phones[first].stress=1;phones[first].known=1;
  for(unsigned i=first+1;i<used;i++)if(phones[i].stress==1)phones[i].stress=.5f;
 }
 *count=used;return 0;
}
