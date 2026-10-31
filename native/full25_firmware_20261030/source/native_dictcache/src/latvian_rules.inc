/* Latvian branch of frozen eu_baltic._parse_backend and syllable context. */
#include "latvian_alphabet.h"
static int latvian_core(char*out,const char*name){
 const unsigned marks[]={0x2d0,0x2d1,0x2b2,0x2e0,0x2b0,0x303,0x329,0x32f,0x32a,0x325,0x32c,0x306,0x2de,0x31d,0x31e,0x31f,0x320};
 const unsigned char*p=(const unsigned char*)name;unsigned n=0;
 while(*p){uint32_t cp;if(get_cp(&p,&cp))return -1;int skip=0;
  for(unsigned i=0;i<sizeof(marks)/sizeof(marks[0]);i++)if(cp==marks[i])skip=1;
  if(skip)continue;if(n+4>=64)return -1;n+=put_cp(out+n,cp);
 }out[n]=0;return 0;
}
static int append_latvian(FourPhone*phones,unsigned*count,unsigned capacity,const char*token,float*pending,const char*word){
 if(!*token)return 0;
 if(!strcmp(token,"ʐ")&&strstr(word,"dž")&&*count&&!strcmp(phones[*count-1].phone->name,"d")){
  phones[*count-1].phone=phone_by_name("dʒ");return phones[*count-1].phone?0:-4;
 }
 if(!strcmp(token,"a`"))token="a";else if(!strcmp(token,"o`"))token="o";else if(!strcmp(token,"z-"))token="z";
 const N25FourPhone*p=phone_by_name(token);if(!p)return -4;
 if(*count&&!p->features[31]&&phones[*count-1].phone==p){
  phones[*count-1].phone=finnic_suffix(p,"ː");return phones[*count-1].phone?0:-4;
 }
 if(*count>=capacity)return -4;
 phones[(*count)++]=(FourPhone){p,p->features[31]?*pending:0,0};if(p->features[31])*pending=0;return 0;
}
static int parse_latvian(const char*raw,const char*word,FourPhone*phones,unsigned*count,unsigned capacity){
 char token[64];unsigned used=0;float pending=0;uint32_t previous=0;*count=0;
 const unsigned char*p=(const unsigned char*)raw;
 while(*p){uint32_t cp;if(get_cp(&p,&cp))return -3;
  if(cp==0x2c8||cp==0x2cc){pending=cp==0x2c8?1.f:.5f;continue;}
  if(cp==0x361||cp==0x35c)continue;if(cp=='g')cp=0x261;if(cp==0x30a)cp=0x325;
  if((cp==0x2d0||cp==0x2b2)&&cp==previous)continue;
  if(cp=='_'||cp==' '||cp=='\t'||cp=='\n'){
   token[used]=0;int rc=append_latvian(phones,count,capacity,token,&pending,word);if(rc)return rc;used=0;previous=0;continue;
  }
  if(used+4>=sizeof(token))return -3;used+=put_cp(token+used,cp);previous=cp;
 }
 token[used]=0;int rc=append_latvian(phones,count,capacity,token,&pending,word);if(rc||pending||!*count)return -4;
 for(unsigned i=0;i<*count;i++)if(phones[i].phone->features[31])return 0;return -4;
}
static int latvian_sonorant(const char*name){
 char core[64];if(latvian_core(core,name))return 0;
 static const char*values[]={"m","n","ŋ","ɲ","l","ɫ","ʎ","r","ɾ"};
 for(unsigned i=0;i<sizeof(values)/sizeof(values[0]);i++)if(!strcmp(core,values[i]))return 1;return 0;
}
