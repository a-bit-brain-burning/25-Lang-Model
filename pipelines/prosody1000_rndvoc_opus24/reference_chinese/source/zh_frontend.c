#if defined(__GNUC__)
#pragma GCC optimize("O3")
#endif
/* Mandarin text frontend. No input audio, timings, features or predicted plans. */
#include "zh_frontend.h"
#include "zh_tables.h"
#include "unicode_native.h"
#include <string.h>
#include <math.h>
#include "generated/zh_properties.h"
#include "generated/zh_vocabulary.h"
#define MAX_HAN 128
typedef struct{uint16_t root;uint8_t tone;} Reading;
typedef struct{int start,end,tag;} Word;
typedef struct{unsigned char body[512];const unsigned char*main,*supp,*correct,*surname,*erhua;uint32_t frequency;int tag,flags;size_t used;} Record;
typedef struct{int start,end,first,stop,group,level;Reading raw,lexical,surface;int citation,known,hint;} Syllable;
typedef struct{
 uint32_t cp[MAX_HAN];int count,levels[MAX_HAN],run_end[MAX_HAN],question_end[MAX_HAN],question_kind[MAX_HAN],endpoints;
 Reading primary[MAX_HAN],reading[MAX_HAN],hint[MAX_HAN];uint8_t resolved[MAX_HAN],protected[MAX_HAN],fused[MAX_HAN];
 Word context[MAX_HAN],groups[MAX_HAN];int nc,ng,context_at[MAX_HAN],previous_tag[MAX_HAN],route[MAX_HAN+1];double score[MAX_HAN+1];
 Syllable syllables[MAX_HAN];int ns,n;uint8_t comma[256],pair_final[MAX_HAN];int32_t pairs[MAX_HAN][4];int np;
} ZhState;
static ZhState state __attribute__((section(".lexcache"),aligned(8)));static int ready;
static uint16_t memo_index[MAX_HAN][ZH_MAX_WORD+1] __attribute__((section(".lexcache")));
static unsigned char memo_body[16384] __attribute__((section(".lexcache")));static unsigned memo_used;
static const float contours[6][3]={{0,0,0},{1,1,1},{-.5f,0,1},{-.5f,-1,.5f},{1,0,-1},{0,0,0}};
static int variable(const unsigned char**p,const unsigned char*end,uint32_t*v){return native_zh_varint(p,end,v);}
static uint16_t u16(const unsigned char*p){return(uint16_t)(p[0]|(uint16_t)p[1]<<8);}
static int same_root(Reading a,Reading b){return a.root==b.root;}
static const char*root(Reading p){return p.root<ZH_ROOT_COUNT?zh_roots[p.root].name:"";}
static int root_id(const char*s){unsigned lo=0,hi=ZH_ROOT_COUNT;while(lo<hi){unsigned m=lo+(hi-lo)/2;if(strcmp(zh_roots[m].name,s)<0)lo=m+1;else hi=m;}return lo<ZH_ROOT_COUNT&&!strcmp(zh_roots[lo].name,s)?(int)lo:-1;}
static Reading from_id(unsigned id){Reading r={65535,0};if(id<ZH_TOKEN_COUNT){r.root=zh_tokens[id].root;r.tone=zh_tokens[id].tone;}return r;}
static int valid(Reading r){return r.root<ZH_ROOT_COUNT&&r.tone>=1&&r.tone<=5&&zh_roots[r.root].n>0;}
static int is_han(uint32_t c){return c>=0x3400&&c<=0x9fff;}
static int whitespace(uint32_t c){return(c>=9&&c<=13)||(c>=28&&c<=32)||c==0x85||c==0xa0||c==0x1680||(c>=0x2000&&c<=0x200a)||c==0x2028||c==0x2029||c==0x202f||c==0x205f||c==0x3000;}
static int set_has(uint32_t c,const char*s){while(*s){uint32_t v;if(native_utf8_read(&s,&v)<1)return 0;if(v==c)return 1;}return 0;}
static int equal(int a,int b,const char*s){while(*s&&a<b){uint32_t c;if(native_utf8_read(&s,&c)<1||state.cp[a++]!=c)return 0;}return !*s&&a==b;}
static int one(int a,const char*s){return equal(a,a+1,s);}
static int tag_prefix(int id,const char*letters){return id>0&&id<(int)(sizeof(zh_tags)/sizeof(*zh_tags))&&strchr(letters,zh_tags[id][0])!=NULL;}
static int tag_equal(int id,const char*s){return id>=0&&id<(int)(sizeof(zh_tags)/sizeof(*zh_tags))&&!strcmp(zh_tags[id],s);}
static int decode_hint(const char*p,const char**end,Reading*out){char token[32];int n=0;while(*p&&*p!=']'){uint32_t c;if(native_utf8_read(&p,&c)<1)return 0;if(c>='A'&&c<='Z')c+=32;if(c==0xfc)c='v';if(c==':'&&n&&token[n-1]=='u'){token[n-1]='v';continue;}if(c>127||n==31)return 0;token[n++]=(char)c;}if(*p!=']'||n<2||token[n-1]<'0'||token[n-1]>'5')return 0;unsigned tone=token[--n]-'0';token[n]=0;int id=root_id(token);if(id<0)return 0;out->root=id;out->tone=tone?tone:5;*end=p+1;return 1;}
static int parse_text(const char*text){memset(&state,0,sizeof(state));int run=0;for(const char*p=text;*p;){uint32_t c;if(native_utf8_read(&p,&c)<1)return -1;if(c=='#'&&*p>='1'&&*p<='4'){p++;continue;}
  if(is_han(c)){if(state.count==MAX_HAN)return -2;int at=state.count++;state.cp[at]=c;if(*p=='['){const char*end;Reading hint;if(decode_hint(p+1,&end,&hint)){state.hint[at]=hint;p=end;}}run=1;continue;}
  if(state.count&&run)state.run_end[state.count-1]=1;run=0;if(!whitespace(c)&&!set_has(c,"，。！？；：、,.!?;:…—“”‘’（）()《》〈〉【】[]「」『』·"))return -4;if(!state.count)continue;
  int level=set_has(c,"。！？.!?")?4:set_has(c,"；：;:")?3:set_has(c,"，、,")?2:0;if(level>state.levels[state.count-1])state.levels[state.count-1]=level;
  if(set_has(c,"。.!?！？；;")){if(state.endpoints==MAX_HAN)return -2;int i=state.endpoints++;state.question_end[i]=state.count;state.question_kind[i]=set_has(c,"?？")?1:set_has(c,"!！")?2:0;}
  if(set_has(c,"，,"))state.comma[state.count-1]=1;
 }if(!state.count)return -4;state.run_end[state.count-1]=1;return 0;}
static int skip_delta(const unsigned char**p,const unsigned char*end){uint32_t n,pos;if(variable(p,end,&n)||n>128)return -1;for(unsigned i=0;i<n;i++){if(variable(p,end,&pos)||end-*p<2)return -1;*p+=2;}return 0;}
static int record(int a,int b,Record*r,int*prefix){size_t used=0;int rc=1,has_prefix=0;int length=b-a;
 if(length>=1&&length<=ZH_MAX_WORD&&a>=0&&b<=state.count){uint16_t cached=memo_index[a][length];
  if(cached==65535||cached==65534){has_prefix=cached==65534;}
  else if(cached){unsigned offset=cached-1;used=u16(memo_body+offset);if(offset+2+used>memo_used||used>sizeof(r->body))return -1;memcpy(r->body,memo_body+offset+2,used);has_prefix=1;rc=0;}
  else{rc=native_zh_lookup(0,state.cp+a,length,r->body,sizeof(r->body),&used,&has_prefix);if(rc==1)memo_index[a][length]=has_prefix?65534:65535;else if(!rc&&memo_used+2+used<=sizeof(memo_body)){unsigned offset=memo_used;memo_body[offset]=used&255;memo_body[offset+1]=used>>8;memcpy(memo_body+offset+2,r->body,used);memo_used+=2+used;memo_index[a][length]=offset+1;}}
 }
 if(prefix)*prefix=has_prefix;r->main=r->supp=r->correct=r->surname=r->erhua=NULL;r->frequency=0;r->tag=r->flags=0;r->used=used;if(rc)return rc;if(!used)return -1;const unsigned char*p=r->body,*end=p+used;r->flags=*p++;uint32_t v;
 if(r->flags&1){if(variable(&p,end,&r->frequency)||p>=end)return -1;r->tag=*p++;}
 const unsigned char**fields[]={&r->main,&r->supp,&r->correct};for(int i=0;i<3;i++)if(r->flags&(2<<i)){*fields[i]=p;if(skip_delta(&p,end))return -1;}
 if(r->flags&16){r->surname=p;if(variable(&p,end,&v)||v>256)return -1;for(unsigned i=0;i<v;i++)if(skip_delta(&p,end))return -1;}
 if(r->flags&32){r->erhua=p;if(variable(&p,end,&v)||v>128)return -1;for(unsigned i=0;i<v;i++){uint32_t x;if(variable(&p,end,&x)||variable(&p,end,&x)||end-p<2)return -1;p+=2;}}
 return p==end?0:-1;
}
static int read_delta(const Record*r,const unsigned char*p,int a,int b,Reading*out){for(int i=a;i<b;i++)out[i-a]=state.primary[i];uint32_t n,pos;const unsigned char*end=r->body+r->used;if(variable(&p,end,&n)||n>(unsigned)(b-a))return -1;for(unsigned i=0;i<n;i++){if(variable(&p,end,&pos)||pos>=(unsigned)(b-a)||end-p<2)return -1;out[pos]=from_id(u16(p));p+=2;}return 0;}
static int character(int at,unsigned char*body,size_t cap,size_t*used){return native_zh_lookup(1,state.cp+at,1,body,cap,used,NULL);}
static int load_primary(void){for(int i=0;i<state.count;i++){unsigned char body[256];size_t used;int rc=character(i,body,sizeof(body),&used);if(rc)return -5;const unsigned char*p=body,*end=p+used;uint32_t count;if(variable(&p,end,&count)||!count||end-p<(int)(count*2))return -5;state.primary[i]=from_id(u16(p));if(!valid(state.primary[i]))return -5;state.reading[i]=state.primary[i];}return 0;}
static int segment(int a,int b,Word*out,int*count){state.score[b]=0;state.route[b]=b;double total=log(ZH_TOTAL_FREQUENCY);for(int i=b-1;i>=a;i--){double best=-HUGE_VAL;int choice=i+1,found=0;for(int end=i+1;end<=b&&end-i<=ZH_MAX_WORD;end++){Record r;int prefix=0,rc=record(i,end,&r,&prefix);if(rc<0)return rc;if(!rc&&r.frequency){double value=log((double)r.frequency)-total+state.score[end];if(value>best||(value==best&&end>choice)){best=value;choice=end;}found=1;}if(!prefix)break;}if(!found){best=-total+state.score[i+1];choice=i+1;}state.score[i]=best;state.route[i]=choice;}
 for(int i=a;i<b;){if(*count==MAX_HAN)return -2;Word*w=&out[(*count)++];w->start=i;w->end=state.route[i];Record r;int rc=record(i,w->end,&r,NULL);if(rc<0)return rc;w->tag=rc?0:r.tag;i=w->end;}return 0;}
static int build_context(void){for(int a=0;a<state.count;){int b=a+1;while(b<state.count&&!state.run_end[b-1])b++;int start=state.nc,rc=segment(a,b,state.context,&state.nc);if(rc)return rc;int previous=0;for(int wi=start;wi<state.nc;wi++){Word*w=&state.context[wi];for(int i=w->start;i<w->end;i++){state.context_at[i]=wi;state.previous_tag[i]=previous;}previous=w->tag;}a=b;}return 0;}
static int longest_main(int a,int b){int best=a+1;for(int end=a+1;end<=b&&end-a<=ZH_MAX_WORD;end++){Record r;int prefix=0,rc=record(a,end,&r,&prefix);if(rc<0)return rc;if(!rc&&r.main)best=end;if(!prefix)break;}return best;}
static int resolve_readings(void){
 for(int start=0;start<state.count;){int stop=start+1;while(stop<state.count&&!state.run_end[stop-1])stop++;for(int at=start;at<stop;){int end=longest_main(at,stop);if(end<0)return end;if(end>at+1)for(int i=at;i<end;i++)state.protected[i]=1;at=end;}
  for(int at=start;at<stop;){int end=longest_main(at,stop);if(end<0)return end;if(!state.protected[at]&&state.context[state.context_at[at]].start==at){for(int last=at+2;last<=stop&&last-at<=ZH_MAX_WORD;last++){int protect=0;for(int i=at;i<last;i++)protect|=state.protected[i];if(protect)break;Record r;int prefix=0,rc=record(at,last,&r,&prefix);if(rc<0)return rc;if(!rc&&r.supp&&state.context[state.context_at[last-1]].end==last)end=last;if(!prefix)break;}}
   Record r;int rc=record(at,end,&r,NULL);if(rc<0)return rc;Reading original[MAX_HAN],values[MAX_HAN];for(int i=at;i<end;i++)original[i-at]=state.primary[i];if(!rc&&r.main&&read_delta(&r,r.main,at,end,original))return -5;memcpy(values,original,(size_t)(end-at)*sizeof(*values));if(!rc&&r.supp){if(read_delta(&r,r.supp,at,end,values))return -5;for(int i=at;i<end;i++)if(one(i,"得")&&!strcmp(root(values[i-at]),"de")&&values[i-at].tone==5&&(!same_root(original[i-at],values[i-at])||original[i-at].tone!=5)&&!tag_prefix(state.previous_tag[i],"va"))values[i-at]=original[i-at];for(int i=at;i<end;i++)state.resolved[i]=1;}
   for(int i=at;i<end;i++)state.reading[i]=values[i-at];at=end;
  }start=stop;
 }
 for(int wi=0;wi<state.nc;wi++){Word*w=&state.context[wi];Record r;int rc=record(w->start,w->end,&r,NULL);if(rc<0)return rc;if(!rc&&r.correct){Reading values[MAX_HAN];if(read_delta(&r,r.correct,w->start,w->end,values))return -5;for(int i=w->start;i<w->end;i++){state.reading[i]=values[i-w->start];state.resolved[i]=1;}}}
 for(int at=0;at<state.count;at++){if(!one(at,"姓")||state.context[state.context_at[at]].start!=at)continue;int stop=at+1;while(stop<state.count&&!state.run_end[stop-1])stop++;int a=at+1,best=0;Reading best_values[MAX_HAN];for(int end=a+1;end<=stop&&end-a<=ZH_MAX_WORD;end++){Record r;int rc=record(a,end,&r,NULL);if(rc<0)return rc;if(rc||!r.surname)continue;const unsigned char*p=r.surname,*limit=r.body+sizeof(r.body);uint32_t options;if(variable(&p,limit,&options))return -5;if(options!=1)continue;if(end!=stop){Word*w=&state.context[state.context_at[end]];if(w->start!=end||w->end!=end+1||!one(end,"的")||!(tag_equal(w->tag,"uj")||tag_equal(w->tag,"u")||tag_equal(w->tag,"uz")))continue;}if(read_delta(&r,p,a,end,best_values))return -5;best=end;}
  if(best)for(int i=a;i<best;i++){state.reading[i]=best_values[i-a];state.resolved[i]=1;}
 }
 for(int i=0;i<state.count;i++)if(state.hint[i].tone)state.reading[i]=state.hint[i];return 0;
}
static int fuse_erhua(void){for(int wi=0;wi<state.nc;wi++){Word*w=&state.context[wi];Record r;int rc=record(w->start,w->end,&r,NULL);if(rc<0)return rc;if(rc||!r.erhua)continue;const unsigned char*p=r.erhua,*end=r.body+sizeof(r.body);uint32_t n;if(variable(&p,end,&n)||n>MAX_HAN)return -5;int indices[MAX_HAN],roots[MAX_HAN],accepted=1;
  for(unsigned j=0;j<n;j++){uint32_t a,b;if(variable(&p,end,&a)||variable(&p,end,&b)||end-p<2||b!=a+2||b>(unsigned)(w->end-w->start))return -5;Reading identity=from_id(u16(p));p+=2;int at=w->start+a;indices[j]=at;char rhotic[32];const char*name=root(state.reading[at]);if(strlen(name)+1>=sizeof(rhotic))return -5;strcpy(rhotic,name);strcat(rhotic,"r");roots[j]=root_id(rhotic);if(!one(at+1,"儿")||!same_root(state.reading[at],identity)||strcmp(root(state.reading[at+1]),"er")||state.levels[at]||state.fused[at]||state.fused[at+1]||roots[j]<0)accepted=0;}
  if(accepted)for(unsigned j=0;j<n;j++){int at=indices[j];state.fused[at]=1;state.reading[at].root=roots[j];}
 }return 0;}
static int build_syllables(void){for(int a=0;a<state.count;){int b=a+1;while(b<state.count&&state.levels[b-1]<2)b++;int rc=segment(a,b,state.groups,&state.ng);if(rc)return rc;a=b;}
 Word merged[MAX_HAN];int ng=0,start=0;for(int i=0;i<state.ng;i++){int end=state.groups[i].end;if(end<state.count&&state.fused[end-1])continue;Word*w=&merged[ng++];w->start=start;w->end=end;Record r;int rc=record(start,end,&r,NULL);if(rc<0)return rc;w->tag=rc?0:r.tag;start=end;}memcpy(state.groups,merged,ng*sizeof(*merged));state.ng=ng;
 for(int i=0;i<state.count;){Syllable*s=&state.syllables[state.ns++];s->start=i;s->end=i+(state.fused[i]?2:1);s->raw=s->lexical=s->surface=state.reading[i];s->level=state.levels[s->end-1];s->hint=state.hint[i].tone>0;for(int wi=0;wi<state.ng;wi++)if(i>=state.groups[wi].start&&i<state.groups[wi].end)s->group=wi;if(!valid(s->raw))return -5;i=s->end;}return 0;
}
static int word_in(const Word*w,const char*const*set,unsigned n){for(unsigned i=0;i<n;i++)if(equal(w->start,w->end,set[i]))return 1;return 0;}
static const char*const kinship[]={"妈妈","爸爸","哥哥","姐姐","弟弟","妹妹","爷爷","奶奶","叔叔","舅舅","姑姑","伯伯","婶婶"};
static const char*const aspect[]={"看过","听过","读过","学过","见过","说过","喝过","吃过","去过","来过","用过","做过","买过","写过","玩过","住过","试过","睡过"};
static Word*group_word(int si){return &state.groups[state.syllables[si].group];}
static int single_character(int si,const char*s){Syllable*v=&state.syllables[si];return equal(v->start,v->end,s);}
static int adverbial_context(int i){if(!i||i+1>=state.ns||state.syllables[i-1].level||state.syllables[i].level)return 0;Word*previous=group_word(i-1),*context=&state.context[state.context_at[state.syllables[i].start]];int width=previous->end-previous->start;if(context->end-context->start>1&&one(context->end-1,"地")&&tag_prefix(context->tag,"da"))return 1;if(!tag_equal(context->tag,"uv"))return 0;int modifier=tag_prefix(previous->tag,"adizo")||(tag_prefix(previous->tag,"v")&&width>1)||(width>=2&&state.cp[previous->end-1]==state.cp[previous->end-2]);if(!modifier)return 0;int seen=-1;
 for(int j=i+1;j<state.ns&&j<i+6;j++){if(j>i+1&&state.syllables[j-1].level)return 0;Syllable*s=&state.syllables[j];if(s->group==seen)continue;seen=s->group;Word*w=&state.context[state.context_at[s->start]];if(tag_prefix(w->tag,"va"))return 1;if(!tag_prefix(w->tag,"dm"))return 0;}return 0;
}
static int modal_context(int i){if(i&&state.syllables[i-1].level==0){int j=i-1,seen=-1,licensed=0;while(j>=0&&i-j<=8){Syllable*s=&state.syllables[j];if(s->group==seen){j--;continue;}seen=s->group;Word*w=group_word(j);if((tag_prefix(w->tag,"r")&&set_has(state.cp[w->start],"我你他她它咱您"))||tag_prefix(w->tag,"t")){licensed=1;break;}if(!tag_prefix(w->tag,"d"))break;int first=j;while(first&&state.syllables[first-1].group==s->group)first--;if(!first){licensed=1;break;}j=first-1;if(state.syllables[j].level){licensed=1;break;}}if(!licensed)return 0;}
 int seen=-1;for(int j=i+1;j<state.ns&&j<i+6;j++){if(state.syllables[j-1].level)return 0;Syllable*s=&state.syllables[j];if(seen==s->group)continue;seen=s->group;Word*w=group_word(j);if(equal(w->start,w->end,"把")||equal(w->start,w->end,"被")||tag_prefix(w->tag,"va"))return 1;if(!tag_prefix(w->tag,"d"))return 0;}return 0;
}
static int assign_reading(Syllable*s,const char*name,int tone){int id=root_id(name);if(id<0)return -5;s->lexical.root=id;s->lexical.tone=tone;return 0;}
static int neutral_policy(void){for(int i=0;i<state.ns;i++){Syllable*s=&state.syllables[i];Word*w=group_word(i);const char*name=root(s->lexical);int final=s->end==w->end,width=w->end-w->start,rc=0;
  if(final&&word_in(w,kinship,sizeof(kinship)/sizeof(*kinship)))s->lexical.tone=5;
  else if(final&&width==2&&state.cp[w->start]==state.cp[w->start+1]&&tag_prefix(w->tag,"v"))s->lexical.tone=5;
  else if(single_character(i,"地")&&equal(w->start,w->end,"地")&&tag_equal(w->tag,"uv")&&(!strcmp(name,"de")||adverbial_context(i)))rc=assign_reading(s,"de",5);
  else if(single_character(i,"的")&&equal(w->start,w->end,"的")&&!strcmp(name,"de"))s->lexical.tone=5;
  else if(single_character(i,"得")&&equal(w->start,w->end,"得")&&!strcmp(name,"de")){int continuous=i&&state.syllables[i-1].level==0;if(continuous&&tag_prefix(group_word(i-1)->tag,"va"))s->lexical.tone=5;else if(continuous&&set_has(state.cp[state.syllables[i-1].start],"我你他她它咱您")&&i+1<state.ns&&tag_prefix(group_word(i+1)->tag,"v"))rc=assign_reading(s,"dei",3);else if(modal_context(i))rc=assign_reading(s,"dei",3);}
  else if(single_character(i,"过")&&word_in(w,aspect,sizeof(aspect)/sizeof(*aspect))&&!strcmp(name,"guo"))s->lexical.tone=5;
  else if((single_character(i,"了")||single_character(i,"着"))&&(!strcmp(name,"le")||!strcmp(name,"zhe")))s->lexical.tone=5;
  else if(s->end-s->start==1&&set_has(state.cp[s->start],"吗呢吧啊嘛啦")&&width==1&&(i==state.ns-1||s->level>=2))s->lexical.tone=5;
  else if(single_character(i,"们")&&!strcmp(name,"men")&&(s->lexical.tone==5||tag_equal(w->tag,"k")||equal(w->start,w->end,"我们")||equal(w->start,w->end,"你们")||equal(w->start,w->end,"他们")||equal(w->start,w->end,"她们")||equal(w->start,w->end,"它们")||equal(w->start,w->end,"咱们")||equal(w->start,w->end,"人们")))s->lexical.tone=5;
  if(rc)return rc;if(s->hint)s->lexical=s->raw;
 }return 0;}
static int citation_policy(void){for(int i=0;i<state.ns;i++){Syllable*s=&state.syllables[i];Word*w=group_word(i);s->citation=0;s->known=0;char a[32],b[32];strcpy(a,root(s->lexical));strcpy(b,root(s->raw));size_t an=strlen(a),bn=strlen(b);if(an&&a[an-1]=='r'&&strcmp(a,"er"))a[an-1]=0;if(bn&&b[bn-1]=='r'&&strcmp(b,"er"))b[bn-1]=0;
  if(strcmp(a,b)){if((single_character(i,"地")||single_character(i,"得"))&&((!strcmp(a,"de")&&s->lexical.tone==5)||(!strcmp(a,"dei")&&s->lexical.tone==3))){s->citation=s->lexical.tone;s->known=1;}}
  else if(single_character(i,"一")&&!strcmp(a,"yi")){s->citation=1;s->known=1;}
  else if(single_character(i,"不")&&!strcmp(a,"bu")){s->citation=4;s->known=1;}
  else if(s->lexical.tone==5){s->citation=5;s->known=1;}
  else{unsigned char body[256];size_t used;int rc=character(s->start,body,sizeof(body),&used);if(rc)return -5;const unsigned char*p=body,*end=p+used;uint32_t count;if(variable(&p,end,&count)||end-p<(int)(count*2))return -5;unsigned tones=0;for(unsigned j=0;j<count;j++){Reading r=from_id(u16(p+j*2));if(!strcmp(root(r),a)&&r.tone>=1&&r.tone<=5)tones|=1u<<r.tone;}if(tones&&!(tones&(tones-1))){s->known=1;for(int t=1;t<=5;t++)if(tones&(1u<<t))s->citation=t;}else{Record r;rc=record(w->start,w->end,&r,NULL);if(rc<0)return rc;if(!rc&&r.main){s->known=1;s->citation=s->lexical.tone;}}}
  if((state.resolved[s->start]&&same_root(s->lexical,s->raw)&&s->lexical.tone==s->raw.tone)||s->hint){s->known=1;s->citation=s->lexical.tone;}
 }return 0;}
static void surface_policy(void){for(int i=0;i<state.ns;i++){Syllable*s=&state.syllables[i];s->surface=s->lexical;int next=i+1<state.ns&&s->level==0?state.syllables[i+1].lexical.tone:0;
  if((single_character(i,"一")||single_character(i,"不"))&&i&&i+1<state.ns&&state.syllables[i-1].end-state.syllables[i-1].start==1&&state.syllables[i+1].end-state.syllables[i+1].start==1&&state.cp[state.syllables[i-1].start]==state.cp[state.syllables[i+1].start]&&state.syllables[i-1].level==0&&s->level==0)s->surface.tone=5;
  else if(single_character(i,"一")&&!strcmp(root(s->lexical),"yi")){int numerical=(i&&state.syllables[i-1].end-state.syllables[i-1].start==1&&set_has(state.cp[state.syllables[i-1].start],"第零一二三四五六七八九十百千万亿"))||(i+1<state.ns&&state.syllables[i+1].end-state.syllables[i+1].start==1&&set_has(state.cp[state.syllables[i+1].start],"零一二三四五六七八九十百千万亿"));s->surface.tone=numerical||!next?1:next==4?2:next<4?4:1;}
  else if(single_character(i,"不")&&!strcmp(root(s->lexical),"bu"))s->surface.tone=next==4?2:4;
 }
 for(int i=0;i<state.ns;){if(state.syllables[i].lexical.tone!=3){i++;continue;}int end=i+1;while(end<state.ns&&state.syllables[end].lexical.tone==3&&state.syllables[end-1].level==0)end++;if(end-i>=2)state.syllables[end-2].surface.tone=2;i=end;}
 for(int i=0;i+1<state.ns;i++){Syllable*a=&state.syllables[i],*b=a+1;Word*w=group_word(i);if(a->group!=b->group||a->end-a->start!=1||b->end-b->start!=1||w->start!=a->start||w->end!=b->end||!word_in(w,kinship,sizeof(kinship)/sizeof(*kinship))||a->level||a->hint||b->hint||!same_root(a->surface,b->surface))continue;b->surface.tone=5;if(a->surface.tone==2&&a->known&&a->citation==3)a->surface.tone=3;}
}
static int vocabulary(uint32_t cp){unsigned lo=0,hi=sizeof(zh_vocabulary)/sizeof(*zh_vocabulary);while(lo<hi){unsigned m=lo+(hi-lo)/2;if(zh_vocabulary[m].cp<cp)lo=m+1;else hi=m;}return lo<sizeof(zh_vocabulary)/sizeof(*zh_vocabulary)&&zh_vocabulary[lo].cp==cp?zh_vocabulary[lo].id:1;}
void native_zh_reset(void){ready=0;}
int native_zh_encode(const Acoustic*m,const char*text,float*x,int*n,int*w,float*numeric,int32_t*owner,int32_t*characters,uint8_t*edge){
 ready=0;memset(memo_index,0,sizeof(memo_index));memo_used=0;if(!m||strcmp(m->language,"zh")||m->phonetic_dim!=32||m->context_dim!=42)return -1;int rc=parse_text(text);if(rc)return rc;rc=load_primary();if(rc)return rc;rc=build_context();if(rc)return rc;rc=resolve_readings();if(rc)return rc;rc=fuse_erhua();if(rc)return rc;rc=build_syllables();if(rc)return rc;rc=neutral_policy();if(rc)return rc;rc=citation_policy();if(rc)return rc;surface_policy();int dim=74,units=1;memset(x,0,TTS_MAX_UNITS*dim*4);memset(edge,0,TTS_MAX_UNITS);memset(state.comma,0,sizeof(state.comma));x[0]=1;edge[0]=1;owner[0]=-1;
 for(int si=0;si<state.ns;si++){Syllable*s=&state.syllables[si];if(!valid(s->surface))return -5;const ZhRoot*r=&zh_roots[s->surface.root];if(units+(int)r->n+1>TTS_MAX_UNITS)return -2;s->first=units;
  int cue=0;for(int i=0;i<state.endpoints;i++)if(state.question_end[i]>=s->end){cue=state.question_kind[i];break;}
  for(unsigned j=0;j<r->n;j++){unsigned index=j==0&&r->n==2?r->initial:r->rime;float*out=x+units*dim,*c=out+32;memcpy(out,zh_phone_bits[index],32*4);c[8]=j==r->n-1?s->level/4.f:0;c[9]=j==0&&si?state.syllables[si-1].level/4.f:0;memcpy(c+14,contours[s->surface.tone],12);c[17]=1;c[20]=j==0;c[21]=j==r->n-1;if(s->known){memcpy(c+22,contours[s->citation],12);c[25]=1;}if(si&&state.syllables[si-1].level==0){memcpy(c+26,contours[state.syllables[si-1].surface.tone],12);c[29]=1;}c[30]=s->surface.tone==5;c[31]=s->known&&s->citation!=s->surface.tone;c[32]=cue==1;c[33]=cue==2;owner[units++]=si;
  }s->stop=units;x[units*dim]=1;edge[units]=1;owner[units]=si;if(si+1<state.ns){x[units*dim+40]=x[units*dim+41]=s->level/4.f;}units++;
 }
 int internal=units-2;for(int i=1;i<units-1;i++){float*c=x+i*dim+32;c[5]=(i-1)/(float)(internal>1?internal-1:1);c[6]=fminf(internal/200.f,2);c[7]=fminf(state.ng/40.f,2);}x[42]=1;x[(units-1)*dim+43]=1;x[38]=x[(units-1)*dim+38]=fminf(internal/200.f,2);
 for(int gi=0;gi<state.ng;gi++){int first=-1,last=-1;for(int si=0;si<state.ns;si++)if(state.syllables[si].group==gi){if(first<0)first=state.syllables[si].first;last=state.syllables[si].stop;}if(first<0)return -5;int size=last-first;for(int i=first;i<last;i++){float*c=x+i*dim+32;c[0]=i==first;c[1]=i==last-1;c[2]=(i-first)/(float)(size>1?size-1:1);c[3]=fminf(size/20.f,2);c[4]=gi/(float)(state.ng>1?state.ng-1:1);}}
 for(int si=0;si<state.ns;si++){Syllable*s=&state.syllables[si];characters[si]=vocabulary(state.cp[s->start]);for(int j=0;j<66;j++){float sum=0;for(int i=s->first;i<s->stop;i++)sum+=x[i*dim+j];numeric[si*66+j]=sum/(s->stop-s->first);}if(si+1<state.ns){for(int i=s->end;i<state.syllables[si+1].start;i++)if(state.levels[i]>=2)break;/* comma is supplied by its actual punctuation category below */}}
 /* Re-read punctuation to distinguish commas from other level2 marks. Hints
  * and #markers never count as Chinese syllables. */
 int han=0;for(const char*p=text;*p;){uint32_t c;if(native_utf8_read(&p,&c)<1)return -4;if(is_han(c)){han++;if(*p=='['){const char*end;Reading hint;if(decode_hint(p+1,&end,&hint))p=end;}}else if(set_has(c,"，,")){for(int si=0;si+1<state.ns;si++)if(state.syllables[si].end==han)state.comma[state.syllables[si].stop]=1;}}
 state.n=units;state.np=0;for(int si=0;si+1<state.ns;si++){Syllable*a=&state.syllables[si],*b=a+1;Word*word=group_word(si);if(a->group!=b->group||word->start!=a->start||word->end!=b->end||a->end-a->start!=1||b->end-b->start!=1||a->level||a->hint||b->hint||!same_root(a->surface,b->surface)||b->surface.tone!=5||!word_in(word,kinship,sizeof(kinship)/sizeof(*kinship)))continue;int k=state.np++;state.pairs[k][0]=a->first;state.pairs[k][1]=a->stop;state.pairs[k][2]=b->first;state.pairs[k][3]=b->stop;state.pair_final[k]=si+1==state.ns-1||b->level>=2;}
 *n=units;*w=state.ns;ready=1;return 0;
}
int native_zh_policy(int n,uint8_t*comma,int32_t(*ranges)[2],int*ns,int32_t(*pairs)[4],uint8_t*final,int*np){if(!ready||n!=state.n)return 0;memcpy(comma,state.comma,n);*ns=state.ns;for(int i=0;i<state.ns;i++){ranges[i][0]=state.syllables[i].first;ranges[i][1]=state.syllables[i].stop;}*np=state.np;memcpy(pairs,state.pairs,state.np*16);memcpy(final,state.pair_final,state.np);return 1;}
