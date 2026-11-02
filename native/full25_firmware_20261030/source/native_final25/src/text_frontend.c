/* Original reference lexicon decisions -> original NEW25 whole-rime features.
 * Scope: Han text, punctuation and explicit pinyin hints; unsupported text fails.
 */
#include "text_frontend.h"
#include "whole_rime_map.h"
#include "../frontend_source/bridge.c"
static const N25Rime*lookup_rime(const char*name){
 unsigned lo=0,hi=N25_RIME_COUNT;
 while(lo<hi){unsigned m=lo+(hi-lo)/2;if(strcmp(n25_rimes[m].name,name)<0)lo=m+1;else hi=m;}
 return lo<N25_RIME_COUNT&&!strcmp(n25_rimes[lo].name,name)?n25_rimes+lo:0;
}
static void silence(N25TextFeatures*out,unsigned at){out->phones[at]=N25_SIL_PHONE;out->pause[at]=1;out->phonetic[at*32]=1.f;}
int n25_chinese_text_frontend(const char*text,N25TextFeatures*out){
 if(!text||!out)return -1;
 int rc=qb_reference_state(text);if(rc<0)return rc;
 int separate=0;
 for(int i=0;i<state.ns;i++){const N25Rime*r=lookup_rime(root(state.syllables[i].surface));if(!r||!r->count){if(state.syllables[i].end-state.syllables[i].start!=2)return -31;separate=1;}}
 if(separate){rc=qb_reference_state_separate(text);if(rc<0)return rc;}
 /* Zero only the entries this text can use: the full 132 KiB structure zeroing
  * dominated the first-request frontend cost. */
 unsigned total=1;for(int i=0;i<state.ns;i++){const N25Rime*r=lookup_rime(root(state.syllables[i].surface));if(!r||!r->count)return -32;total+=r->count+1;}
 if(total>N25_TEXT_PHONE_CAP)return -32;
 memset(out->phones,0,total*4);memset(out->pause,0,total*4);
 memset(out->phonetic,0,(size_t)total*32*4);memset(out->context,0,(size_t)total*34*4);
 unsigned count=1,starts[128],stops[128];silence(out,0);
 for(int i=0;i<state.ns;i++){
  Syllable*s=state.syllables+i;const N25Rime*r=lookup_rime(root(s->surface));
  if(!r||!r->count||r->count+1>N25_TEXT_PHONE_CAP-count)return -32;
  starts[i]=count;int tone=s->surface.tone,cue=0;
  for(int q=0;q<state.endpoints;q++)if(state.question_end[q]>=s->end){cue=state.question_kind[q];break;}
  for(unsigned j=0;j<r->count;j++,count++){
   out->phones[count]=r->phones[j];memcpy(out->phonetic+count*32,n25_phone_features[r->features[j]],32*4);
   float*c=out->context+count*34;c[8]=j+1==r->count?(float)s->level/4.f:0;
   c[9]=j==0&&i?(float)state.syllables[i-1].level/4.f:0;
   for(int k=0;k<3;k++)c[14+k]=contours[tone][k];c[17]=1;c[20]=j==0;c[21]=j+1==r->count;
   if(s->known&&s->citation){for(int k=0;k<3;k++)c[22+k]=contours[s->citation][k];c[25]=1;}
   if(i&&!state.syllables[i-1].level){for(int k=0;k<3;k++)c[26+k]=contours[state.syllables[i-1].surface.tone][k];c[29]=1;}
   c[30]=tone==5;c[31]=s->known&&s->citation&&s->citation!=tone;c[32]=cue==1;c[33]=cue==2;
  }
  stops[i]=count;silence(out,count);
  if(i+1<state.ns)out->context[count*34+8]=out->context[count*34+9]=(float)s->level/4.f;
  count++;
 }
 out->count=count;unsigned groups=state.syllables[state.ns-1].group+1;
 for(unsigned i=0;i<count;i++){
  float*c=out->context+i*34;c[6]=(float)fmin((double)(count-2)/200.,2.);
  if(i&&i+1<count){c[5]=(float)(i-1)/(float)(count>3?count-3:1);c[7]=(float)fmin((double)groups/40.,2.);}
 }
 out->context[10]=1;out->context[(count-1)*34+11]=1;
 for(unsigned g=0;g<groups;g++){
  unsigned first=count,last=0;
  for(int i=0;i<state.ns;i++)if((unsigned)state.syllables[i].group==g){if(first==count)first=starts[i];last=stops[i];}
  if(first==count)continue;unsigned size=last-first;
  for(unsigned i=first;i<last;i++){
   float*c=out->context+i*34;c[0]=i==first;c[1]=i+1==last;
   c[2]=(float)((double)(i-first)/(double)(size>1?size-1:1));c[3]=(float)fmin((double)size/20.,2.);
   c[4]=(float)((double)g/(double)(groups>1?groups-1:1));
  }
 }
 return 0;
}
