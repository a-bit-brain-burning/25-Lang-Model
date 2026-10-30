/* Export the preserved frontend's linguistic decisions, without model weights. */
#include "zh_t2s.c"
#include <stdio.h>
static int export_syllables(const char *text,char *out,int capacity,int separate) {
 char normalized[4096], phones[4096];
 int rc=qb_t2s(text,normalized,sizeof(normalized));if(rc<0)return rc;
 if(!separate){rc=qb_zh_kokoro(normalized,phones,sizeof(phones));if(rc<0)return rc;}
 else {
  ready=0;memset(memo_index,0,sizeof(memo_index));memo_used=0;
  if((rc=parse_text(normalized))||(rc=load_primary())||(rc=build_context())||
     (rc=resolve_readings())||(rc=build_syllables())||(rc=neutral_policy())||
     (rc=citation_policy()))return rc;
  surface_policy();
 }
 int used=0;out[0]=0;
 for(int i=0;i<state.ns;i++) {
  const Syllable*s=&state.syllables[i];char line[256];
  int n=snprintf(line,sizeof(line),"%d\t%d\t%s%d\t%s%d\t%d\t%d\t%d\t%d\n",
   s->start,s->end,root(s->surface),s->surface.tone,root(s->raw),s->raw.tone,
   s->known?s->citation:0,s->group,s->level,s->hint);
  if(n<0||n>=(int)sizeof(line)||append(out,capacity,&used,line))return -20;
 }
 return used;
}

int qb_reference_syllables(const char*t,char*out,int cap){return export_syllables(t,out,cap,0);}
int qb_reference_syllables_separate(const char*t,char*out,int cap){return export_syllables(t,out,cap,1);}
