/* Export the preserved frontend's linguistic decisions, without model weights. */
#include "zh_t2s.c"
#include <stdio.h>
/* Run the reference decision chain and leave the syllable state in the globals.
 * Returns the syllable count (>=0) or the original negative error code. */
static int run_pipeline(const char *text,int separate) {
 char normalized[4096];
 int rc=qb_t2s(text,normalized,sizeof(normalized));if(rc<0)return rc;
 if(!separate){char phones[4096];rc=qb_zh_kokoro(normalized,phones,sizeof(phones));return rc<0?rc:state.ns;}
 ready=0;memset(memo_index,0,sizeof(memo_index));memo_used=0;
 if((rc=parse_text(normalized))||(rc=load_primary())||(rc=build_context())||
    (rc=resolve_readings())||(rc=build_syllables())||(rc=neutral_policy())||
    (rc=citation_policy()))return rc;
 surface_policy();
 return state.ns;
}
int qb_reference_state(const char*t){return run_pipeline(t,0);}
int qb_reference_state_separate(const char*t){return run_pipeline(t,1);}
static int export_syllables(const char *text,char *out,int capacity,int separate) {
 int rc=run_pipeline(text,separate);if(rc<0)return rc;
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
