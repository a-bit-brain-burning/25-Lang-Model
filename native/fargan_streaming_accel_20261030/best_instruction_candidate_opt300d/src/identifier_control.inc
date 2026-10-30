/* CEA342e identifier controls. Prepared metadata, not a native text parser.
 * Exactly the original control order; no model/PCM modification or training. */
#include "../assets/identifier_metadata.h"
static float identifier_original_logf0[MAX_FRAMES],identifier_original_voice[MAX_FRAMES],identifier_after_logf0[MAX_FRAMES];
static float identifier_pitch_records[sizeof(identifier_units)/sizeof(identifier_units[0])*8];
static unsigned identifier_group_begin[PHONE_COUNT],identifier_group_end[PHONE_COUNT],identifier_group_done[PHONE_COUNT];
static unsigned identifier_pitch_first_request,identifier_pitch_max_raw_ahead;
static void identifier_observe_physical(unsigned frame,unsigned dimension,float physical){
 if(!identifier_unit_count)return;
 if(dimension==0){identifier_original_logf0[frame]=physical;identifier_after_logf0[frame]=physical;}
 if(dimension==ACOUSTIC_DIM-1)identifier_original_voice[frame]=physical;
}
static unsigned identifier_stages[7][PHONE_COUNT];
static unsigned identifier_gap_count,identifier_digit_count,identifier_u_count,identifier_c_count;
static unsigned identifier_pitch_frames,identifier_pitch_groups;
static uint64_t identifier_prepare_ticks;
static unsigned id_sum(const unsigned*d,unsigned a,unsigned b){unsigned q=0;for(unsigned i=a;i<b;i++)q+=d[i];return q;}
static float id_median(float*x,unsigned n){
 if(!n)fail("IDENTIFIER_EMPTY_MEDIAN");
 for(unsigned i=1;i<n;i++){float q=x[i];unsigned j=i;while(j&&x[j-1]>q){x[j]=x[j-1];j--;}x[j]=q;}
 return n&1?x[n/2]:(x[n/2-1]+x[n/2])*.5f;
}
static float id_clip(float x,float lo,float hi){return x<lo?lo:x>hi?hi:x;}
static int id_alpha(char c){return(c>='A'&&c<='Z')||(c>='a'&&c<='z');}
static int id_digit(char c){return c>='0'&&c<='9';}
static void id_commit(float*p,unsigned*d,const unsigned*next,unsigned stage){
 for(unsigned i=0;i<PHONE_COUNT;i++){
  if(d[i]!=next[i]){
   if(input_pause[i]){p[i*5+1]=logf((float)(next[i]?next[i]:1));p[i*5+2]=next[i]?1.f:-1.f;}
   else{if(!next[i])fail("IDENTIFIER_ZERO_PHONE");p[i*5]=logf((float)next[i]);}
   d[i]=next[i];
  }
 }
 memcpy(identifier_stages[stage],d,PHONE_COUNT*sizeof(unsigned));
}
static void id_set_gap(unsigned*d,unsigned a,unsigned b,unsigned frames_){
 if(a>=b)fail("IDENTIFIER_MISSING_GAP");
 for(unsigned i=a;i<b;i++){if(!input_pause[i])fail("IDENTIFIER_NONPAUSE_GAP");d[i]=0;}d[a]=frames_;
}
static void identifier_adjust_durations(float*p,unsigned*d){
 if(!identifier_unit_count)return;
 unsigned next[PHONE_COUNT];float med[PHONE_COUNT];
 memcpy(identifier_stages[0],d,sizeof(identifier_stages[0]));memcpy(next,d,sizeof(next));
 /* identifier_rhythm.install: original model totals determine group target. */
 for(unsigned g=0;g<identifier_group_count;g++){
  unsigned nm=0;
  for(unsigned u=0;u<identifier_unit_count;u++){const IdentifierUnit*q=&identifier_units[u];if(q->group==g&&q->symbol!='W')med[nm++]=(float)id_sum(d,q->start,q->end);}
  unsigned target=(unsigned)id_clip(nearbyintf(id_median(med,nm)),20,28);
  unsigned previous=identifier_unit_count;
  for(unsigned u=0;u<identifier_unit_count;u++){
   const IdentifierUnit*q=&identifier_units[u];if(q->group!=g)continue;
   unsigned nc=0,consonants=0,core_total=0,budget=target*(q->symbol=='W'?3:1);unsigned char core[PHONE_COUNT];double ideal[PHONE_COUNT];
   for(unsigned i=q->start;i<q->end;i++)nc+=identifier_rhythm_core[i]!=0;
   for(unsigned i=q->start;i<q->end;i++){core[i]=nc?identifier_rhythm_core[i]:1;if(core[i])core_total+=d[i];else consonants+=d[i];}
   if(!nc)nc=q->end-q->start;
   if(budget<consonants+nc)budget=consonants+nc;
   unsigned got=0;
   for(unsigned i=q->start;i<q->end;i++)if(core[i]){ideal[i]=(double)d[i]/(core_total?core_total:1)*(budget-consonants);unsigned v=(unsigned)floor(ideal[i]);next[i]=v?v:1;got+=next[i];}
   while(got<budget-consonants){unsigned best=q->start;double score=-1e100;for(unsigned i=q->start;i<q->end;i++)if(core[i]&&ideal[i]-next[i]>score){score=ideal[i]-next[i];best=i;}next[best]++;got++;}
   while(got>budget-consonants){unsigned best=q->start;double score=-1e100;for(unsigned i=q->start;i<q->end;i++)if(core[i]&&next[i]>1&&(double)next[i]-ideal[i]>score){score=(double)next[i]-ideal[i];best=i;}if(score== -1e100)fail("IDENTIFIER_RHYTHM_BUDGET");next[best]--;got--;}
   if(previous<identifier_unit_count)id_set_gap(next,identifier_units[previous].end,q->start,3);
   previous=u;
  }
 }
 id_commit(p,d,next,1);
 /* U natural: keep yiu1 total, 4-frame i glide, retained u nucleus. */
 memcpy(next,d,sizeof(next));
 for(unsigned u=0;u<identifier_unit_count;u++){
  const IdentifierUnit*q=&identifier_units[u];if(q->symbol!='U')continue;
  unsigned total=id_sum(d,q->start,q->end);if(q->end-q->start!=2||total<=4)fail("IDENTIFIER_U_SHAPE");next[q->start]=4;next[q->start+1]=total-4;identifier_u_count++;
  for(unsigned v=0;v<identifier_unit_count;v++){
   const IdentifierUnit*r=&identifier_units[v];if(r->group!=q->group)continue;
   if(r->symbol_index+1==q->symbol_index)id_set_gap(next,r->end,q->start,id_alpha(r->symbol)?1:2);
   if(r->symbol_index==q->symbol_index+1)id_set_gap(next,q->end,r->start,id_alpha(r->symbol)?1:2);
  }
 }
 id_commit(p,d,next,2);
 /* Long C: at most90ms consonant, >=180ms vowel or prior+60ms. */
 memcpy(next,d,sizeof(next));
 for(unsigned u=0;u<identifier_unit_count;u++){
  const IdentifierUnit*q=&identifier_units[u];if(q->symbol!='C')continue;
  if(q->end-q->start!=2)fail("IDENTIFIER_C_SHAPE");
  next[q->start]=d[q->start]<9?d[q->start]:9;unsigned v=d[q->start+1]+6;next[q->start+1]=v>18?v:18;identifier_c_count++;
 }
 id_commit(p,d,next,3);
 /* First digit: final phone +50ms, energy prosody +.25 and first gap. */
 memcpy(next,d,sizeof(next));
 for(unsigned u=0;u<identifier_unit_count;u++){
  const IdentifierUnit*q=&identifier_units[u];if(!q->first_digit)continue;
  next[q->end-1]+=5;p[(q->end-1)*5+4]+=.25f;
  unsigned a=q->start;while(a&&input_pause[a-1])a--;
  unsigned old=id_sum(d,a,q->start),gap=a?old+5:old;if(gap<10)gap=10;id_set_gap(next,a,q->start,gap);identifier_digit_count++;
 }
 id_commit(p,d,next,4);
 /* Existing flow stage (retains first-digit emphasis and hyphens). */
 memcpy(next,d,sizeof(next));
 for(unsigned u=1;u<identifier_unit_count;u++){
  const IdentifierUnit*l=&identifier_units[u-1],*r=&identifier_units[u];
  if(!r->connect_previous||(id_digit(r->symbol)&&!id_digit(l->symbol)))continue;
  id_set_gap(next,l->end,r->start,1);
 }
 id_commit(p,d,next,5);
 /* CEA connection overrides same-code nonhyphen gaps to0, letter->digit6. */
 memcpy(next,d,sizeof(next));
 for(unsigned u=1;u<identifier_unit_count;u++){
  const IdentifierUnit*l=&identifier_units[u-1],*r=&identifier_units[u];if(!r->connect_previous)continue;
  id_set_gap(next,l->end,r->start,id_digit(r->symbol)&&!id_digit(l->symbol)?6:0);identifier_gap_count++;
 }
 id_commit(p,d,next,6);
 if(bench_write_file("identifier_duration_stages.u32",identifier_stages,sizeof(identifier_stages)))fail("IDENTIFIER_DURATION_EXPORT");
}
/* Raw decoder keeps the original identifier baseline's 16-frame boundaries.
 * Only the consumer-facing wrapper publishes pitch-final condition rows. */
static void identifier_raw_through(unsigned end){
 if(end>frames)end=frames;
 if(!identifier_unit_count){ensure_raw_condition_chunk(end);return;}
 while(decoded_frames<end){unsigned next=decoded_frames+16;if(next>frames)next=frames;ensure_raw_condition_chunk(next);}
}
static void identifier_pitch_prepare(void){
 unsigned ends[PHONE_COUNT+1];ends[0]=0;for(unsigned i=0;i<PHONE_COUNT;i++)ends[i+1]=ends[i]+predicted_durations[i];
 for(unsigned g=0;g<identifier_group_count;g++){
  unsigned first=identifier_unit_count,last=0;
  for(unsigned u=0;u<identifier_unit_count;u++)if(identifier_units[u].group==g){if(first==identifier_unit_count)first=u;last=u;}
  if(first==identifier_unit_count)fail("IDENTIFIER_EMPTY_GROUP");
  identifier_group_begin[g]=ends[identifier_units[first].start];identifier_group_end[g]=ends[identifier_units[last].end];identifier_group_done[g]=0;
  if(g&&identifier_group_begin[g]<identifier_group_end[g-1])fail("IDENTIFIER_GROUP_OVERLAP");
 }
}
static void identifier_pitch_group(unsigned g,unsigned request){
 uint64_t start=bench_now();
 if(!identifier_pitch_first_request)identifier_pitch_first_request=request;
 identifier_raw_through(identifier_group_end[g]);
 if(decoded_frames>request&&decoded_frames-request>identifier_pitch_max_raw_ahead)identifier_pitch_max_raw_ahead=decoded_frames-request;
 size_t mark=bench_arena_mark();
 float*before=identifier_original_logf0,*after=identifier_after_logf0,*voice=identifier_original_voice,*work=alloc(frames),*target=alloc(frames),*smooth=alloc(frames),*records=identifier_pitch_records;
 unsigned ends[PHONE_COUNT+1];ends[0]=0;for(unsigned i=0;i<PHONE_COUNT;i++)ends[i+1]=ends[i]+predicted_durations[i];
 {
  unsigned first=identifier_unit_count,last=0,count=0;
  for(unsigned u=0;u<identifier_unit_count;u++)if(identifier_units[u].group==g){if(first==identifier_unit_count)first=u;last=u;count++;}
  if(!count)fail("IDENTIFIER_EMPTY_GROUP");
  unsigned begin=ends[identifier_units[first].start],end=ends[identifier_units[last].end],selected=0;
  for(unsigned i=begin;i<end;i++)if(input_phonetic[frame_owner[i]*PHONETIC_DIM+31]>.5f&&voice[i]>.55f)work[selected++]=before[i];
  if(selected<3){selected=0;for(unsigned i=begin;i<end;i++)if(voice[i]>.55f)work[selected++]=before[i];}
  float base=selected?id_median(work,selected):norm_mean[0];base=id_clip(base,norm_mean[0]-1.1f*norm_std[0],norm_mean[0]+1.1f*norm_std[0]);
  unsigned last_end=0,j=0;
  for(unsigned u=first;u<=last;u++){
   const IdentifierUnit*q=&identifier_units[u];if(q->group!=g)continue;
   unsigned a=ends[q->start],b=ends[q->end],n=b-a;if(!n)fail("IDENTIFIER_EMPTY_SYMBOL");
   double level64=(double)base+.012-.025*j/(count>1?count-1:1);float level=(float)level64;j++;
   if(id_alpha(q->symbol)){
    for(unsigned k=0;k<n;k++){
     float phase=n>1?(float)k/(n-1):0.f;
     target[k]=level;if(q->symbol!='U')target[k]+=.018f*sinf(3.14159265358979323846f*phase);
    }
   }else{
    unsigned local=0;for(unsigned i=a;i<b;i++)if(input_phonetic[frame_owner[i]*PHONETIC_DIM+31]>.5f&&voice[i]>.55f)work[local++]=before[i];
    if(!local){for(unsigned i=a;i<b;i++)work[local++]=before[i];}
    float center=id_median(work,local);float reg=(float)(level64+.4*((double)center-base));
    for(unsigned k=0;k<n;k++)target[k]=reg+.85f*(before[a+k]-center);
   }
   for(unsigned k=0;k<n;k++)target[k]=id_clip(target[k],before[a+k]-.18f,before[a+k]+.18f);
   if(n>=5){
    for(unsigned k=0;k<n;k++){double sum=0;const unsigned w[5]={1,2,3,2,1};for(int t=-2;t<=2;t++){int at=(int)k+t;if(at<0)at=0;if(at>=(int)n)at=n-1;sum+=(double)target[at]*(double)w[t+2]/9.;}smooth[k]=(float)sum;}
   }else memcpy(smooth,target,n*4);
   for(unsigned k=0;k<n;k++){float confidence=id_clip((voice[a+k]-.3f)/.5f,0,1);/* np.convolve promotes target to F64 for n>=5. */
    if(n>=5){double sum=0;const unsigned w[5]={1,2,3,2,1};for(int t=-2;t<=2;t++){int at=(int)k+t;if(at<0)at=0;if(at>=(int)n)at=n-1;sum+=(double)target[at]*(double)w[t+2]/9.;}after[a+k]=(float)((double)before[a+k]+(double)confidence*(sum-before[a+k]));}
    else after[a+k]=before[a+k]+confidence*(smooth[k]-before[a+k]);}
   if(last_end&&a>last_end){double left=after[last_end-1],right=after[a];for(unsigned k=0;k<a-last_end;k++)after[last_end+k]=(float)(left+(right-left)*(k+1)/(a-last_end+1));}
   for(unsigned k=0;k<n;k++){work[k]=before[a+k];}float mb=id_median(work,n);for(unsigned k=0;k<n;k++){work[k]=after[a+k];}float ma=id_median(work,n);
   float*rec=records+u*8;rec[0]=(float)a;rec[1]=(float)b;rec[2]=base;rec[3]=mb;rec[4]=ma;rec[5]=(float)q->tone;rec[6]=(float)q->symbol;rec[7]=.18f;
   last_end=b;
  }
  unsigned fade=(end-begin)/3;if(fade>3)fade=3;
  for(unsigned k=0;k<fade;k++){double weight=.5-.5*cos(3.14159265358979323846*(k+1)/(fade+1));after[begin+k]=(float)((double)before[begin+k]*(1-weight)+(double)after[begin+k]*weight);after[end-fade+k]=(float)((double)after[end-fade+k]*(1-weight)+(double)before[end-fade+k]*weight);}
  identifier_pitch_groups++;
 }
 for(unsigned i=identifier_group_begin[g];i<identifier_group_end[g];i++){if(!isfinite(after[i]))fail("IDENTIFIER_NONFINITE_PITCH");if(after[i]!=before[i])identifier_pitch_frames++;condition_cache[i*ACOUSTIC_DIM]=(after[i]-norm_mean[0])/norm_std[0];}
 identifier_group_done[g]=1;rewind_to(mark);identifier_prepare_ticks+=bench_now()-start;
}
static void ensure_condition(unsigned end){
 if(end>frames)end=frames;
 /* Strict intersection: a right halo touching a group must finish that whole
  * group's median and pitch before any adapter reads its first condition. */
 for(unsigned g=0;g<identifier_group_count;g++)if(!identifier_group_done[g]&&identifier_group_begin[g]<end)identifier_pitch_group(g,end);
 identifier_raw_through(end);
}
static void identifier_pitch_export(void){
 if(!identifier_unit_count)return;
 for(unsigned g=0;g<identifier_group_count;g++)if(!identifier_group_done[g])fail("IDENTIFIER_UNFINISHED_GROUP");
 if(bench_write_file("identifier_voice.f32",identifier_original_voice,frames*4)||bench_write_file("identifier_pitch_before.f32",identifier_original_logf0,frames*4)||bench_write_file("identifier_pitch_after.f32",identifier_after_logf0,frames*4)||bench_write_file("identifier_pitch_units.f32",identifier_pitch_records,identifier_unit_count*8*4))fail("IDENTIFIER_PITCH_EXPORT");
}

static void identifier_snapshot_first(void){}
static void identifier_report(void){identifier_pitch_export();}
