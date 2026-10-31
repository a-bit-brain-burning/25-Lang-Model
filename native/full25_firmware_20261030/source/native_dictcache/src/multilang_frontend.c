#include "bench.h"
#include "romfs.h"
#include "storage.h"
#include "language_norms.h"
#include "text_frontend.h"
#include "four_frontend.h"
#include "paged_lexicon.h"
#include <espeak-ng/speak_lib.h>
#include <string.h>
const N25LanguageNorm *n25_current_norm=n25_language_norms+24;

static const unsigned char norm_sha256[32]={145,36,138,213,207,125,8,15,29,32,156,147,106,78,69,136,44,147,161,230,251,165,217,133,232,55,137,61,20,116,184,203};
static unsigned frontend_initialized;
int n25_chinese_text_frontend(const char*,N25TextFeatures*);
void n25_frontend_runtime_init(void){
 if(frontend_initialized)return;
 unsigned char digest[32];storage_sha256((const unsigned char*)n25_language_norms,sizeof(n25_language_norms),digest);
 if(memcmp(digest,norm_sha256,32))bench_exit(81);
 size_t mark=bench_arena_mark();void*work=bench_alloc(582144,16);
 if(!work||n25_romfs_set_workspace(work,582144))bench_exit(82);
 n25_romfs_initialize();
 if(espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS,0,"/espeak-ng-data",0)<=0)bench_exit(83);
 if(!espeak_ListVoices(NULL))bench_exit(85);
 if(n25_romfs_set_workspace(NULL,0)||bench_arena_rewind(mark))bench_exit(84);
 if(n25_paged_lexicon_initialize())bench_exit(86);
 frontend_initialized=1;bench_metric("all25_normalization_banks_SHA256_passed",25);
}
int n25_text_frontend(const char*text,N25TextFeatures*out){
 if(n25_current_norm->language_id==24)return n25_chinese_text_frontend(text,out);
 static const struct {unsigned id;const char*name;} languages[]={
{0,"bg"},
{1,"cs"},
{2,"da"},
{3,"de"},
{4,"el"},
{5,"en"},
{6,"es"},
{7,"et"},
{8,"fi"},
{9,"fr"},
{10,"hr"},
{11,"hu"},
{12,"it"},
{13,"lt"},
{14,"lv"},
{15,"nl"},
{16,"no"},
{17,"pl"},
{18,"pt"},
{19,"ro"},
{20,"sk"},
{21,"sl"},
{22,"sv"},
{23,"tr"},
 };
 const char*language=NULL;
 for(unsigned i=0;i<sizeof(languages)/sizeof(languages[0]);i++)if(languages[i].id==n25_current_norm->language_id)language=languages[i].name;
 if(!language||!frontend_initialized)return -99;
 size_t mark=bench_arena_mark();
 N25FourOutput view={0,0,0,out->phones,out->pause,out->phonetic,out->context};
 int rc=n25_four_frontend_to_view(language,text,&view);
 if(!rc){
  if(view.phonetic_dim!=n25_current_norm->phonetic_dim||view.context_dim!=n25_current_norm->context_dim)rc=-97;
  else out->count=view.count;
 }
 if(n25_romfs_set_workspace(NULL,0)||bench_arena_rewind(mark))return -96;
 return rc;
}
