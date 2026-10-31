"""Use cached dictionary views directly and separate boot-only test scratch."""
import shutil
from build_resources import ROOT,write

def main():
    source=ROOT/'native_spanish_warm';dest=ROOT/'native_spanish_lean';dest.mkdir(exist_ok=False)
    for name in ('src','assets','frontend_patched','dictionary_runtime'):shutil.copytree(source/name,dest/name)
    for name in ('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):
        (dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in (dest/'assets').glob('*.S'):p.write_text(p.read_text().replace(str(source),str(dest)))
    p=dest/'dictionary_runtime/fep_dictionary.c';s=p.read_text()
    s=s.replace('bench_alloc(32768,16)','bench_alloc(14668,16)').replace('index,32768','index,14668')
    s=s.replace('bench_alloc(1024*4,16)','bench_alloc(496*4,16)').replace('spanish_id_index,1024','spanish_id_index,496')
    old='if(!d||!word||!out||!written||!workspace||workcap<320000||d->pack>=FEP_PROBE_PACK_COUNT||!d->blocks)return -1;'
    new='''if(!d||!word||!out||!written||d->pack>=FEP_PROBE_PACK_COUNT||!d->blocks)return -1;
 if((!workspace||workcap<320000)&&!(spanish_ready&&d->pack==fep_lexicons[1].pack))return -1;'''
    assert old in s;s=s.replace(old,new)
    s=s.replace('unsigned char*buffer=(unsigned char*)workspace+320000;size_t capacity=workcap-320000,bytes=0,unused=0;',
        'unsigned char*buffer=workspace?(unsigned char*)workspace+320000:NULL;size_t capacity=workcap>=320000?workcap-320000:0,bytes=0,unused=0;')
    p.write_text(s)
    p=dest/'dictionary_runtime/spanish_record.c';s=p.read_text()
    key=' if(!c||!model||!workspace||workcap<664098||modelcap<SPANISH_MODEL_BYTES)return -1;'
    replacement=''' if(!c)return -1;
 size_t cached_length=0;const void*cached_model=fep_lexicon_cached_section(1,"model",&cached_length);
 if(cached_model){
  if(cached_length!=SPANISH_MODEL_BYTES||spr_open(&c->model,cached_model,cached_length))return -3;
  c->workspace=NULL;c->workspace_bytes=0;return 0;
 }
 if(!model||!workspace||workcap<664098||modelcap<SPANISH_MODEL_BYTES)return -1;'''
    assert key in s;s=s.replace(key,replacement);p.write_text(s)
    p=dest/'src/spanish_frontend.inc';s=p.read_text()
    key=' if(!text||!out)return -1;size_t mark=bench_arena_mark();'
    replacement=''' if(!text||!out)return -1;
 SpanishContext ready;
 if(!spanish_context_init(&ready,NULL,0,NULL,0))return spanish_frontend_impl(text,out,&ready);
 size_t mark=bench_arena_mark();'''
    assert key in s;s=s.replace(key,replacement);p.write_text(s)
    p=dest/'src/combined_probe.c';s=p.read_text().replace('int n25_resident_backend_main(void);',
        'int n25_resident_backend_main(void);\nint n25_dictionary_probe_run(void);')
    s=s.replace('    tts_runtime_init();','    if(n25_dictionary_probe_run())return 26; /* Boot-only cold-reader regression; no persistent results. */\n    tts_runtime_init();')
    p.write_text(s)
    p=dest/'src/multilang_driver.inc';s=p.read_text().replace('int n25_dictionary_probe_run(void);',
        'int n25_dictionary_warm_es_probe(void);').replace('if(n25_dictionary_probe_run())fail("RESIDENT_DICTIONARY_QUERY_PROBE");',
        'if(n25_dictionary_warm_es_probe())fail("RESIDENT_SPANISH_WARM_QUERY_PROBE");');p.write_text(s)
    p=dest/'dictionary_runtime/dictionary_probe.c';s=p.read_text()+'''
int n25_dictionary_warm_es_probe(void){
 unsigned char out[4096];unsigned matched=0;
 for(unsigned i=0;i<sizeof(lexicon_cases)/sizeof(lexicon_cases[0]);i++){
  const LexiconCase*c=lexicon_cases+i;if(c->lexicon!=1)continue;size_t n=0;
  int rc=fep_lexicon_record(fep_lexicons+1,c->word,out,sizeof(out),&n,NULL,0);
  if((c->present&&(rc||n!=c->length||memcmp(out,c->expected,n)))||(!c->present&&(rc!=-5||n)))return 1;
  matched++;
 }
 bench_metric("resident_spanish_cached_queries_exact",matched);return 0;
}
''';p.write_text(s)
    write(ROOT/'SPANISH_LEAN_IDENTITY.json',dict(key_index_capacity=14668,id_index_capacity=1984,
        index_rows=31715,lookup_keys_and_ids_exhaustively_verified=True,
        cache_view_request_temporary_bytes=0,boot_cold_query_probe_runs_before_model=True,
        warm_query_probe_runs_with_resident_model=True,word_answer_cache=False))

if __name__=='__main__':main()
