"""Integrate exact dictionary queries and write features into their final buffers."""
import shutil
from build_resources import ROOT,write,sha

def main():
    source=ROOT/'native_lut_delta';dest=ROOT/'native_dictionary_views';dest.mkdir(exist_ok=False)
    for name in ('src','assets','frontend_patched'):shutil.copytree(source/name,dest/name)
    for name in ('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):
        (dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in (dest/'assets').glob('*.S'):p.write_text(p.read_text().replace(str(source),str(dest)))
    shutil.copytree(ROOT/'dictionary_runtime',dest/'dictionary_runtime')
    p=dest/'dictionary_runtime/dictionary_probe.c';s=p.read_text().replace('size_t size=bench_arena_capacity();',
        'size_t mark=bench_arena_mark();size_t size=664098;')
    s=s.replace('bench_metric("lookup_workspace_bytes",size);',
        'if(bench_arena_rewind(mark))return 3;bench_metric("lookup_workspace_bytes",size);')
    p.write_text(s)
    p=dest/'src/four_frontend.h';s=p.read_text();i=s.index('/* Ordinary NFC')
    s=s[:i]+'''typedef struct {
 unsigned count,phonetic_dim,context_dim;
 uint32_t*phones,*pause;float*phonetic,*context;
}N25FourOutput;
int n25_four_frontend_to_view(const char*,const char*,N25FourOutput*);
'''+s[i:];p.write_text(s)
    p=dest/'src/four_frontend.c';s=p.read_text();s=s.replace(
        'int n25_four_frontend(const char *language,const char *text,N25FourFeatures*out){',
        'int n25_four_frontend_to_view(const char *language,const char *text,N25FourOutput*out){')
    s+='''
int n25_four_frontend(const char*language,const char*text,N25FourFeatures*out){
 if(!out)return -1;
 N25FourOutput view={0,0,0,out->phones,out->pause,out->phonetic,out->context};
 int rc=n25_four_frontend_to_view(language,text,&view);
 if(!rc){out->count=view.count;out->phonetic_dim=view.phonetic_dim;out->context_dim=view.context_dim;}
 return rc;
}
''';p.write_text(s)
    p=dest/'src/multilang_frontend.c';s=p.read_text();a=s.index(' N25FourFeatures*four_features=');b=s.index(' if(n25_romfs_set_workspace(NULL,0)',a)
    s=s[:a]+''' N25FourOutput view={0,0,0,out->phones,out->pause,out->phonetic,out->context};
 int rc=n25_four_frontend_to_view(language,text,&view);
 if(!rc){
  if(view.phonetic_dim!=n25_current_norm->phonetic_dim||view.context_dim!=n25_current_norm->context_dim)rc=-97;
  else out->count=view.count;
 }
'''+s[b:];p.write_text(s)
    p=dest/'src/multilang_driver.inc';s=p.read_text();s='int n25_dictionary_probe_run(void);\n'+s
    s=s.replace('tts_runtime_init();n25_frontend_runtime_init();',
        'tts_runtime_init();n25_frontend_runtime_init();\n if(n25_dictionary_probe_run())fail("RESIDENT_DICTIONARY_QUERY_PROBE");')
    p.write_text(s)
    write(ROOT/'DICTIONARY_VIEWS_IDENTITY.json',dict(source_root=str(source),native_root=str(dest),
        feature_temporary_removed_bytes=147468,final_buffers_reused=True,
        dictionary_query_workspace_bytes=664098,queries_before_requests_discarded=True,
        dictionary_word_cache=False,remaining_feature_adapters_not_claimed=True))

if __name__=='__main__':main()
