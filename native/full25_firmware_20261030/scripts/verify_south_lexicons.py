"""Exhaustive native-C lookup and independent semantic decode for BG/SL pages."""
import ctypes as C
import json
from build_resources import ROOT,run,write,sha
def main():
    native=ROOT/'native_text_regression';out=ROOT/'build/south_full_lookup_host_v1';out.mkdir(exist_ok=False)
    shim=out/'shim.c';shim.write_text('''#include <stddef.h>
#include <stdint.h>
static unsigned char arena[8192];static size_t used;
void*bench_alloc(size_t n,size_t alignment){size_t at=(used+alignment-1)&~(alignment-1);if(at+n>sizeof(arena))return NULL;used=at+n;return arena+at;}
size_t bench_arena_mark(void){return used;}int bench_arena_rewind(size_t mark){if(mark>used)return -1;used=mark;return 0;}
void bench_metric(const char*n,uint64_t v){(void)n;(void)v;}
void storage_sha256(const unsigned char*p,size_t n,unsigned char*out){(void)p;(void)n;(void)out;}
''')
    sources=[shim,native/'dictionary_runtime/paged_lexicon.c',native/'dictionary_runtime/south_lexicons.c',
        native/'src/frontend_inflate.c',native/'third_party/puff/puff.c']
    library=out/'native_reader.so';command=run(['cc','-shared','-fPIC','-O2','-std=c11',
        '-I'+str(native/'src'),'-I'+str(native/'dictionary_runtime'),*sources,'-o',library],out/'compile.log')
    lib=C.CDLL(str(library));lookup=lib.n25_paged_lexicon_record
    lookup.argtypes=[C.c_char_p,C.c_char_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)];lookup.restype=C.c_int
    buffer=C.create_string_buffer(256);rows=[]
    for language,file in [('bg','bg_expanded_lexicon.json'),('sl','sl_lexicon.json')]:
        entries=json.loads((ROOT/'host_reference'/file).read_text())['entries'];count=0
        for word,entry in entries.items():
            size=C.c_size_t();assert lookup(language.encode(),word.encode(),buffer,len(buffer),C.byref(size))==0,word
            raw=buffer.raw[:size.value];n,stress=raw[:2];pos=2+n;vowels=[]
            signature=['aeiouər'[value]for value in raw[2:pos]]
            number=raw[pos];pos+=1
            for i in range(number):
                syllable,flags,length=raw[pos:pos+3];pos+=3;name=raw[pos:pos+length].decode();pos+=length
                vowels.append(dict(syllable=syllable,ipa=name,quantity_known=bool(flags&1),quality_known=bool(flags&2)))
            assert pos==len(raw),word
            expected=entry['prosody'];assert n==expected['nucleus_count']and signature==expected['nucleus_signature'],word
            assert (stress-1 if stress else None)==expected['stress_syllable'],word
            assert bool(stress)==entry['stress_known'],word
            original=[dict(syllable=v['syllable'],ipa=v['ipa'],quantity_known=bool(v.get('quantity_known')),quality_known=bool(v.get('quality_known')))for v in expected['vowels']]
            assert vowels==original,word;count+=1
        absent=0
        for word in ['zzzztest','zzzzzzzzzzzz','not-a-real-native-word','']:
            if word in entries:continue
            size=C.c_size_t();assert lookup(language.encode(),word.encode(),buffer,len(buffer),C.byref(size))==-5;absent+=1
        rows.append(dict(language=language,all_records_exact=count,absent_queries_exact=absent,source_sha256=sha(ROOT/'host_reference'/file)))
    report=ROOT/'reports/south_full_lookup_host_v1';report.mkdir(exist_ok=False)
    result=dict(passed=True,platform='Host compiler executing the same portable C lookup; target smoke and fresh feature probes provide M55 coverage',
        records=rows,command=command,shared_library_sha256=sha(library),native_source_sha256={str(p):sha(p)for p in sources},
        SHA_stub_scope='Only the unused boot initializer symbol is stubbed. Native lookup, DEFLATE, UTF-8 prefix reconstruction and record bytes are executed unchanged.')
    write(report/'RESULT.json',result);print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
