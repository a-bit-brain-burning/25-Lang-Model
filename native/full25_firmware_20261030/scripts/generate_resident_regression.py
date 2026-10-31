"""Test-only feature hashes run inside the complete resident firmware budget."""
import json
from build_resources import ROOT
def main():
    native=ROOT/'native_text_regression';cases=json.loads((ROOT/'generated/FRESH_TEXT_FEATURE_CASES.json').read_text())['cases']
    languages='bg cs da de el en es et fi fr hr hu it lt lv nl no pl pt ro sk sl sv tr zh'.split()
    lines=['/* Test-only hashes; excluded from normal firmware builds. */','#include "romfs.h"','static int resident_frontend_regression(void){',
        ' static const struct {unsigned language,count,phonetic,context;const char*text;unsigned char sha256[4][32];} tests[]={']
    for row in cases:
        hashes=','.join('{'+','.join(str(b)for b in bytes.fromhex(row['sha256'][k]))+'}'for k in('phones','pause','phonetic','context'))
        context=row.get('context_dim',34 if row['language']in ['fi','et','hr','lv','lt','bg','sl']else 22)
        lines.append('{'+','.join(map(str,[languages.index(row['language']),row['count'],row['dim'],context]))+','+json.dumps(row['text'],ensure_ascii=False)+',{'+hashes+'}},')
    lines+=[' };',' size_t mark=bench_arena_mark();',' for(unsigned i=0;i<sizeof(tests)/sizeof(tests[0]);i++){',
        '  n25_current_norm=n25_language_norms+tests[i].language;',
        '  int rc=n25_text_frontend(tests[i].text,&n25_text_input);',
        '  if(rc||n25_text_input.count!=tests[i].count){bench_metric("resident_feature_failed_case",i);bench_metric("resident_feature_error",(unsigned)-rc);return -1;}',
        '  const void*data[]={n25_text_input.phones,n25_text_input.pause,n25_text_input.phonetic,n25_text_input.context};',
        '  unsigned bytes[]={tests[i].count*4,tests[i].count*4,tests[i].count*tests[i].phonetic*4,tests[i].count*tests[i].context*4};',
        '  for(unsigned k=0;k<4;k++){unsigned char digest[32];storage_sha256(data[k],bytes[k],digest);if(memcmp(digest,tests[i].sha256[k],32)){bench_metric("resident_feature_failed_case",i);bench_metric("resident_feature_array",k);return -2;}}',
        '  if(bench_arena_mark()!=mark)return -3;',
        ' }',' bench_metric("resident_raw_text_feature_cases_exact",sizeof(tests)/sizeof(tests[0]));bench_metric("resident_raw_text_feature_heap_peak",n25_romfs_heap_peak());return 0;','}']
    (native/'src/resident_feature_regression.inc').write_text('\n'.join(lines)+'\n')
    p=native/'src/multilang_driver.inc';s=p.read_text();assert 'N25_FRONTEND_REGRESSION'not in s
    s=s.replace('int main(void){','#ifdef N25_FRONTEND_REGRESSION\n#include "resident_feature_regression.inc"\n#endif\nint main(void){')
    s=s.replace('static const struct {unsigned language;const char *text;} cases[]={','#ifdef N25_FRONTEND_REGRESSION\n if(resident_frontend_regression())fail("RESIDENT_RAW_TEXT_FEATURE_REGRESSION");\n#endif\n static const struct {unsigned language;const char *text;} cases[]={')
    p.write_text(s)
if __name__=='__main__':main()
