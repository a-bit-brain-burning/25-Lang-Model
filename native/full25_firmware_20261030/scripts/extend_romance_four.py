"""Add selected French phrasing and Romanian nonsyllabic-vowel coordinates."""
import json
import re
import sys
from build_resources import ROOT,write
from build_espeak import VENDOR
from prepare_romance_live import add_cases,write_cases
def main():
    native=ROOT/'native_romance_four';p=native/'src/romance_constants.h';s=p.read_text()
    vocab=json.loads((VENDOR.parent.parent/'int8/assets/vocabulary.json').read_text())
    palette=re.search(r'romance_values\[\]=\{([^}]*)\}',s)[1].split(',');zero=next(i for i,x in enumerate(palette)if float.fromhex(x[:-1])==0)
    a=s.index('static const unsigned char romance_patches');b=s.index('static const RomanceConstant romance_constants')
    patches=re.findall(r'\{(\d+),(\d+)\}',s[a:b]);rows=re.findall(r'\{("[^"\n]*"),(\d+),(\d+),(\d+),(\d+)\}',s[b:]);byname={json.loads(row[0]):row for row in rows}
    for name in('e̯','i̯','o̯'):
        assert name not in byname;base=byname[name[0]];assert int(base[4])==0
        rows.append((json.dumps(name,ensure_ascii=False),str(vocab.get(name,1)),base[2],str(len(patches)),'1'));patches.append(('31',str(zero)))
    rows.sort(key=lambda row:json.loads(row[0]));s=s[:a]+'static const unsigned char romance_patches[][2]={'+','.join('{'+','.join(pair)+'}'for pair in patches)+'};\nstatic const RomanceConstant romance_constants[]={\n'+',\n'.join('{'+','.join(row)+'}'for row in rows)+'\n};\n';p.write_text(s)
    sys.path[:0]=[str(VENDOR/'runtime/current25'),str(VENDOR)]
    from project.src.frontend import french_prosody as fr
    words=sorted(fr.FUNCTION_WORDS);starts=fr.POLICY['clause_starts']
    (native/'src/french_words.h').write_text('static const char*const french_functions[]={'+','.join(json.dumps(x,ensure_ascii=False)for x in words)+'};\nstatic const char*const french_clause_starts[]={'+','.join(json.dumps(x,ensure_ascii=False)for x in starts)+'};\n')
    p=native/'src/romance_dictionary.inc';s=p.read_text()
    helper='''#include "french_words.h"
static int romance_word_in(const char*text,const FourWord*w,const char*const*names,unsigned count,int strip){
 char word[512];unsigned bytes=w->hi-w->lo;if(bytes>=sizeof(word))return 0;memcpy(word,text+w->lo,bytes);
 if(strip)while(bytes&&word[bytes-1]=='\\'')bytes--;word[bytes]=0;
 for(unsigned i=0;i<count;i++)if(!strcmp(word,names[i]))return 1;return 0;
}
static void french_group(const char*normalized,const FourWord*words,unsigned first,unsigned last,const RomanceConstant*const*phones,signed char*stresses){
 int content=0;unsigned target=0;int found=0;
 for(unsigned wi=first;wi<=last;wi++){
  if(!romance_word_in(normalized,words+wi,french_functions,sizeof(french_functions)/sizeof(french_functions[0]),1))content=1;
  for(unsigned i=words[wi].phone_start;i<words[wi].phone_end;i++)if(romance_nucleus(phones[i])&&strncmp(phones[i]->name,"ə",2)){target=i;found=1;}
 }if(content&&found)stresses[target]=1;
}
'''
    s=s.replace('#include "romance_record.h"','#include "romance_record.h"\n'+helper)
    at=' unsigned total=count+wn+1;'
    s=s.replace(at,''' if(!strcmp(language,"fr")){
  memset(stresses,-1,count);unsigned first=0;
  for(unsigned wi=0;wi<wn;wi++){
   if(wi-first>=2&&romance_word_in(normalized,words+wi,french_clause_starts,sizeof(french_clause_starts)/sizeof(french_clause_starts[0]),0)){
    french_group(normalized,words,first,wi-1,phones,stresses);first=wi;
   }
   if(words[wi].after>0){french_group(normalized,words,first,wi,phones,stresses);first=wi+1;}
  }if(first<wn)french_group(normalized,words,first,wn-1,phones,stresses);
 }
'''+at)
    p.write_text(s)
    cases=json.loads((ROOT/'generated/ROMANCE_LIVE_FEATURE_CASES.json').read_text())['cases'];add_cases(cases,['fr','ro']);write_cases(cases)
    write(ROOT/'generated/ROMANCE_FOUR_FEATURE_CASES.json',dict(cases=cases))
if __name__=='__main__':main()
