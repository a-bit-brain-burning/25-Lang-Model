"""Mandarin interface prototype, never an acoustic Chinese synthesizer.

Requires pypinyin. Standard citation-form pinyin maps to segmented broad IPA;
phrase disambiguation uses the package dictionary. Sandhi, word segmentation,
erhua and text-number normalization are intentionally not claimed complete.
"""
import re
import numpy as np
from .universal import LanguageFrontend,UniversalPhone,encode_phones

INITIALS = {'':'','b':'p','p':'pʰ','m':'m','f':'f','d':'t','t':'tʰ','n':'n','l':'l',
 'g':'k','k':'kʰ','h':'x','j':'tɕ','q':'tɕʰ','x':'ɕ','zh':'tʂ','ch':'tʂʰ','sh':'ʂ','r':'ɻ',
 'z':'ts','c':'tsʰ','s':'s'}
FINALS = {'a':['a'],'o':['ɔ'],'e':['ɤ'],'ai':['a','i'],'ei':['e','i'],'ao':['ɑ','ʊ'],
 'ou':['o','ʊ'],'an':['a','n'],'en':['ə','n'],'ang':['ɑ','ŋ'],'eng':['ə','ŋ'],
 'ong':['ʊ','ŋ'],'i':['i'],'ia':['j','a'],'ie':['j','ɛ'],'iao':['j','ɑ','ʊ'],
 'iou':['j','o','ʊ'],'iu':['j','o','ʊ'],'ian':['j','ɛ','n'],'in':['i','n'],
 'iang':['j','ɑ','ŋ'],'ing':['i','ŋ'],'iong':['j','ʊ','ŋ'],'u':['u'],'ua':['w','a'],
 'uo':['w','ɔ'],'uai':['w','a','i'],'uei':['w','e','i'],'ui':['w','e','i'],
 'uan':['w','a','n'],'uen':['w','ə','n'],'un':['w','ə','n'],'uang':['w','ɑ','ŋ'],
 'ueng':['w','ə','ŋ'],'ü':['y'],'üe':['ɥ','ɛ'],'üan':['ɥ','ɛ','n'],'ün':['y','n'],
 'er':['ə˞']}
TONE_CONTOURS={1:(1.,1.,1.),2:(-.5,0.,1.),3:(-.5,-1.,.5),4:(1.,0.,-1.),5:(0.,0.,0.)}

def split_pinyin(syllable):
    from pypinyin.contrib.tone_convert import to_initials,to_finals
    m=re.fullmatch(r'([a-züv]+)([1-5])',syllable)
    if not m:raise ValueError('Unsupported pinyin syllable: '+syllable)
    base=m.group(1).replace('v','ü')
    initial=to_initials(base,strict=True)
    final=to_finals(base,strict=True).replace('v','ü')
    if initial not in INITIALS or final not in FINALS:
        raise ValueError('Unsupported pinyin initial/final: '+syllable+' '+initial+'/'+final)
    tail=FINALS[final]
    if final=='i' and initial in ('z','c','s'):tail=['ɹ̩']
    if final=='i' and initial in ('zh','ch','sh','r'):tail=['ɻ̩']
    return ([INITIALS[initial]] if initial else [])+tail,int(m.group(2))

class ChineseFrontend(LanguageFrontend):
    language='zh'
    def encode(self,text):
        try:
            from pypinyin import lazy_pinyin,Style,__version__
        except ImportError as e:
            raise ImportError('ChineseFrontend requires pypinyin; install project/configs/universal_frontend_requirements.txt') from e
        if not isinstance(text,str) or not text.strip():raise ValueError('Empty Chinese text')
        if re.search(r'[^\u3400-\u9fff\s，。！？；：、,.!?;:]',text):
            raise ValueError('Prototype accepts Han text and punctuation; normalize digits/Latin text in a language frontend first')
        chunks=list(re.finditer(r'[\u3400-\u9fff]+',text))
        if not chunks:raise ValueError('No Han characters')
        phones=[UniversalPhone('sil')]; syllables=[]; positions=[[0.]*12]
        for ci,match in enumerate(chunks):
            def reject(characters):raise ValueError('No pinyin pronunciation: '+characters)
            tokens=lazy_pinyin(match.group(),style=Style.TONE3,neutral_tone_with_five=True,tone_sandhi=False,errors=reject)
            for ti,token in enumerate(tokens):
                ipas,tone=split_pinyin(token)
                syllables.append({'pinyin':token,'tone':tone,'ipa':ipas})
                before=text[chunks[ci-1].end() if ci else 0:match.start()] if ti==0 else ''
                after=text[match.end():chunks[ci+1].start() if ci+1<len(chunks) else len(text)] if ti==len(tokens)-1 else ''
                def strength(p):return 1. if any(c in '。！？.!?' for c in p) else .66 if any(c in '；：;:' for c in p) else .33 if any(c in '，、,' for c in p) else 0.
                for pi,ipa in enumerate(ipas):
                    phones.append(UniversalPhone(ipa,tone_contour=TONE_CONTOURS[tone],syllable_start=pi==0,syllable_end=pi==len(ipas)-1))
                    row=[0.]*12
                    row[8]=strength(after) if pi==len(ipas)-1 else 0.
                    row[9]=strength(before) if pi==0 else 0.
                    positions.append(row)
        phones.append(UniversalPhone('sil'));positions.append([0.]*12)
        n=len(phones)-2
        for i in range(1,len(phones)-1):
            positions[i][5]=(i-1)/max(n-1,1);positions[i][6]=min(n/200.,2.)
        positions[0][10]=positions[-1][11]=1.
        positions[0][6]=positions[-1][6]=min(n/200.,2.)
        result=encode_phones(phones,np.asarray(positions,np.float32),metadata={
            'language':'zh','text':text,'pypinyin_version':__version__,'syllables':syllables,
            'tone_sandhi_applied':False,'tone_contour_kind':'nominal_citation_template_not_measured_F0',
            'word_segmentation':'unavailable; word context coordinates are zero',
            'acoustic_Chinese_model_available':False})
        return result
