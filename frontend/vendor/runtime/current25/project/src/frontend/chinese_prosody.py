"""Mandarin prosody adapter; frozen segment/context contract stays independent.

Supplied corpus pinyin is authoritative. Auxiliary prosody coordinates encode
linguistic categories, never prescribed measured F0, duration, or prominence.
"""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
import re
import numpy as np
from .chinese_training import ChineseTrainingFrontend, normalize_pinyin_token, normalize_text
from .universal_chinese import TONE_CONTOURS

VERSION = 'chinese_prosody_frontend_v2'
PROSODY_FEATURE_NAMES = ('citation_start','citation_mid','citation_end','citation_known',
    'previous_surface_start','previous_surface_mid','previous_surface_end','previous_surface_known',
    'neutral_or_reduction_cue','citation_surface_difference','question_punctuation_cue','exclamation_punctuation_cue')
PROSODY_DIM = len(PROSODY_FEATURE_NAMES)
_KINSHIP_NEUTRAL = frozenset(('妈妈','爸爸','哥哥','姐姐','弟弟','妹妹','爷爷','奶奶','叔叔','舅舅','姑姑','伯伯','婶婶'))
_ASPECT_GUO = frozenset(('看过','听过','读过','学过','见过','说过','喝过','吃过','去过','来过','用过','做过','买过','写过','玩过','住过','试过','睡过'))


def _legacy_polyphone_policy():
    from .chinese_polyphone import current_mode
    return current_mode()=='legacy'


def _adverbial_de_context(index,syllables,words):
    """A UV tag alone is insufficient: require a local modifier + predicate.

    This protects content-word di (including rare words split by the tokenizer)
    while retaining ordinary adjective/adverb/reduplicated modifiers.
    """
    if index==0 or index+1>=len(syllables) or syllables[index-1]['boundary_after_level']!=0 or syllables[index]['boundary_after_level']!=0:return False
    previous=words[index-1];tag=previous.get('context_pos') or previous.get('pos') or '';word=previous['word']
    context_tag=words[index].get('context_pos') or '';context_word=words[index].get('context_word') or ''
    if context_word.endswith('地') and len(context_word)>1 and context_tag.startswith(('d','a')):return True
    if context_tag!='uv':return False
    modifier=tag.startswith(('a','d','i','z','o')) or (tag.startswith('v') and len(word)>1) or (len(word)>=2 and word[-1]==word[-2])
    if not modifier:return False
    seen=set()
    for j in range(index+1,min(index+6,len(syllables))):
        if j>index+1 and syllables[j-1]['boundary_after_level']!=0:return False
        group=tuple(words[j]['indices'])
        if group in seen:continue
        seen.add(group);following=words[j];nxt=following.get('context_pos') or following.get('pos') or ''
        if nxt.startswith(('v','a')):return True
        if not nxt.startswith(('d','m')):return False
    return False


def _modal_dei_context(index,syllables,words):
    """A free 得 before a predicate may have intervening adverbs/disposal 把.

    Lexical multi-character words never enter this rule. A directly preceding
    predicate keeps the earlier complement rule; ambiguous syntax is retained.
    """
    if index>0 and syllables[index-1]['boundary_after_level']==0:
        # Do not infer modality from an arbitrary isolated POS tag. A preceding
        # personal subject/time phrase (possibly separated by adverbs) is the
        # conservative extension of the existing personal-pronoun rule.
        j=index-1;seen_left=set();licensed=False
        while j>=0 and index-j<=8:
            group=tuple(words[j]['indices'])
            if group in seen_left:j-=1;continue
            seen_left.add(group);word=words[j]['word'];tag=words[j].get('pos') or ''
            if (tag.startswith('r') and word[0] in '我你他她它咱您') or tag.startswith('t'):
                licensed=True;break
            if not tag.startswith('d'):break
            if words[j]['indices'][0]==0:licensed=True;break
            j=words[j]['indices'][0]-1
            if j>=0 and syllables[j]['boundary_after_level']!=0:licensed=True;break
        if not licensed:return False
    seen=set()
    for j in range(index+1,min(index+6,len(syllables))):
        if syllables[j-1]['boundary_after_level']!=0:return False
        group=tuple(words[j]['indices'])
        if group in seen:continue
        seen.add(group);word=words[j]['word'];tag=words[j].get('pos') or ''
        if word in ('把','被'):return True
        if tag.startswith(('v','a')):return True
        if not tag.startswith('d'):return False
    return False


@lru_cache(maxsize=8192)
def _character_candidates(character):
    from pypinyin import pinyin, Style
    values=pinyin(character,style=Style.TONE3,heteronym=True,neutral_tone_with_five=True,errors='default')[0]
    return tuple(normalize_pinyin_token(x) for x in values if re.fullmatch(r'[a-zvü]+[1-5]',x))


def _base(token):
    root=token[:-1]
    return root[:-1] if root.endswith('r') and root!='er' else root


def _word_info(result):
    syllables=result['metadata']['syllables'];groups={}
    for i,s in enumerate(syllables):groups.setdefault(s['group_index'],[]).append(i)
    from .chinese_training import _tokenizer
    tokenizer=_tokenizer()
    words={}
    for indices in groups.values():
        word=''.join(syllables[i]['character'] for i in indices)
        # POS is metadata/guard evidence only, never a shared-model language rule.
        import jieba.posseg as posseg
        tagged=list(posseg.cut(word,HMM=False))
        pos=tagged[0].flag if len(tagged)==1 and tagged[0].word==word else None
        for i in indices:words[i]={'word':word,'pos':pos,'indices':indices}
    # Tag full clauses as well. Tagging a standalone 地 always returns UV and
    # loses the distinction between a particle and a content noun in context.
    context_at={};position=0
    text=re.sub(r'#[1-4]','',result['metadata']['normalized_text'])
    for run in re.finditer(r'[\u3400-\u9fff]+',text):
        for token in posseg.cut(run[0],HMM=False):
            for offset in range(len(token.word)):context_at[position+offset]=(token.word,token.flag)
            position+=len(token.word)
    position=0
    for i,s in enumerate(syllables):
        context_word,context_pos=context_at.get(position,(None,None))
        words[i].update(context_word=context_word,context_pos=context_pos);position+=len(s['character'])
    return words


def _neutral_policy(syllables,tokens,words):
    """Small grammatical/lexical cases; unknown emphasis never becomes a label."""
    result=list(tokens);types=[None]*len(tokens);sources=[[] for _ in tokens]
    for i,(s,token) in enumerate(zip(syllables,tokens)):
        char=s['character'];word=words[i]['word'];pos=words[i]['pos'];root=_base(token)
        if token.endswith('5'):
            types[i]='dictionary_neutral';sources[i].append('pypinyin_dictionary_neutral')
        if word in _KINSHIP_NEUTRAL and i==words[i]['indices'][-1]:
            result[i]=token[:-1]+'5';types[i]='lexical_neutral';sources[i].append('closed_kinship_lexicon')
        elif len(word)==2 and word[0]==word[1] and pos and pos.startswith('v') and i==words[i]['indices'][-1]:
            result[i]=token[:-1]+'5';types[i]='grammatical_neutral';sources[i].append('verb_AA_reduplication')
        elif char=='地' and word=='地' and pos=='uv' and (root=='de' or _legacy_polyphone_policy() or _adverbial_de_context(i,syllables,words)):
            result[i]='de5';types[i]='grammatical_neutral';sources[i].append('adverbial_de_particle_POS_uv')
        elif char=='的' and word=='的' and root=='de':
            result[i]='de5';types[i]='grammatical_neutral';sources[i].append('structural_de_particle')
        elif char=='得' and word=='得' and root=='de':
            prev=words.get(i-1,{}).get('pos');nxt=words.get(i+1,{}).get('pos')
            continuous=i>0 and syllables[i-1]['boundary_after_level']==0
            if continuous and prev and prev.startswith(('v','a')):
                result[i]='de5';types[i]='grammatical_neutral';sources[i].append('postpredicate_complement_de')
            elif continuous and syllables[i-1]['character'] in '我你他她它咱您' and nxt and nxt.startswith('v'):
                result[i]='dei3';types[i]=None;sources[i].append('restricted_modal_dei_context')
            elif not _legacy_polyphone_policy() and _modal_dei_context(i,syllables,words):
                result[i]='dei3';types[i]=None;sources[i].append('free_modal_before_predicate_with_optional_adverbs')
        elif char=='过' and word in _ASPECT_GUO and root=='guo':
            result[i]='guo5';types[i]='grammatical_neutral';sources[i].append('closed_experiential_aspect_context')
        elif char in ('了','着') and root in ('le','zhe'):
            result[i]=root+'5';types[i]='grammatical_neutral';sources[i].append('aspect_particle_pronunciation_guard')
        elif char in ('吗','呢','吧','啊','嘛','啦') and word==char and (i==len(tokens)-1 or s['boundary_after_level']>=2):
            result[i]=token[:-1]+'5';types[i]='grammatical_neutral';sources[i].append('sentence_particle_at_boundary')
        elif char=='们' and root=='men' and (token.endswith('5') or pos=='k' or word in ('我们','你们','他们','她们','它们','咱们','人们')):
            result[i]='men5';types[i]='grammatical_neutral';sources[i].append('plural_men_suffix_guard')
    return result,types,sources


def _sentence_cues(text,syllables):
    # Question/exclamation punctuation is a conditioning cue, not a pitch command.
    endpoints=[];count=0
    for c in re.sub(r'#[1-4]','',text):
        if '\u3400'<=c<='\u9fff':count+=1
        elif c in '。.!?！？；;':endpoints.append((count,'question' if c in '?？' else 'exclamation' if c in '!！' else 'statement'))
    kinds=[];position=0
    for s in syllables:
        position+=len(s['character']);kinds.append(next((kind for end,kind in endpoints if end>=position),'unmarked'))
    return kinds


def _citation(s,token,word,raw_dictionary):
    """Known only with compatible segmental reading and lexical evidence."""
    if raw_dictionary is None:return None,False,'dictionary_lookup_unavailable'
    candidate=normalize_pinyin_token(token);root=_base(candidate);rawroot=_base(raw_dictionary)
    if root!=rawroot:
        # POS-guarded grammatical particles can have another dictionary default.
        if candidate in ('de5','dei3') and s['character'] in ('地','得'):
            return int(candidate[-1]),True,'guarded_grammatical_reading'
        return None,False,'segmental_reading_differs'
    if s['character']=='一' and root=='yi':return 1,True,'lexical_yi_citation'
    if s['character']=='不' and root=='bu':return 4,True,'lexical_bu_citation'
    if candidate.endswith('5'):return 5,True,'dictionary_or_grammatical_neutral'
    options={int(x[-1]) for x in _character_candidates(s['character'][0]) if _base(x)==root}
    if len(options)==1:return next(iter(options)),True,'single_lexical_tone_for_reading'
    from pypinyin.constants import PHRASES_DICT
    if word['word'] in PHRASES_DICT:
        return int(candidate[-1]),True,'phrase_dictionary_disambiguated'
    return None,False,'ambiguous_lexical_tone'


class ChineseProsodyFrontend(ChineseTrainingFrontend):
    """Adds prosody_features[N,12]; supplied-pinyin base arrays stay bitexact."""
    def __init__(self,*,digit_policy='reject',tone_sandhi=True,erhua_policy='lexicon'):
        super().__init__(digit_policy=digit_policy,tone_sandhi=False,use_corpus_prosody=False,erhua_policy=erhua_policy)
        self.apply_prosodic_sandhi=bool(tone_sandhi)

    def encode(self,text,pinyin=None):
        supplied=pinyin is not None
        base=ChineseTrainingFrontend(digit_policy=self.digit_policy,tone_sandhi=False,erhua_policy=self.erhua_policy)
        try:dictionary=base.encode(text)
        except ValueError:
            if not supplied:raise
            dictionary=None
        source=base.encode(text,pinyin) if supplied else dictionary
        syllables=source['metadata']['syllables'];words=_word_info(source)
        # Map corpus fused-erhua syllables to the first character of its source.
        dict_at={};position=0
        if dictionary:
            for s in dictionary['metadata']['syllables']:
                dict_at[position]=s['pinyin'];position+=len(s['character'])
        raw=[];position=0
        for s in syllables:raw.append(dict_at.get(position));position+=len(s['character'])
        dictionary_tokens=[r if r is not None else s['pinyin'] for r,s in zip(raw,syllables)]
        lexical,neutral_types,neutral_sources=_neutral_policy(syllables,dictionary_tokens,words)
        reading_events=dictionary['metadata'].get('pronunciation_events',[]) if dictionary else []
        hint_at={e['start']:e['pinyin'] for e in reading_events if e['source']=='explicit_citation_hint'}
        position=0
        for i,s in enumerate(syllables):
            if position in hint_at:
                lexical[i]=dictionary_tokens[i];neutral_types[i]='dictionary_neutral' if lexical[i].endswith('5') else None
                neutral_sources[i]=['explicit_citation_hint']
            position+=len(s['character'])
        citation=[_citation(s,t,words[i],raw[i]) for i,(s,t) in enumerate(zip(syllables,lexical))]
        resolved_at={}
        for event in reading_events:
            if event['source'] in ('public_large_pinyin_add_only','train_supported_public_word_correction','explicit_surname_context_public_lexicon'):
                for offset in range(len(event['word'])):resolved_at[event['start']+offset]=event['source']
            elif event['source']=='contextual_polyphone_model':resolved_at[event['start']]=event['source']
        position=0
        for i,s in enumerate(syllables):
            if position in resolved_at and lexical[i]==dictionary_tokens[i]:
                citation[i]=(int(lexical[i][-1]),True,resolved_at[position])
            if position in hint_at:citation[i]=(int(hint_at[position][-1]),True,'explicit_citation_hint')
            position+=len(s['character'])
        if supplied:
            for i,(s,(tone,known,cause)) in enumerate(zip(syllables,citation)):
                actual=source['pinyin'][i]
                if _base(actual)!=_base(lexical[i]):
                    citation[i]=(None,False,'annotated_segmental_reading_differs_from_dictionary')
                elif known and int(actual[-1])!=tone:
                    surface_tone=int(actual[-1]);char=s['character']
                    compatible=(tone==3 and surface_tone==2) or surface_tone==5 or (char=='一' and tone==1 and surface_tone in (2,4)) or (char=='不' and tone==4 and surface_tone==2)
                    if not compatible:citation[i]=(None,False,'annotated_tone_not_explained_by_conservative_sandhi')
        if supplied:
            surface=list(source['pinyin']);sandhi_sources=[['authoritative_corpus_annotation_no_second_sandhi'] for _ in surface]
            uncertain=[False]*len(surface)
        else:
            surface=list(lexical);sandhi_sources=[list(s) for s in neutral_sources];uncertain=[False]*len(surface)
            if self.apply_prosodic_sandhi:
                surface,sandhi_sources,uncertain=self._surface_rules(syllables,surface,words,sandhi_sources)
            source=base.encode(text,surface)
            syllables=source['metadata']['syllables']
        cues=_sentence_cues(normalize_text(text,self.digit_policy),syllables)
        auxiliary=np.zeros((len(source['ipa']),PROSODY_DIM),np.float32)
        for i,s in enumerate(syllables):
            tone=int(surface[i][-1]);ct,known,citation_source=citation[i]
            previous=int(surface[i-1][-1]) if i>0 and syllables[i-1]['boundary_after_level']==0 else None
            reduced=tone==5
            row=np.zeros(PROSODY_DIM,np.float32)
            if known:row[:4]=(*TONE_CONTOURS[ct],1.)
            if previous is not None:row[4:8]=(*TONE_CONTOURS[previous],1.)
            row[8]=reduced;row[9]=known and ct!=tone;row[10]=cues[i]=='question';row[11]=cues[i]=='exclamation'
            auxiliary[s['phone_start']:s['phone_end']]=row
            neutral_type=neutral_types[i] if reduced else None
            if reduced and neutral_type is None:neutral_type='corpus_surface_neutral_unresolved'
            s.update({'dictionary_pinyin':raw[i],'citation_pinyin':None if not known else _base(surface[i])+str(ct),
                'citation_tone':ct,'citation_known':known,'citation_source':citation_source,
                'surface_tone_category':tone,'surface_pinyin':surface[i],'sandhi_source':sandhi_sources[i],
                'sandhi_uncertain':uncertain[i],'lexical_neutral':neutral_types[i] in ('dictionary_neutral','lexical_neutral'),
                'neutral_type':neutral_type,'contextual_weak':None if neutral_type=='corpus_surface_neutral_unresolved' else neutral_types[i]=='grammatical_neutral' and reduced,
                'previous_surface_tone':previous,'sentence_type':cues[i],'focus_status':'unknown',
                'prominence_status':'not_inferred','stress_target_known':False,
                'erhua_status':('annotated_fused' if supplied else 'lexicon_fused') if len(s['character'])>1 and surface[i][:-1].endswith('r') else 'not_automatically_fused',
                'F0_target_kind':'learned_from_recording_not_fixed_contour'})
        source['prosody_features']=auxiliary;source['prosody_feature_names']=PROSODY_FEATURE_NAMES
        source['metadata'].update({'frontend_version':VERSION if supplied or self.erhua_policy=='separate' else 'chinese_prosody_frontend_v3_erhua','prosody_feature_version':'mandarin_prosody12_v1',
            'prosody_features_dimension':PROSODY_DIM,'pinyin_source':'corpus_annotation' if supplied else 'pypinyin_phrase_dictionary_prosody_v2',
            'provided_pinyin_authoritative':supplied,'tone_sandhi_applied':not supplied and self.apply_prosodic_sandhi,
            'sentence_pitch_rule':'none; punctuation cues only','focus_annotation_available':False,
            'shared_context22_semantics_changed':False,'citation_contour_kind':'nominal_category_not_measured_F0'})
        if not supplied:
            source['metadata'].update({'erhua_policy':self.erhua_policy,
                'pronunciation_events':reading_events,
                'erhua_fusions':dictionary['metadata'].get('erhua_fusions',[])})
            from .chinese_polyphone import current_mode
            if current_mode()!='legacy':
                source['metadata'].update(frontend_version='chinese_prosody_frontend_v4_polyphone',
                    polyphone_policy=current_mode(),polyphone_resource='public_large_pinyin_citation_validated_v1',
                    pinyin_source='text_lexical_context_resolution_then_neutral_sandhi')
        return source

    @staticmethod
    def _surface_rules(syllables,tokens,words,sources):
        result=list(tokens);uncertain=[False]*len(tokens)
        for i,(s,t) in enumerate(zip(syllables,tokens)):
            next_tone=int(tokens[i+1][-1]) if i+1<len(tokens) and s['boundary_after_level']==0 else None
            char=s['character'];root=_base(t)
            if char in ('一','不') and i>0 and i+1<len(tokens) and syllables[i-1]['character']==syllables[i+1]['character'] and syllables[i-1]['boundary_after_level']==0 and s['boundary_after_level']==0:
                result[i]=root+'5';sources[i].append('A_yi_bu_A_neutral')
            elif char=='一' and root=='yi':
                numerical=(i>0 and syllables[i-1]['character'] in '第零一二三四五六七八九十百千万亿') or (i+1<len(tokens) and syllables[i+1]['character'] in '零一二三四五六七八九十百千万亿')
                tone=1 if numerical or next_tone is None else 2 if next_tone==4 else 4 if next_tone in (1,2,3) else 1
                result[i]=root+str(tone);sources[i].append('yi_citation_or_following_tone')
            elif char=='不' and root=='bu':
                result[i]=root+('2' if next_tone==4 else '4');sources[i].append('bu_before_fourth' if next_tone==4 else 'bu_citation')
        i=0
        while i<len(tokens):
            if not tokens[i].endswith('3'):i+=1;continue
            end=i+1
            while end<len(tokens) and tokens[end].endswith('3') and syllables[end-1]['boundary_after_level']==0:end+=1
            if end-i==2:
                result[i]=result[i][:-1]+'2';sources[i].append('adjacent_third_tone_pair')
            elif end-i>=3:
                # Preserve ambiguity instead of claiming jieba words identify
                # syntactic/prosodic feet. Only a final local pair is realized;
                # earlier tones retain lexical categories and uncertainty.
                result[end-2]=result[end-2][:-1]+'2';sources[end-2].append('conservative_final_pair_in_ambiguous_third_run')
                for j in range(i,end):uncertain[j]=True;sources[j].append('long_third_run_prosodic_grouping_unknown')
            i=end
        return result,sources,uncertain
