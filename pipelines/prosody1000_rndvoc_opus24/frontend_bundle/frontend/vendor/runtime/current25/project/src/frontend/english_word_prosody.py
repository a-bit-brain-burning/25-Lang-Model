"""Text-only numeric English word guidance, separate from the base frontend.

No recording, target duration or acoustic statistic is read here. Function
classes are generic closed classes; no whole-sentence or content-word patches.
"""
from __future__ import annotations
import numpy as np
from .english import EnglishFrontend, PHONES, PHONE_TO_ID

PLAN_VERSION='english_word_plan5_pause_v1'
PLAN_NAMES=('log_duration_residual_z','log_energy_residual_z','pitch_level_residual_z','pause_probability','log1p_pause_frames_z')
PLAN_DIM=5
FUNCTION_CLASSES={
    'determiner':frozenset('a an the this that these those each every either neither another any some such'.split()),
    'pronoun':frozenset('i me my mine myself you your yours yourself he him his himself she her hers herself it its itself we us our ours ourselves they them their theirs themselves one'.split()),
    'auxiliary_modal':frozenset("am is are was were be being been have has had do does did can could may might must shall should will would ought".split()),
    'preposition':frozenset('about above across after against along among around at before behind below beneath beside between beyond by despite down during except for from in inside into near of off on onto opposite outside over past since through throughout till to toward towards under underneath until up upon via with within without'.split()),
    'conjunction':frozenset('and but or nor so yet although as because if lest once since than though unless until when whenever whereas whether while'.split()),
    'negation':frozenset("not no never neither nor cannot don't doesn't didn't isn't aren't wasn't weren't hasn't haven't hadn't won't wouldn't shouldn't couldn't mustn't".split()),
    'interrogative':frozenset('who whom whose what which when where why how'.split()),
    'quantifier':frozenset('all both each enough few fewer little less many more most much several some any none'.split()),
}
SCALARS=('character_length_div20','phone_length_div12','vowels_div6','primary_stresses_div3','secondary_stresses_div3',
         'first_primary_position','dictionary_stress_known','approximate_pronunciation','word_position','words_remaining_fraction',
         'sentence_words_div50','sentence_phones_div200','punctuation_before','punctuation_after','first_word','last_word',
         'apostrophe','vowel_phone_fraction')
BASE_FEATURE_NAMES=SCALARS+tuple('function_'+k for k in FUNCTION_CLASSES)+tuple('phone_fraction_'+p for p in PHONES)
FEATURE_NAMES=tuple(prefix+'/'+name for prefix in ('current','previous','next')for name in BASE_FEATURE_NAMES)
FEATURE_DIM=len(FEATURE_NAMES)
VOWELS=frozenset('AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW'.split())


def encode_frontend(front):
    """Return W×F text features and the exact original phone-to-word mapping."""
    words=front['words'];mapping=np.asarray(front['word_indices'],np.int64)
    pause=np.asarray(front.get('internal_pause_mask',np.zeros(len(mapping),bool)),bool)
    ids=np.asarray(front['phone_ids'],np.int64);text=np.asarray(front['text_features'],np.float32)
    stress=np.asarray(front['stress'],np.int64);n=len(ids);w=len(words)
    if mapping.shape!=(n,)or text.shape!=(n,12)or stress.shape!=(n,):raise ValueError('Malformed English word frontend')
    if pause.shape!=(n,):raise ValueError('Pause mask differs from unit axis')
    if 'pause_after_word'in front and np.any(mapping[pause]!=np.asarray(front['pause_after_word'])[pause]):raise ValueError('Pause must belong to its left word')
    if np.any((mapping<-1)|(mapping>=w)):raise ValueError('Phone/word mapping outside text')
    base=np.zeros((w,len(BASE_FEATURE_NAMES)),np.float32);ranges=[];boundary=np.zeros(w,np.float32)
    for wi,word in enumerate(words):
        positions=np.flatnonzero((mapping==wi)&~pause)
        if not len(positions)or not np.array_equal(positions,np.arange(positions[0],positions[-1]+1)):
            raise ValueError('Noncontiguous or absent word phones')
        a,b=int(positions[0]),int(positions[-1]+1);ranges.append([a,b]);names=front['phones'][a:b]
        vowel=np.array([p in VOWELS for p in names]);local=stress[a:b];primary=np.flatnonzero(local==1)
        before=float(text[a,9]);after=float(text[b-1,8]);boundary[wi]=after
        scalar=[min(len(word)/20,2),min((b-a)/12,2),min(vowel.sum()/6,2),min((local==1).sum()/3,2),min((local==2).sum()/3,2),
                float(primary[0]/max(1,b-a-1))if len(primary)else 0.,float(front['stress_sources'][wi]=='cmudict'),
                float(front['pronunciation_sources'][wi]=='approximate_spelling_rules'),wi/max(1,w-1),(w-1-wi)/max(1,w-1),
                min(w/50,2),min(((mapping>=0)&~pause).sum()/200,2),before,after,float(wi==0),float(wi==w-1),float("'"in word),float(vowel.mean())]
        lexical=[float(word in values)for values in FUNCTION_CLASSES.values()]
        histogram=[names.count(phone)/(b-a)for phone in PHONES]
        base[wi]=np.array(scalar+lexical+histogram,np.float32)
    previous=np.zeros_like(base);previous[1:]=base[:-1];following=np.zeros_like(base);following[:-1]=base[1:]
    return {'numeric':np.concatenate((base,previous,following),axis=1),'boundary':boundary,
            'word_indices':mapping.copy(),'word_phone_ranges':np.asarray(ranges,np.int64),'words':list(words),
            'text_only':True,'source':'current_EnglishFrontend_dictionary_and_text_only','plan_version':PLAN_VERSION}


def expanded_frontend(raw):
    """Preserve dictionary word metadata while adding the shared pause topology."""
    from .english_pause_units import insert_word_pauses
    expanded=insert_word_pauses(raw);front={**raw,**expanded}
    names={v:k for k,v in PHONE_TO_ID.items()};front['phones']=[names[int(i)]for i in front['phone_ids']];front['phonemes']=front['phones']
    return front


def encode_text(text,frontend=None):
    front=expanded_frontend((frontend or EnglishFrontend()).encode(text))
    return front,encode_frontend(front)


def zero_plan(front):
    return {'word_plan':np.zeros((len(front['words']),PLAN_DIM),np.float32),
            'unit_plan':np.zeros((len(front['phone_ids']),PLAN_DIM),np.float32),
            'plan_source':'ZERO_NEUTRAL','plan_version':PLAN_VERSION}


def plan_for_units(word_plan,word_indices):
    plan=np.asarray(word_plan,np.float32);mapping=np.asarray(word_indices,np.int64)
    if plan.ndim!=2 or plan.shape[1]!=PLAN_DIM or mapping.ndim!=1:raise ValueError('Expected word_plan[W,5] and phone_to_word[N]')
    if not np.isfinite(plan).all()or np.any(mapping<-1)or np.any(mapping>=len(plan)):raise ValueError('Invalid plan/mapping')
    result=np.zeros((len(mapping),PLAN_DIM),np.float32);valid=mapping>=0;result[valid]=plan[mapping[valid]]
    return result


def validate_record(front,record):
    """Exact text/phone/stress identity is required before using native labels."""
    checks={name:bool(np.array_equal(np.asarray(front[name]),np.asarray(record[name])))
            for name in ('phone_ids','text_features','stress')}
    for name in ('source_indices','word_indices','internal_pause_mask','pause_after_word'):
        if name in record:checks[name]=bool(np.array_equal(np.asarray(front[name]),np.asarray(record[name])))
    if not all(checks.values()):raise ValueError('Text/reference mapping mismatch: '+str(checks))
    return checks
