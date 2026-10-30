"""Configurable abbreviation decisions before the unchanged acoustic frontend."""
from pathlib import Path
from decimal import Decimal
import copy,hashlib,json,re,unicodedata

ROOT=Path(__file__).resolve().parent
LETTER_NAMES=dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ','ay bee see dee ee ef gee aitch eye jay kay el em en oh pee cue ar ess tee you vee double_you ex why zee'.split()))
EN_DIGITS='zero one two three four five six seven eight nine'.split()
_WORD=r"[A-Za-z]+(?:['’][A-Za-z]+)*"
_DOT=r'[A-Za-z](?:\.[A-Za-z])+(?:\.)?'
_SPACE=r'(?:[A-Z][ \t]+)+[A-Z]'
_CODE=r'(?:[A-Za-z]+(?:[-_./]?\d+)[A-Za-z0-9]*(?:[._-]\d+)*|[A-Za-z]{1,3}[ \t]+\d+[A-Za-z0-9]*|\d+[A-Za-z]+[A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*|[A-Za-z]+(?:[-_][A-Za-z0-9]+)+)'

def clean(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text.replace('º','°')) if unicodedata.category(c)!='Cf')

def _merge(base,updates):
    for key,value in updates.items():
        if isinstance(value,dict) and isinstance(base.get(key),dict):_merge(base[key],value)
        elif isinstance(value,list) and key in ['initialisms','case_sensitive_initialisms','lexical_words']:
            base[key]=list(dict.fromkeys(base.get(key,[])+value))
        else:base[key]=copy.deepcopy(value)

class ReadingPolicy:
    def __init__(self,config=None,mode='auto'):
        if mode not in ['auto','spell','word']:raise ValueError('abbreviation mode must be auto, spell or word')
        self.data=json.loads((ROOT/'frontend_profiles/default.json').read_text())
        if config is not None:
            supplied=json.loads(Path(config).read_text()) if isinstance(config,(str,Path)) else config
            if not isinstance(supplied,dict):raise ValueError('Frontend configuration must be a JSON object')
            if set(supplied)-set(self.data):raise ValueError('Unknown frontend configuration keys: '+str(sorted(set(supplied)-set(self.data))))
            _merge(self.data,supplied)
        if self.data['version']!=1:raise ValueError('Unsupported frontend configuration version')
        self.mode=mode
        for key in ['initialisms','case_sensitive_initialisms','lexical_words']:
            if not isinstance(self.data[key],list) or any(not isinstance(v,str) or not v.isascii() or not v.isalpha() for v in self.data[key]):raise ValueError(key+' must be a list of ASCII words')
        if not isinstance(self.data['overrides'],dict):raise ValueError('overrides must be an object keyed by language')
        for language,entries in self.data['overrides'].items():
            if not isinstance(entries,dict):raise ValueError('Each language override must be an object')
            for key,value in entries.items():
                if not isinstance(key,str) or not key or any(0xe000<=ord(c)<=0xf8ff for c in key):raise ValueError('Invalid override key')
                if not isinstance(value,dict) or value.get('mode') not in ['spell','word','expand']:raise ValueError('Override mode must be spell, word or expand: '+key)
                if value['mode']=='expand' and not value.get('spoken'):raise ValueError('Expanded reading needs spoken text: '+key)
                if 'spoken' in value and (not isinstance(value['spoken'],str) or not value['spoken'].strip()):raise ValueError('Invalid spoken reading: '+key)
                if value['mode']=='word' and 'spoken' in value and re.search(r'[^A-Za-z\s\x27.,!?-]',value['spoken']):raise ValueError('Word mode needs English spelling; use expand for Chinese text: '+key)
        self.digest=hashlib.sha256(json.dumps(dict(profile=self.data,mode=mode),ensure_ascii=False,sort_keys=True).encode()).hexdigest()

    def identity(self):return dict(version=1,profile_sha256=self.digest,abbreviation_mode=self.mode)

    def _patterns(self,language):
        overrides=self.data['overrides'].get(language,{})
        builtins=set(self.data['specials'])|set(self.data['abbreviations'].get(language,{}))
        end=lambda value:r'(?![A-Za-z0-9])' if value[-1].isalnum() else ''
        custom=[re.escape(x)+end(x) for x in sorted(overrides,key=len,reverse=True)]
        regular=[r'(?i:'+re.escape(x)+')'+end(x) for x in sorted(builtins,key=len,reverse=True)]
        literals='|'.join(custom+regular) or r'(?!)'
        units='|'.join(re.escape(x) for x in sorted(self.data['units'],key=len,reverse=True))
        unit=r'\d[\d,]*(?:\.\d+)?[ \t]*(?:'+units+r')(?![A-Za-z0-9])'
        code_pattern=_CODE
        if language=='en':
            code_pattern=code_pattern.replace(r'[A-Za-z]{1,3}[ \t]+\d',r'(?!(?i:IS|AM|ARE|WAS|HAS|HAD|CAN|NOT|THE|AND|FOR|OF|IN|ON|AT|TO|AS|BE|BY|DO|IF|MY|OUR|WE|HE|SHE|IT|ITS|YOU|SEE|SAY|ALL|GET)\b)[A-Z]{1,3}[ \t]+\d')
        return re.compile(r'(?<![A-Za-z0-9])(?P<literal>'+literals+r')|(?<![A-Za-z0-9])(?P<unit>'+unit+r')|(?<![A-Za-z0-9])(?P<dotted>'+_DOT+r')(?![A-Za-z0-9])|(?<![A-Za-z0-9])(?P<spaced>'+_SPACE+r')(?![A-Za-z0-9])|(?<![A-Za-z0-9])(?P<code>'+code_pattern+r')(?![A-Za-z0-9])|(?P<word>'+_WORD+r')')

    def scan(self,text,language,lexical=None):
        """Offsets refer to the NFKC source; literal overrides are case-sensitive."""
        text=clean(text);events=[];overrides=self.data['overrides'].get(language,{})
        for match in self._patterns(language).finditer(text):
            raw=match[0];kind=match.lastgroup;letters=re.sub('[^A-Za-z0-9]','',raw)
            item=dict(start=match.start(),end=match.end(),source=raw)
            if raw in overrides:
                value=overrides[raw];mode=value['mode'];spoken=value.get('spoken',raw.lower())
                if mode=='spell' and not re.fullmatch(r'[A-Za-z0-9 ._-]+',raw):raise ValueError('Spelling requires letters or digits: '+raw)
                item.update(mode=mode,value=letters if mode=='spell' else spoken,rule='custom_override')
            elif any(raw.casefold()==k.casefold() for k in self.data['specials']):
                key=next(k for k in self.data['specials'] if raw.casefold()==k.casefold())
                spoken=self.data['specials'][key].get(language)
                if spoken is None:continue
                item.update(mode='expand',value=spoken,rule='special_reading')
            elif any(raw.casefold()==k.casefold() for k in self.data['abbreviations'].get(language,{})):
                key=next(k for k in self.data['abbreviations'][language] if raw.casefold()==k.casefold())
                item.update(mode='expand',value=self.data['abbreviations'][language][key],rule='word_abbreviation')
            elif kind=='unit':
                value=re.fullmatch(r'(\d[\d,]*(?:\.\d+)?)[ \t]*(.+)',raw);number,unit=value.groups();reading=self.data['units'][unit][language]
                if language=='en':reading=reading[0 if Decimal(number.replace(',',''))==1 else 1]
                item.update(mode='expand',value=number+' '+reading,rule='number_and_unit')
            elif kind=='code':
                prefix=re.match(r'[A-Za-z]+',raw)
                prefix=prefix[0] if prefix else ''
                if not any(c.isdigit() for c in raw) and not raw.isupper() and not any(p.upper() in self.data['initialisms'] for p in re.split('[-_]',raw)):continue
                if re.search(r'\s',raw) and prefix.lower() in ['in','on','at','to','of','for','and','or','no','pro','max','air']:continue
                if language=='en' and re.search(r'\s',raw):
                    common={'is','am','are','was','has','had','can','not','the','an','as','be','by','do','if','my','our','we','he','she','it','its','you','see','say','all','get'}
                    if prefix.lower() in common or (not prefix.isupper() and prefix.upper() not in self.data['initialisms']):continue
                suffix=re.sub(r'^'+re.escape(prefix)+r'[._ -]*','',raw) if prefix else ''
                word=self.data['word_acronyms'].get(prefix.upper())
                special=next((v.get(language) for k,v in self.data['specials'].items() if k.casefold()==prefix.casefold()),None)
                if re.fullmatch(r'\d+(?:\.\d+)?',suffix) and (word or special) and not (language=='zh' and prefix.isupper() and 2<=len(prefix)<=5):
                    item.update(mode='expand',value=(word or special)+' '+suffix,rule='word_acronym_with_number')
                else:item.update(mode='spell',value=letters,rule='alphanumeric_identifier')
            elif kind=='word' and ("'" in raw or '’' in raw):
                item.update(mode='word',value=raw.replace('’',"'"),rule='contraction_or_possessive')
            else:
                canonical=letters.upper()
                explicit=kind in ['dotted','spaced']
                initialism=canonical in self.data['initialisms'] or raw in self.data['case_sensitive_initialisms']
                acronym=canonical in self.data['word_acronyms']
                candidate=explicit or initialism or acronym or (raw.isupper() and len(raw)>1)
                if candidate and self.mode!='auto':
                    item.update(mode=self.mode,value=canonical if self.mode=='spell' else self.data['word_acronyms'].get(canonical,letters.lower()),rule='global_mode')
                elif explicit or initialism:item.update(mode='spell',value=canonical,rule='explicit_letters' if explicit else 'initialism_dictionary')
                elif language=='zh' and raw.isupper() and 2<=len(raw)<=5 and raw not in {'YES','NO','OK','HI','OH','HEY','STOP','WATCH','PRO'}:
                    item.update(mode='spell',value=canonical,rule='accepted_Chinese_initialism')
                elif acronym:item.update(mode='word',value=self.data['word_acronyms'][canonical],rule='word_acronym_dictionary')
                elif len(raw)==1 and raw.isupper():
                    follows_word=bool(re.match(r'\s+[A-Za-z]',text[match.end():]))
                    label=bool(re.search(r'\b(?:press|key|drive|column|row|option|grade)\s*$',text[:match.start()],re.I))
                    if raw in ['A','I'] and follows_word and not label:continue
                    item.update(mode='spell',value=raw,rule='single_letter')
                elif raw.isupper() and len(raw)>1:
                    lexical_kind=lexical(raw) if lexical else None
                    if canonical in self.data['lexical_words'] or lexical_kind=='word':item.update(mode='word',value=raw.lower(),rule='lexical_word')
                    else:item.update(mode='spell',value=canonical,rule='lexical_initialism' if lexical_kind=='spell' else 'unknown_uppercase_initialism')
                else:continue
            events.append(item)
        return text,events

    def atomic_spans(self,text,language):
        # Map normalized tokens back to original-character spans for chunking.
        normalized=[];owners=[]
        for i,char in enumerate(text):
            value=clean(char);normalized.append(value);owners.extend([i]*len(value))
        value=''.join(normalized);spans=[]
        for match in self._patterns(language).finditer(value):
            end=match.end()
            if match.lastgroup in ['literal','dotted'] and match[0].endswith('.'):
                following=re.match(r"[ \t]*[A-Za-z0-9][A-Za-z0-9'-]*",value[end:])
                if following and (match.lastgroup=='literal' or following[0].lstrip()[0].islower()):end+=following.end()
            if match.start()<end:spans.append((owners[match.start()],owners[end-1]+1))
        return spans

def spell_english(value):
    return ' '.join(LETTER_NAMES[c.upper()].replace('_',' ') if c.isalpha() else EN_DIGITS[int(c)] for c in value)

class LexicalLookup:
    """Reuse the already bundled CMU lexicon; no extra dictionary or model."""
    def __init__(self,front):self.front=front;self.dictionary=None
    def __call__(self,word):
        if self.dictionary is None:
            self.front._session('en');self.dictionary=self.front.resources.documents['english'][0]
        value=self.dictionary.get(word.lower())
        if value is None:return None
        norm=lambda x:tuple(re.sub(r'[012]','',x).split())
        spelling=[]
        for char in word:
            for name in LETTER_NAMES[char].replace('_',' ').split():
                part=self.dictionary.get(name)
                if part is None:return 'word'
                spelling.extend(norm(part))
        return 'spell' if norm(value)==tuple(spelling) else 'word'
