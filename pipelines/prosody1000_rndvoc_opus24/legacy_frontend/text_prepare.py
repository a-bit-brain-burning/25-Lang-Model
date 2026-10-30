"""Host text preparation for spreadsheet listening, with an auditable transcript.

This does not change the compact G2P libraries or any training/benchmark text.
Numbers, units and mixed-language passages are expanded with general rules.
"""
import re
import unicodedata
from decimal import Decimal
import cn2an
from num2words import num2words

ZH_DIGITS='零一二三四五六七八九'
EN_DIGITS='zero one two three four five six seven eight nine'.split()
NUMBER=r'\d[\d,]*(?:\.\d+)?'
LETTER_NAMES=dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',[
    'ay','bee','see','dee','ee','ef','gee','aitch','eye','jay','kay','el','em',
    'en','oh','pee','cue','ar','ess','tee','you','vee','double you','ex','why','zee']))


def zh_number(value):
    return cn2an.an2cn(value.replace(',',''))


def en_number(value,ordinal=False):
    value=value.replace(',','')
    return num2words(int(value),to='ordinal') if ordinal else num2words(value)


def digits(value,language):
    table=ZH_DIGITS if language=='zh' else EN_DIGITS
    return ('' if language=='zh' else ' ').join(table[int(c)] for c in value if c.isdigit())


def base(text):
    text=text.replace('º','°')
    text=unicodedata.normalize('NFKC',text).replace('\u00a0',' ')
    text=''.join(c for c in text if unicodedata.category(c)!='Cf')
    return text.replace('–','-').replace('−','-').replace('’',"'").replace('‘',"'")


def normalize_zh(text):
    text=base(text)
    quoted=False;converted=[]
    for char in text:
        if char=='“':quoted=True
        elif char=='”':quoted=False
        elif char=='"':char='”' if quoted else '“';quoted=not quoted
        converted.append(char)
    text=''.join(converted).replace('~','。')
    # Long telephone runs, including area codes, are digit sequences.
    text=re.sub(r'(?<!\d)(?:\+?\d[\d -]{5,}\d)(?!\d)',
                lambda m:digits(m[0],'zh') if len(re.sub(r'\D','',m[0]))>=7 else m[0],text)
    text=re.sub(r'(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?',
                lambda m:zh_number(m[1])+'点'+(zh_number(m[2])+'分' if int(m[2]) else '整')+
                (zh_number(m[3])+'秒' if m[3] else ''),text)
    text=re.sub(r'(?<!\d)(\d{4})(?=年)',lambda m:digits(m[1],'zh'),text)
    text=re.sub(r'(?i)(?<=航班)(\d+)',lambda m:digits(m[1],'zh'),text)
    text=re.sub(r'([A-Za-z]{1,4})(\d{2,})',lambda m:m[1]+' '+digits(m[2],'zh'),text)
    text=re.sub(r'(\d{4,6})(?=热线|转接|电话)',lambda m:digits(m[1],'zh'),text)
    text=re.sub(r'('+NUMBER+r')\s*%\s*-\s*('+NUMBER+r')\s*%',
                lambda m:'百分之'+zh_number(m[1])+'到百分之'+zh_number(m[2]),text)
    text=re.sub(r'('+NUMBER+r')\s*-\s*('+NUMBER+r')\s*%',
                lambda m:'百分之'+zh_number(m[1])+'到百分之'+zh_number(m[2]),text)
    text=re.sub(r'('+NUMBER+r')\s*%',lambda m:'百分之'+zh_number(m[1]),text)
    text=re.sub(r'('+NUMBER+r')\s*°\s*([CF])',
                lambda m:zh_number(m[1])+('摄氏度' if m[2]=='C' else '华氏度'),text)
    currencies={'$':'美元','£':'英镑','€':'欧元','¥':'元','￥':'元'}
    text=re.sub(r'([$£€¥￥])\s*('+NUMBER+r')',lambda m:zh_number(m[2])+currencies[m[1]],text)
    text=re.sub(r'('+NUMBER+r')\s*/\s*('+NUMBER+r')',lambda m:zh_number(m[2])+'分之'+zh_number(m[1]),text)
    text=re.sub(r'('+NUMBER+r')\+(?!\d)',lambda m:zh_number(m[1])+'多',text)
    text=re.sub(r'('+NUMBER+r')\s*-\s*('+NUMBER+r')',lambda m:zh_number(m[1])+'到'+zh_number(m[2]),text)
    text=re.sub(NUMBER,lambda m:zh_number(m[0]),text)
    text=text.replace('/','每').replace('%','百分比').replace('=','等于').replace('+','加')
    return re.sub(r'\s+',' ',text).strip()


def money(match):
    symbol=match[1];value=Decimal(match[2].replace(',',''))
    units={'£':('pound','pence'),'$':('dollar','cents'),'€':('euro','cents')}
    unit,cents=units[symbol];whole=int(value);fraction=int((value-whole)*100)
    result=en_number(str(whole))+' '+unit+('s' if whole!=1 else '')
    if fraction:result+=' and '+en_number(str(fraction))+' '+('penny' if symbol=='£' and fraction==1 else 'cent' if fraction==1 else cents)
    return result


def normalize_en(text):
    text=base(text)
    text=re.sub(r'([$£€])\s*('+NUMBER+r')',money,text)
    text=re.sub(r'('+NUMBER+r')\s*°\s*([CF])',
                lambda m:en_number(m[1])+' degrees '+('Celsius' if m[2]=='C' else 'Fahrenheit'),text)
    text=re.sub(r'(?<!\w)(?:\+?\d[\d -]{5,}\d)(?!\w)',
                lambda m:digits(m[0],'en') if len(re.sub(r'\D','',m[0]))>=7 else m[0],text)
    def clock(m):
        h,minute=int(m[1]),int(m[2])
        if h==minute==0:return 'midnight'
        return en_number(str(h))+(' hundred' if minute==0 else (' oh ' if minute<10 else ' ')+en_number(str(minute)))
    text=re.sub(r'(?<!\d)(\d{1,2}):(\d{2})(?!\d)',clock,text)
    month=r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)'
    text=re.sub(r'\b(\d{1,2})\s*-\s*(\d{1,2})\s+'+month+r'\b',
                lambda m:en_number(m[1],True)+' to '+en_number(m[2],True)+' '+m[3],text,flags=re.I)
    text=re.sub(r'\b'+month+r'\.?\s+(\d{1,2})(?:,\s*(\d{4}))?\b',
                lambda m:m[1]+' '+en_number(m[2],True)+(', '+num2words(int(m[3]),to='year') if m[3] else ''),text,flags=re.I)
    text=re.sub(r'\b(account|number|flight)\s+(\d{4,})\b',lambda m:m[1]+' '+digits(m[2],'en'),text,flags=re.I)
    text=re.sub(r'('+NUMBER+r')\s*-\s*('+NUMBER+r')\s*%',lambda m:en_number(m[1])+' to '+en_number(m[2])+' percent',text)
    text=re.sub(r'('+NUMBER+r')\s*%',lambda m:en_number(m[1])+' percent',text)
    units={'kV':'kilovolt','V':'volt','km':'kilometer','min':'minute','kg':'kilogram','cm':'centimeter','mm':'millimeter','GB':'gigabyte','MB':'megabyte'}
    text=re.sub(r'('+NUMBER+r')\s*('+ '|'.join(units)+r')\b',
                lambda m:en_number(m[1])+' '+units[m[2]]+('s' if Decimal(m[1].replace(',',''))!=1 else ''),text)
    text=re.sub(r'(\d+)(st|nd|rd|th)\b',lambda m:en_number(m[1],True),text,flags=re.I)
    text=re.sub(r'('+NUMBER+r')\s*-\s*('+NUMBER+r')',lambda m:en_number(m[1])+' to '+en_number(m[2]),text)
    text=re.sub(r'\b(in|since|during)\s+((?:19|20)\d{2})\b',lambda m:m[1]+' '+num2words(int(m[2]),to='year'),text,flags=re.I)
    text=re.sub(NUMBER,lambda m:en_number(m[0]),text)
    text=re.sub(r'(?<=[A-Za-z])\s*>\s*(?=[A-Za-z])',' then ',text)
    text=text.replace('>',' greater than ').replace('<',' less than ')
    text=text.replace('=',' equals ').replace('&',' and ').replace('+',' plus ').replace('/',' slash ')
    # Preserve lexical letters, converting only diacritics unsupported by the native ASCII adapter.
    text=''.join(c for c in unicodedata.normalize('NFKD',text) if not unicodedata.combining(c))
    return re.sub(r'\s+',' ',text).strip()


def is_han(c):return '\u3400'<=c<='\u9fff' or '\uf900'<=c<='\ufaff'


def language_runs(text,default):
    runs=[];current='';language=default
    for char in text:
        kind='zh' if is_han(char) else 'en' if char.isascii() and char.isalpha() else None
        if kind is not None and kind!=language:
            if any(is_han(c) or c.isalpha() for c in current):runs.append((language,current));current=''
            language=kind
        current+=char
    if current:runs.append((language,current))
    assert ''.join(s for _,s in runs)==text
    return runs


def split_long(text,language):
    limit=62 if language=='zh' else 190
    pieces=[];remaining=text
    while len(remaining)>limit:
        candidates=[i+1 for i,c in enumerate(remaining[:limit]) if c in '。！？.!?；;，,:：\n']
        if language=='en':candidates += [i+1 for i,c in enumerate(remaining[:limit]) if c==' ']
        at=max(candidates) if candidates else limit
        if at<limit//3 and len(remaining)>2*limit:at=limit
        pieces.append(remaining[:at]);remaining=remaining[at:]
    if remaining:pieces.append(remaining)
    assert ''.join(pieces)==text
    return pieces


def segments(text,language):
    normalized=normalize_zh(text) if language=='zh' else normalize_en(text)
    assert not re.search(r'\d',normalized),normalized
    result=[]
    for lang,run in language_runs(normalized,language):
        if lang=='en':
            run=re.sub(r'(?<=[a-z])(?=[A-Z])',' ',run)
            # Initialisms use letter names. Common short exclamations remain words.
            run=re.sub(r'\b[A-Z]{2,4}\b',lambda m:m[0] if m[0] in ('YES','NO','OK','HI','OH','HEY','STOP')
                       else ' '.join(LETTER_NAMES[c] for c in m[0]),run)
            if language=='zh' and re.fullmatch(r'\s*[A-Z]\s*',run):run=LETTER_NAMES[run.strip()]
        for chunk in split_long(run,lang):
            if any(is_han(c) or c.isalpha() for c in chunk):result.append(dict(language=lang,text=chunk.strip()))
    assert result
    return normalized,result
