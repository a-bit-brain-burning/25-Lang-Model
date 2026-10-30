"""Reuse the accepted bilingual normalization/classification, not its weights."""
from pathlib import Path
import ast,re,unicodedata,hashlib
LEGACY=Path(__file__).resolve().parent/'legacy_frontend'
def selected(path,names,namespace):
    tree=ast.parse(path.read_text());nodes=[]
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:nodes.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in node.targets):nodes.append(node)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
ns=dict(re=re,unicodedata=unicodedata)
selected(LEGACY/'text_prepare.py',{'ZH_DIGITS','EN_DIGITS','NUMBER','base','digits','normalize_zh','is_han'},ns)
selected(LEGACY/'frontend.py',{'LETTER_PHONES','LEXICAL_UPPER','COMMON_INITIALISMS','letter_spans'},ns)
letter_spans=ns['letter_spans'];LETTER_PHONES=ns['LETTER_PHONES']
def normalize_chinese(text):
    from frontend_repair import cardinal
    ns['zh_number']=lambda value:cardinal(value.replace(',',''))
    text=ns['base'](text);protected=[]
    def keep(value,kind='identifier'):
        marker=chr(0xe000+len(protected));protected.append(dict(marker=marker,kind=kind,value=re.sub(r'\s+','',value)));return marker
    # Protect each whole identifier before numeric expansion. Never switch its
    # letters to the English acoustic model; retain leading zeroes.
    text=re.sub(r'(?<![A-Za-z0-9])(?:(?:[A-Z]{1,4}|[a-z]{1,3})\s*\d[A-Za-z0-9]*|\d+[A-Za-z]{1,4})(?![A-Za-z0-9])',lambda m:keep(m[0]),text)
    text=re.sub(r'(?<!\d)\d{2,4}[ -]\d{5,9}(?!\d)',lambda m:keep(m[0],'phone'),text)
    text=re.sub(r'((?:电话(?:是)?|热线|拨打))([0-9]{3,})',lambda m:m[1]+keep(m[2],'digits'),text)
    text=re.sub(r'(?<!\d)(\d{4,6})(?=热线|转接|电话)',lambda m:keep(m[1],'digits'),text)
    # The old normalizer handles times, years, ranges, units and punctuation.
    text=ns['normalize_zh'](text)
    for start,end,word in reversed(letter_spans(text)):
        text=text[:start]+keep(word)+text[end:]
    # Existing trained inventory has en, not the syllabic n used for bare 嗯.
    text=text.replace('嗯','嗯[en5]')
    return text,protected
def identity():
    return {str(LEGACY/name):hashlib.sha256((LEGACY/name).read_bytes()).hexdigest() for name in ['frontend.py','text_prepare.py','audio_join.py']}
