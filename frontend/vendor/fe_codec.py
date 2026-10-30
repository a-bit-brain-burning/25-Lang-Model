import ctypes as C, hashlib, json, lzma, struct, tempfile, collections, unicodedata, types, zlib
from pathlib import Path
import espeakng_loader
HERE=Path(__file__).resolve().parent/'frontend'
WINDOW=262144
FILTERS=[{'id':lzma.FILTER_LZMA2,'dict_size':WINDOW,'preset':9}]
HEADER=struct.Struct('<8s6I');SECTION=struct.Struct('<32s6I');MAGIC=b'FEPACK1\0'
def varint(v):
 out=bytearray()
 while v>=128:out.append((v&127)|128);v>>=7
 out.append(v);return bytes(out)
def get_varint(raw,pos):
 value=0;shift=0
 while True:
  if pos>=len(raw)or shift>63:raise ValueError('Invalid varint')
  b=raw[pos];pos+=1;value|=(b&127)<<shift
  if b<128:return value,pos
  shift+=7
compound=types.SimpleNamespace(varint=varint,get_varint=get_varint)


class Codec:
    def __init__(self,pool):self.pool=pool;self.ids={s:i for i,s in enumerate(pool)}
    def encode(self,value):
        if isinstance(value,dict) and set(value)=={'$bytes'}:
            raw=bytes.fromhex(value['$bytes']);return b'\x08'+varint(len(raw))+raw
        if value is None:return b'\x00'
        if value is False:return b'\x01'
        if value is True:return b'\x02'
        if isinstance(value,int):return b'\x03'+varint(value*2 if value>=0 else -value*2-1)
        if isinstance(value,float):return b'\x07'+struct.pack('<d',value)
        if isinstance(value,str):
            if value in self.ids:return b'\x04'+varint(self.ids[value])
            raw=value.encode();return b'\x09'+varint(len(raw))+raw
        if isinstance(value,list):return b'\x05'+varint(len(value))+b''.join(self.encode(v) for v in value)
        if isinstance(value,dict):return b'\x06'+varint(len(value))+b''.join(self.encode(k)+self.encode(v) for k,v in value.items())
        raise TypeError(type(value))
    def decode(self,raw):
        def read(pos,depth):
            if depth>64 or pos>=len(raw):raise ValueError('Invalid typed value')
            tag=raw[pos];pos+=1
            if tag<3:return (None,False,True)[tag],pos
            if tag==7:
                if pos+8>len(raw):raise ValueError('Truncated float')
                return struct.unpack_from('<d',raw,pos)[0],pos+8
            size,pos=get_varint(raw,pos)
            if tag==3:return (size//2 if size%2==0 else -(size//2)-1),pos
            if tag==4:return self.pool[size],pos
            if tag in (8,9):
                if size>len(raw)-pos:raise ValueError('Truncated string/bytes')
                value=raw[pos:pos+size];return (value.decode() if tag==9 else value),pos+size
            if tag not in (5,6) or size>len(raw):raise ValueError('Invalid collection')
            values=[]
            for _ in range(size*(2 if tag==6 else 1)):
                value,pos=read(pos,depth+1);values.append(value)
            return (dict(zip(values[::2],values[1::2])) if tag==6 else values),pos
        value,position=read(0,0)
        if position!=len(raw):raise ValueError('Trailing value bytes')
        return value


def compressed(raw):return lzma.compress(raw,format=lzma.FORMAT_RAW,filters=FILTERS)


def write(path,sections):
    directory=[];payload=bytearray();offset=HEADER.size+SECTION.size*len(sections)
    for name,raw in sections.items():
        blob=compressed(raw);codec=1
        if len(blob)>=len(raw):blob=raw;codec=0
        assert len(name.encode())<32
        directory.append((name.encode(),codec,offset+len(payload),len(blob),len(raw),zlib.crc32(raw),WINDOW))
        payload.extend(blob)
    data=HEADER.pack(MAGIC,1,len(sections),HEADER.size,offset,offset+len(payload),0)
    data+=b''.join(SECTION.pack(*r) for r in directory)+payload
    Path(path).write_bytes(data)
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'sections':len(sections),
        'max_section_raw_bytes':max(map(len,sections.values()),default=0),
        'parts':{r[0].decode():{'packed':r[3],'raw':r[4]} for r in directory}}


class Reader:
    def __init__(self,path):
        self.path=Path(path);self.data=self.path.read_bytes();magic,v,n,do,pay,size,res=HEADER.unpack_from(self.data)
        if magic!=MAGIC or v!=1 or do!=HEADER.size or pay!=do+n*SECTION.size or size!=len(self.data) or res:raise ValueError('pack header')
        self.sections={}
        for i in range(n):
            name,codec,start,length,raw,crc,window=SECTION.unpack_from(self.data,do+i*SECTION.size);name=name.rstrip(b'\0').decode()
            if not name or name in self.sections or start<pay or start+length>size or codec not in (0,1) or window!=WINDOW:raise ValueError('section')
            self.sections[name]=(codec,start,length,raw,crc)
    def read(self,name):
        codec,start,n,raw,crc=self.sections[name];blob=self.data[start:start+n]
        if codec:blob=lzma.decompress(blob,format=lzma.FORMAT_RAW,filters=FILTERS)
        if len(blob)!=raw or zlib.crc32(blob)!=crc:raise ValueError('section checksum')
        return blob


def parse_ids(raw,n):
    out=[];p=0
    for _ in range(n):v,p=compound.get_varint(raw,p);out.append(v)
    if p!=len(raw):raise ValueError('index extent')
    return out


class NativePack(Reader):
    def __init__(self,path,library=None):
        super().__init__(path);self.lib=C.CDLL(str(library or HERE/'libs/libreader.so'));self.rom=C.create_string_buffer(self.data)
        self.arena=C.create_string_buffer(524288);self.max_arena=0
        self.lib.fep_read.argtypes=[C.c_void_p,C.c_size_t,C.c_char_p,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]
        self.lib.fep_keys.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(C.c_uint32),C.c_size_t,C.c_char_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)]
        self.lib.fep_id.argtypes=[C.c_void_p,C.c_size_t,C.c_uint32,C.POINTER(C.c_uint32)]
        self.lib.fep_pool.argtypes=[C.c_void_p,C.c_size_t,C.c_uint32,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)]
    def read(self,name):
        n=self.sections[name][3]
        if n>2097152:raise ValueError('section exceeds candidate bound')
        out=C.create_string_buffer(max(n,1));used=C.c_size_t();written=C.c_size_t()
        rc=self.lib.fep_read(self.rom,len(self.data),name.encode(),out,n,self.arena,len(self.arena),C.byref(used),C.byref(written))
        if rc:raise ValueError(('native unpack',name,rc))
        if written.value!=n:raise ValueError('native extent')
        self.max_arena=max(self.max_arena,used.value)
        return out.raw[:n]
    def keys(self,raw,alphabet=(),query=None):
        if query is not None and '\0' in query:raise KeyError(query)
        chars=(C.c_uint32*len(alphabet))(*(ord(c) for c in alphabet));written=C.c_size_t()
        cap=4194304 if query is None else 0;out=C.create_string_buffer(max(cap,1))
        rc=self.lib.fep_keys(raw,len(raw),chars,len(alphabet),None if query is None else query.encode(),out,cap,C.byref(written))
        if query is not None:
            if rc==-5:raise KeyError(query)
            if rc<0:raise ValueError(('native query',rc))
            return rc
        if rc:raise ValueError(('native keys',rc))
        return [w.decode() for w in out.raw[:written.value].split(b'\0')[:-1]]
    def identifier(self,raw,index):
        out=C.c_uint32();rc=self.lib.fep_id(raw,len(raw),index,C.byref(out))
        if rc:raise ValueError(('native row ID',rc))
        return out.value
    def pool_value(self,raw,index):
        ptr=C.c_void_p();size=C.c_size_t();rc=self.lib.fep_pool(raw,len(raw),index,C.byref(ptr),C.byref(size))
        if rc:raise ValueError(('native value',rc))
        return C.string_at(ptr,size.value)


def raw_tables(path,library=None):
    p=NativePack(path,library);layout=Codec([]).decode(p.read('metadata'));alphabet=[]
    if layout['mode']==2:
        data=p.read('alphabet');pos=0;cp=-1
        while pos<len(data):v,pos=compound.get_varint(data,pos);cp+=v+1;alphabet.append(chr(cp))
    pools=[p.read('pool/'+str(i)) for i in range(len(layout['columns']))];columns={name:{} for name in layout['columns']}
    for block in range(len(layout['first_keys'])):
        words=p.keys(p.read('keys/'+str(block)),alphabet)
        for i,name in enumerate(columns):
            ids=parse_ids(p.read(f'ids/{i}/{block}'),len(words))
            for word,index in zip(words,ids):
                value=p.pool_value(pools[i],index)
                if value:columns[name][word]=value
    documents={}
    for name,values in columns.items():
        meta=layout['metadata'][name];order=meta['__top_level_order__']
        documents[name]={key:values if key=='entries' else meta[key] for key in order}
    return documents,p


def unpack_metadata(raw):
    result=Codec([]).decode(raw)
    if result.pop('entry_order_codec',None)=='zigzag_rank_deltas_v1':
        encoded=result['entry_order'];position=0;previous=0;order=[]
        while position<len(encoded):
            delta,position=get_varint(encoded,position);previous+=delta//2 if delta%2==0 else -(delta//2)-1;order.append(previous)
        if sorted(order)!=list(range(len(order))):raise ValueError('Invalid original-order permutation')
        result['entry_order']=order
    return result


def decode_record(raw,metadata):
    end=raw.index(255);phones=[metadata['symbols'][i] for i in raw[:end]]
    tid,aux,mask,*extra=Codec(metadata['strings']).decode(raw[end+1:])
    template=metadata['templates'][tid]
    if template is None:entry=phones
    else:
        aux=iter(aux);entry={k:phones if k=='phones' else next(aux) for k in template}
    values={};extra=iter(extra)
    for i,name in enumerate(metadata['overlay']):
        if mask&(1<<i):values[name]=next(extra)
    return entry,values


class View(C.Structure):
    _fields_=[('rom',C.c_void_p)]+[(n,C.c_uint32) for n in ('size','count','index','data')]


class NativePredictor:
    def __init__(self,path,library=None):
        self.lib=C.CDLL(str(library or HERE/'libs/libfrontend4.so'))
        self.lib.spr_open.argtypes=[C.POINTER(View),C.c_void_p,C.c_size_t]
        self.lib.spr_predict.argtypes=[C.POINTER(View),C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)]
        self.lib.spr_apply.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)]
        self.rom=C.create_string_buffer(Path(path).read_bytes());self.view=View()
        if self.lib.spr_open(C.byref(self.view),self.rom,len(self.rom)-1):raise ValueError('native model open')
        self.pred=C.create_string_buffer(1024);self.output=C.create_string_buffer(4096)
    def predict(self,word):
        raw=word.encode() if isinstance(word,str) else word;n=C.c_size_t()
        status=self.lib.spr_predict(C.byref(self.view),raw,len(raw),self.pred,len(self.pred),C.byref(n))
        if status:raise ValueError(('native predict',status,word))
        return self.pred.raw[:n.value]
    def apply(self,prediction,record,capacity=None):
        n=C.c_size_t();status=self.lib.spr_apply(prediction,len(prediction),record,len(record),self.output,
            len(self.output) if capacity is None else capacity,C.byref(n))
        if status:raise ValueError(('native residual',status))
        return self.output.raw[:n.value]


class Values:
    def __init__(self,library=None):
        self.lib=C.CDLL(str(library or HERE/'libs/libfrontend4.so'));self.temp=C.create_string_buffer(4096);self.out=C.create_string_buffer(4096)
        self.lib.en_apply.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)]
        self.lib.phrase_apply.argtypes=[C.POINTER(C.c_uint16),C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)]
        self.lib.primary_id.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.POINTER(C.c_uint16)]
        self.lib.aux_expand.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.POINTER(C.c_size_t)]
    def en(self,prediction,record,capacity=4096):
        n=C.c_size_t();status=self.lib.en_apply(prediction,len(prediction),record,len(record),self.temp,len(self.temp),self.out,capacity,C.byref(n))
        if status:raise ValueError(('en_apply',status))
        return self.out.raw[:n.value]
    def primary(self,record,translation):
        result=C.c_uint16();status=self.lib.primary_id(record,len(record),translation,len(translation),C.byref(result))
        if status:raise ValueError(('primary_id',status))
        return result.value
    def phrase(self,defaults,record,capacity=4096):
        ids=(C.c_uint16*max(1,len(defaults)))(*defaults);n=C.c_size_t()
        status=self.lib.phrase_apply(ids,len(defaults),record,len(record),self.out,capacity,C.byref(n))
        if status:raise ValueError(('phrase_apply',status))
        return self.out.raw[:n.value]
    def aux(self,pool,record,capacity=4096):
        n=C.c_size_t();status=self.lib.aux_expand(pool,len(pool),record,len(record),self.out,capacity,C.byref(n))
        if status:raise ValueError(('aux_expand',status))
        return self.out.raw[:n.value]


ARP={'ɑ':'AA','ɑː':'AA','ɐ':'AH','æ':'AE','ʌ':'AH','ə':'AH','ɔ':'AO','ɔː':'AO','aʊ':'AW','aɪ':'AY',
 'ɛ':'EH','ɜ':'ER','ɜː':'ER','ɚ':'ER','ɝ':'ER','ɜːɹ':'ER','ɚː':'ER','eɪ':'EY','ɪ':'IH','ɪɹ':'IH',
 'i':'IY','iː':'IY','oʊ':'OW','ɔɪ':'OY','ʊ':'UH','u':'UW','uː':'UW','b':'B','tʃ':'CH','d':'D','ð':'DH',
 'f':'F','ɡ':'G','g':'G','h':'HH','dʒ':'JH','k':'K','l':'L','ɫ':'L','m':'M','n':'N','ŋ':'NG','p':'P',
 'r':'R','ɹ':'R','s':'S','ʃ':'SH','t':'T','ɾ':'T','ʔ':'T','θ':'TH','v':'V','w':'W','j':'Y','z':'Z','ʒ':'ZH'}


class Backend:
 def __init__(self,data=None):
  self.lib=C.CDLL(espeakng_loader.get_library_path())
  self.lib.espeak_Initialize.argtypes=[C.c_int,C.c_int,C.c_char_p,C.c_int];self.lib.espeak_Initialize.restype=C.c_int
  self.lib.espeak_SetVoiceByName.argtypes=[C.c_char_p];self.lib.espeak_SetVoiceByName.restype=C.c_int
  self.lib.espeak_TextToPhonemes.argtypes=[C.POINTER(C.c_void_p),C.c_int,C.c_int];self.lib.espeak_TextToPhonemes.restype=C.c_char_p
  assert self.lib.espeak_Initialize(2,0,str(data or espeakng_loader.get_data_path()).encode(),0)>0
 def word(self,word,voice):
  if self.lib.espeak_SetVoiceByName(voice.encode()):raise ValueError('voice unavailable')
  buf=C.create_string_buffer(word.encode());ptr=C.c_void_p(C.addressof(buf));parts=[]
  for _ in range(1000):
   if not ptr.value:break
   old=ptr.value;parts.append((self.lib.espeak_TextToPhonemes(C.byref(ptr),1,2|(ord('_')<<8)) or b'').decode())
   if old==ptr.value:raise ValueError('cursor')
  else:raise ValueError('clause limit')
  return ' '.join(parts).strip()


def english_tokens(raw,ids):
 out=[]
 for piece in raw.replace(' ','_').split('_'):
  if not piece:continue
  stress=2 if 'ˈ' in piece else 3 if 'ˌ' in piece else 1
  phone=piece.translate(str.maketrans('','','ˈˌ'))
  name=ARP.get(phone)
  if name is None:out.append(254);continue
  s=stress if name[0] in 'AEIOU' else 0
  out.append(ids[name]*4+s)
 return bytes(out)


def decode_english(document):
    base={};stress={};symbols=document['symbols']
    for word,raw in document['entries'].items():
        if raw[0]==0:
            n=raw[1];tokens=[symbols[i] for i in raw[2:2+n]];packed=raw[2+n:]
            assert len(packed)==(n+3)//4
            codes=[(packed[i//4]>>(2*(i%4)))&3 for i in range(n)]
            base[word]=' '.join(tokens)
            stress[word]=' '.join(token+('' if code==0 else str(code-1)) for token,code in zip(tokens,codes))
        else:
            assert raw[0]==1
            a,b=Codec([]).decode(raw[1:])
            if a is not None:base[word]=a
            if b is not None:stress[word]=b
    return base,stress


def unpack_cells(raw,strings):
    cells=[];position=0
    while position<len(raw):
        count,position=get_varint(raw,position);cell=[]
        for _ in range(count):
            identifier,position=get_varint(raw,position);cell.append(strings[identifier])
        cells.append(cell)
    return cells


def restore_chinese(name,document):
    strings=document['strings']
    if name=='phrases':return {word:unpack_cells(value,strings) for word,value in document['entries'].items()}
    if name=='characters':return {ord(word):','.join(unpack_cells(value,strings)[0]) for word,value in document['entries'].items()}
    freq={};tags={}
    for word,value in document['entries'].items():
        frequency,position=get_varint(value,0);tag,position=get_varint(value,position);assert position==len(value)
        freq[word]=frequency
        if tag:tags[word]=strings[tag-1]
    if document.get('prefixes_implicit'):
        for word in list(freq):
            for stop in range(1,len(word)):freq.setdefault(word[:stop],0)
    return freq,tags,document['total']


def restore_bgsl(path,library=None):
    p=NativePack(path,library)
    layout=pack.Codec([]).decode(p.read('metadata'))
    alphabet=[]
    if layout['mode']==2:
        raw=p.read('alphabet');pos=0;cp=-1
        while pos<len(raw):v,pos=pack.compound.get_varint(raw,pos);cp+=v+1;alphabet.append(chr(cp))
    pools=[p.read('pool/'+str(i)) for i in range(len(layout['columns']))]
    result={name:{} for name in layout['columns']}
    for block in range(len(layout['first_keys'])):
        words=p.keys(p.read('keys/'+str(block)),alphabet)
        for i,name in enumerate(layout['columns']):
            ids=pack.parse_ids(p.read(f'ids/{i}/{block}'),len(words))
            for word,index in zip(words,ids):
                value=p.pool_value(pools[i],index)
                if value:result[name][word]=value
    # Reconstruct source insertion order, including the top-level order.
    strings=pack.Codec([]).decode(result['__metadata__']['device_strings'])
    codec=pack.Codec(json.loads(strings))
    ordered={}
    for name,table in result.items():
        keys=list(table)
        ranks=pack.parse_ids(p.read('order/'+str(layout['columns'].index(name))),len(keys))
        assert sorted(ranks)==list(range(len(keys)))
        ordered[name]={keys[i]:codec.decode(table[keys[i]]) for i in ranks}
    meta=dict(ordered['__metadata__']);top=meta.pop('__top_level_order__')
    document={key:ordered[key] if key in ordered and key!='__metadata__' else meta[key] for key in top}
    device_layout=document.pop('device_layout');document.pop('device_strings')
    assert 'rules' not in document
    document={k:document[k] for k in device_layout['projected_original_top_order']}
    return document,ordered,p


pack=types.SimpleNamespace(Codec=Codec,parse_ids=parse_ids,compound=compound)
