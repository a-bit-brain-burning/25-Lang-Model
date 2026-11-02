"""Exhaustive key/ID index differential checks against original resource decoding."""
import ctypes as C
import json
import struct
import subprocess
import sys
from build_resources import ROOT,write,sha
from build_espeak import VENDOR

class Index(C.Structure):
    _fields_=[('raw',C.c_void_p),('bytes',C.c_size_t),('alphabet',C.c_void_p),('alphabet_count',C.c_size_t),
        ('data',C.c_void_p),('capacity',C.c_size_t),('groups',C.c_uint),('used',C.c_size_t)]

def main():
    sys.path.insert(0,str(VENDOR));import fe_codec as fe
    out=ROOT/'build/indexed_keys_host_v1';out.mkdir(exist_ok=False)
    source=ROOT/'native_spanish_warm/dictionary_runtime/fep_query_primitives.c';library=out/'libqueries.so'
    command=['gcc','-O2','-shared','-fPIC','-I'+str(source.parent),str(source),'-o',str(library)]
    subprocess.run(command,check=True);lib=C.CDLL(str(library))
    lib.fep_keys_index_build.argtypes=[C.POINTER(Index),C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
    lib.fep_keys_index_query.argtypes=[C.POINTER(Index),C.c_char_p]
    lib.fep_ids_index_build.argtypes=[C.c_void_p,C.c_size_t,C.c_uint,C.c_void_p,C.c_size_t]
    lib.fep_id_index_query.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_uint,C.c_uint,C.POINTER(C.c_uint32)]
    reports=[];total=0
    for locale in 'en-CMU es-ES fr-FR it-IT lt-LT pt-PT ro-RO'.split():
        pack=fe.NativePack(ROOT/'resources/packages'/(locale+'.fep'),VENDOR/'frontend/libs/libreader.so')
        metadata=fe.Codec([]).decode(pack.read('metadata'));alphabet=[]
        if metadata['mode']==2:
            raw=pack.read('alphabet');at=0;previous=-1
            while at<len(raw):value,at=fe.get_varint(raw,at);previous+=value+1;alphabet.append(chr(previous))
        chars=(C.c_uint32*max(1,len(alphabet)))(*(ord(x)for x in alphabet))
        for block in range(len(metadata['first_keys'])):
            raw=pack.read('keys/'+str(block));words=pack.keys(raw,alphabet);owner=C.create_string_buffer(raw)
            memory=C.create_string_buffer(max(262144,len(raw)));index=Index()
            assert lib.fep_keys_index_build(C.byref(index),owner,len(raw),chars,len(alphabet),memory,len(memory))==0
            for row,word in enumerate(words):assert lib.fep_keys_index_query(C.byref(index),word.encode())==row,(locale,block,row,word)
            known=set(words)
            for word in ['', 'A', 'zzzzzzzzzzzzzzzz', '\U0001f600']+[w+'zzzx'for w in words[::max(1,len(words)//64)]]:
                result=lib.fep_keys_index_query(C.byref(index),word.encode())
                if word in known:assert result==words.index(word)
                else:assert result==-5,(locale,block,word,result)
            ids_raw=pack.read('ids/0/'+str(block));ids=fe.parse_ids(ids_raw,len(words));id_owner=C.create_string_buffer(ids_raw)
            directory=(C.c_uint32*((len(words)+63)//64))()
            assert lib.fep_ids_index_build(id_owner,len(ids_raw),len(words),directory,len(directory))==0
            for row,expected in enumerate(ids):
                actual=C.c_uint32();assert lib.fep_id_index_query(id_owner,len(ids_raw),directory,len(words),row,C.byref(actual))==0 and actual.value==expected
            reports.append(dict(locale=locale,block=block,words=len(words),index_bytes_used=index.used,
                key_encoding_mode=struct.unpack_from('<I',raw,8)[0],id_index_bytes=C.sizeof(directory)))
            total+=len(words)
    write(out/'RESULT.json',dict(passed=True,words_and_ids_exact=total,blocks=reports,command=command,
        source_sha256=sha(source),scope='Exhaustive native C index vs original package decoder, host differential; separate M55 integration required'))
    print(json.dumps(dict(passed=True,words=total,blocks=len(reports),spanish=[r for r in reports if r['locale']=='es-ES']),indent=2))

if __name__=='__main__':main()
