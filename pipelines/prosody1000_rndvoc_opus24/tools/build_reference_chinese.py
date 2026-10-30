"""Rebuild the reference Chinese G2P from included source and lexicon; offline."""
from pathlib import Path
import subprocess,argparse,tempfile,json,hashlib
ROOT=Path(__file__).resolve().parents[1]/'reference_chinese'
def build(cc='cc'):
 source=ROOT/'source'
 with tempfile.TemporaryDirectory(prefix='reference-zh-build-') as tmp:
  assets=Path(tmp)/'assets.S'
  assets.write_text('.section .rodata\n'+''.join('.balign 4\n.global '+sym+'\n'+sym+':\n.incbin "'+str(source/'assets'/name)+'"\n' for sym,name in [('zh_words_rom','zh_words.bin'),('zh_chars_rom','zh_chars.bin')])+'.section .note.GNU-stack,"",@progbits\n')
  subprocess.run([cc,'-O2','-fPIC','-shared','-std=c99','-I',str(source),*[str(source/x) for x in ['bridge.c','zh_tables.c','utf8_only.c','frontend_crc.c','inflate_wrapper.c']],str(assets),'-lz','-lm','-o',str(ROOT/'libreference_zh.so')],check=True)
 (ROOT/'BUILD.json').write_text(json.dumps(dict(library_sha256=hashlib.sha256((ROOT/'libreference_zh.so').read_bytes()).hexdigest(),source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*'))if p.is_file()},build_requires=['C compiler','zlib development headers']),indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cc',default='cc');a=p.parse_args();build(a.cc)
