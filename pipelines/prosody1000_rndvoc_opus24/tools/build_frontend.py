"""Rebuild the four bundled Linux frontend libraries from included source.

Requires a C compiler, zlib and liblzma development headers. No network used.
Updates only library files and hashes in the frontend runtime manifest.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]/'frontend_bundle'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cc',default='cc')
    p.add_argument('--include-dir',action='append',default=[])
    p.add_argument('--library-dir',action='append',default=[])
    args=p.parse_args()
    source=ROOT/'frontend/sources'
    libs=ROOT/'frontend/vendor/frontend/libs'
    manifest=ROOT/'frontend/vendor/frontend/MANIFEST.json'
    original=json.loads(manifest.read_text())
    commands=[]
    def run(command):
        commands.append(list(map(str,command)))
        subprocess.run(commands[-1],check=True)
    with tempfile.TemporaryDirectory(prefix='new25-frontend-build-')as tmp:
        tmp=Path(tmp)
        espeak=source/'espeak-ng'
        includes=[espeak,espeak/'src/libespeak-ng',espeak/'src/include',espeak/'src/include/compat',espeak/'src/ucd-tools/src/include']
        flags=['-Os','-fPIC','-fvisibility=hidden','-fwrapv','-ffunction-sections','-fdata-sections',
               '-DLIBESPEAK_NG_EXPORT=1','-DPATH_ESPEAK_DATA="/espeak-ng-data"']
        objects=[]
        for i,filename in enumerate(json.loads((source/'source_list.json').read_text())):
            obj=tmp/(str(i)+'.o')
            run([args.cc,*flags,*[v for d in includes for v in ['-I',d]],'-c',source/filename,'-o',obj])
            objects.append(obj)
        run([args.cc,'-shared','-Wl,--gc-sections','-Wl,--version-script='+str(source/'exports.map'),*objects,'-lm','-o',tmp/'libespeak_phonemes.so'])
        readers=source/'readers'
        for name,files,link in [
            ('libreader.so',['reader.c'],['-llzma','-lz']),
            ('libfrontend4.so',['table_reader.c','predict_reader.c','lookup_reader.c','value_decoders.c'],['-lz']),
            ('libopen_predict.so',['open_predict.c','predict_reader.c'],[])]:
            run([args.cc,'-Os','-shared','-fPIC','-I',readers,
                 *[v for d in args.include_dir for v in ['-I',d]],
                 *[v for d in args.library_dir for v in ['-L',d]],
                 *[readers/f for f in files],*link,'-o',tmp/name])
        for name in original['libraries']:
            data=(tmp/name).read_bytes()
            (libs/name).write_bytes(data)
            original['libraries'][name]=hashlib.sha256(data).hexdigest()
        original['host_native_library_bytes']=sum((libs/name).stat().st_size for name in original['libraries'])
    manifest.write_text(json.dumps(original,ensure_ascii=False,indent=2)+'\n')
    print('Rebuilt all four libraries; run verify.py to validate this platform.')


if __name__=='__main__':main()
