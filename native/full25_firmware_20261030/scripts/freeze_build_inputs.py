"""Archive actual compiler dependencies and incbin payloads by SHA256."""
import argparse
import re
import shutil
import subprocess
from pathlib import Path
from build_resources import ROOT,sha,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('tags',nargs='+');args=parser.parse_args()
    store=ROOT/'frozen_inputs/sha256';store.mkdir(parents=True,exist_ok=True)
    for tag in args.tags:
        build=ROOT/'build'/tag;document=__import__('json').loads((build/'BUILD.json').read_text())
        files=set();steps=[]
        for command in document['commands']:
            if '-c' not in command:
                for value in command:
                    p=Path(value)
                    if p.is_file()and p.suffix in('.a','.ld'):files.add(p)
                continue
            source=Path(command[command.index('-c')+1]);files.add(source)
            dependency_command=command.copy();i=dependency_command.index('-o');del dependency_command[i:i+2]
            dependency_command.remove('-c');dependency_command+=['-M','-MT','dependency']
            result=subprocess.run(dependency_command,capture_output=True,text=True,check=True)
            dependencies=result.stdout.replace('\\\n',' ').split(':',1)[1].split()
            for name in dependencies:files.add(Path(name))
            if source.suffix in('.S','.s'):
                for name in re.findall(r'\.incbin\s+"([^"]+)"',source.read_text()):files.add(Path(name))
            steps.append(dict(command=command,source=str(source),dependencies=dependencies))
        rows=[]
        for p in sorted(files):
            digest=sha(p);target=store/digest
            if target.exists():assert sha(target)==digest
            else:shutil.copyfile(p,target)
            rows.append(dict(path=str(p),sha256=digest,bytes=p.stat().st_size,
                frozen_file=str(target.relative_to(ROOT))))
        receipt=dict(elf_sha256=sha(build/'firmware.elf'),files=rows,compile_steps=steps,
            exact_compiler_dependencies_and_incbin_inputs=True,
            note='Content-addressed source/headers/assets archive; no production pointer or remote repository changes.')
        write(build/'FROZEN_INPUTS.json',receipt)
        print(tag,len(rows),'files',sum(p['bytes']for p in rows),'bytes',flush=True)

if __name__=='__main__':main()
