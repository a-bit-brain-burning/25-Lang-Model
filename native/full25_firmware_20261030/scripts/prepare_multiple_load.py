"""Use architectural multi-register loads for the unchanged packed dot products."""
import shutil
from build_resources import ROOT,write,sha

def replace_kernel(source,pair=False,bwe=False):
    if pair:
        old=''' vldrb.8 q0,[r1],#16
 vldrb.8 q1,[r3],#16
 vldrb.8 q2,[r5],#16
 vmladava.s8 r0,q0,q2
 vmladava.s8 r8,q1,q2
 vldrb.8 q2,[r5],#16
 vmladava.s8 r2,q0,q2
 vmladava.s8 r10,q1,q2
 vldrb.8 q2,[r5],#16
 vmladava.s8 r4,q0,q2
 vmladava.s8 r12,q1,q2
 vldrb.8 q2,[r5],#16
 vmladava.s8 r6,q0,q2
 vmladava.s8 lr,q1,q2'''
        new=''' vldrb.8 q0,[r1],#16
 vldrb.8 q1,[r3],#16
 vldmia r5!,{s8-s23}
 vmladava.s8 r0,q0,q2
 vmladava.s8 r8,q1,q2
 vmladava.s8 r2,q0,q3
 vmladava.s8 r10,q1,q3
 vmladava.s8 r4,q0,q4
 vmladava.s8 r12,q1,q4
 vmladava.s8 r6,q0,q5
 vmladava.s8 lr,q1,q5'''
        assert source.count(old)==2;source=source.replace(old,new)
        source=source.replace('vpush {d8-d11}','vpush {d8-d15}').replace('vpop {d8-d11}','vpop {d8-d15}')
        source=source.replace('vdup.32 q4,r0','vdup.32 q6,r0').replace('vdup.32 q5,r0','vdup.32 q7,r0')
        source=source.replace('vmul.f32 q0,q0,q4','vmul.f32 q0,q0,q6').replace('vmul.f32 q1,q1,q5','vmul.f32 q1,q1,q7')
    else:
        old=''' vldrb.8 q0,[r1],#16
 vldrb.8 q1,[r3],#16
 vmladava.s8 r0,q0,q1
 vldrb.8 q1,[r3],#16
 vmladava.s8 r2,q0,q1
 vldrb.8 q1,[r3],#16
 vmladava.s8 r4,q0,q1
 vldrb.8 q1,[r3],#16
 vmladava.s8 r6,q0,q1
 vldrb.8 q1,[r5],#16
 vmladava.s8 r8,q0,q1
 vldrb.8 q1,[r5],#16
 vmladava.s8 r10,q0,q1
 vldrb.8 q1,[r5],#16
 vmladava.s8 r12,q0,q1
 vldrb.8 q1,[r5],#16
 vmladava.s8 lr,q0,q1'''
        new=''' vldrb.8 q0,[r1],#16
 vldmia r3!,{s4-s19}
 vmladava.s8 r0,q0,q1
 vmladava.s8 r2,q0,q2
 vmladava.s8 r4,q0,q3
 vmladava.s8 r6,q0,q4
 vldmia r5!,{s4-s19}
 vmladava.s8 r8,q0,q1
 vmladava.s8 r10,q0,q2
 vmladava.s8 r12,q0,q3
 vmladava.s8 lr,q0,q4'''
        assert source.count(old)==2;source=source.replace(old,new)
        if bwe:
            source=source.replace(' push {r4-r11,lr}',' push {r4-r11,lr}\n vpush {d8-d9}')
            source=source.replace(' pop {r4-r11,pc}',' vpop {d8-d9}\n pop {r4-r11,pc}')
        else:
            source=source.replace('vdup.32 q4,r0','vdup.32 q5,r0')
            source=source.replace('vmul.f32 q0,q0,q4','vmul.f32 q0,q0,q5').replace('vmul.f32 q1,q1,q4','vmul.f32 q1,q1,q5')
    return source

def main():
    source=ROOT/'native_spanish_lean';dest=ROOT/'native_multiple_load';dest.mkdir(exist_ok=False)
    for name in ('src','assets','frontend_patched','dictionary_runtime'):shutil.copytree(source/name,dest/name)
    for name in ('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):
        (dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in (dest/'assets').glob('*.S'):p.write_text(p.read_text().replace(str(source),str(dest)))
    rows=[]
    for name,pair,bwe in [('packed_matrix8.S',False,False),('packed_matrix_pair.S',True,False),('bwe_packed_matrix8.S',False,True)]:
        p=dest/'src'/name;before=sha(p);p.write_text(replace_kernel(p.read_text(),pair,bwe))
        rows.append(dict(source=name,before_sha256=before,after_sha256=sha(p)))
    write(ROOT/'MULTIPLE_LOAD_IDENTITY.json',dict(kernels=rows,
        arithmetic_unchanged=True,weight_layout_unchanged=True,
        policy='VLDM moves identical packed bytes into the shared FP/MVE register file; all VMLADAVA sums and FP32 epilogues retained. Logical instruction acceptance is measured separately from physical cycles.'))

if __name__=='__main__':main()
