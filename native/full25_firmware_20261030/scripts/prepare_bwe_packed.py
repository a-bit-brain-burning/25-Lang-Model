"""Reorder exact BWE I8 codes for an eight-output MVE dot-product kernel."""
from pathlib import Path
import hashlib
import re
import shutil
from build_resources import ROOT,sha,write

def main():
    source=ROOT/'native_lzma';dest=ROOT/'native_bwe_packed';dest.mkdir(exist_ok=False)
    for name in ('src','assets','frontend_patched','bwe_fast'):
        shutil.copytree(source/name,dest/name)
    for name in ('bin','toolchain','frontend_source','upstream','third_party'):
        (dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in (dest/'assets').glob('*.S'):
        p.write_text(p.read_text().replace(str(source),str(dest)))
    p=dest/'bwe_fast/assets/bbwenet_data_rowmajor.c';s=p.read_text();dims={}
    for row in re.findall(r'linear_init\(([^;]+)',s):
        match=re.search(r'"([^"]+_weights_int8)"',row)
        if match:
            cols,rows=map(int,re.search(r',\s*(\d+),\s*(\d+)\)\)',row).groups())
            assert cols%16==0 and rows%8==0
            dims[match[1]]=(rows,cols)
    receipts=[]
    def repack(m):
        name,n,body=m.groups();values=list(map(int,re.findall(r'-?\d+',body)))
        assert len(values)==int(n)
        rows,cols=dims[name];assert rows*cols==len(values)
        packed=[];restored=[None]*len(values)
        for first in range(0,rows,8):
            for half in (0,4):
                for block in range(0,cols,16):
                    for row in range(first+half,first+half+4):
                        for col in range(block,block+16):
                            packed.append(values[row*cols+col])
                            restored[row*cols+col]=packed[-1]
        assert restored==values
        digest=lambda v:hashlib.sha256(bytes(x&255 for x in v)).hexdigest()
        receipts.append(dict(name=name,rows=rows,cols=cols,canonical_row_major_sha256=digest(values),
            packed8_sha256=digest(packed),inverse_permutation_exact=True))
        return 'static const opus_int8 '+name+'['+n+'] = {\n'+','.join(map(str,packed))+'\n};'
    s=re.sub(r'static const opus_int8 (\w+)\[(\d+)\]\s*=\s*\{(.*?)\};',repack,s,flags=re.S)
    assert len(receipts)==len(dims)==9;p.write_text(s)
    p=dest/'src/bwe_packed_matrix8.S';s=(dest/'src/packed_matrix8.S').read_text()
    s=s.replace('packed_matrix8','bwe_packed_matrix8').replace(' vpush {d8-d11}\n','').replace(' vpop {d8-d11}\n','')
    s=s.replace(' ldr r0,[r9,#28]\n vdup.32 q4,r0\n','')
    s=s.replace(' vmul.f32 q0,q0,q4\n vmul.f32 q1,q1,q4\n','')
    a=s.index(' ldr r0,[r9,#12]\n');b=s.index(' vstrw.u32 q0,[r7],#16\n',a)
    s=s[:a]+s[b:]
    s=s.replace(' * Context is caller-local and scales/bias fields advance across full tiles.',
        ' * Exact BWE conversion then scale multiplication; no extra bias/add-zero.')
    p.write_text(s)
    p=dest/'bwe_fast/dnn/vec.h';s=p.read_text()
    helper='''typedef struct {
 const opus_int8*x,*w;const float*scales,*unused_bias;float*out;
 unsigned blocks,tiles,unused_scale,unused_accumulate;
} BWEPackedMatrix;
extern void bwe_packed_matrix8(BWEPackedMatrix*);
'''
    s=s.replace('#ifdef USE_SU_BIAS',helper+'\n#ifdef USE_SU_BIAS',1)
    pattern=r'static inline void cgemv8x4\(float\*out,const opus_int8\*w,const float\*scale,int rows,int cols,const float\*_x\)\{.*?\n\}'
    replacement='''static inline void cgemv8x4(float*out,const opus_int8*w,const float*scale,int rows,int cols,const float*_x){
 opus_int8 x[MAX_INPUTS];bwe_quantize_exact(x,_x,cols);
 BWEPackedMatrix context={x,w,scale,NULL,out,(unsigned)cols/16,(unsigned)rows/8,0,0};
 bwe_packed_matrix8(&context);
}'''
    s,n=re.subn(pattern,lambda m:replacement,s,flags=re.S);assert n==2;p.write_text(s)
    write(ROOT/'BWE_PACKED_IDENTITY.json',dict(matrices=receipts,code_bytes_unchanged_after_inverse_permutation=True,
        no_new_quantization=True,arithmetic='Same exact int32 dot sums, FP32 conversion then original FP32 scale; bias/diag remain in original caller'))

if __name__=='__main__':main()
