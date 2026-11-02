"""Replace per-sample finite checks with the validated MVE bit checks."""
import shutil
from build_resources import ROOT,write,sha

def main():
    source=ROOT/'native_zero_tail';dest=ROOT/'native_zero_tail_vec';dest.mkdir(exist_ok=False)
    shutil.copytree(source/'upstream',dest/'upstream')
    for name in ('src','assets','frontend_patched','bin','toolchain','frontend_source','bwe_fast','third_party'):
        (dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    p=dest/'upstream/dnn/nndsp.c';s=p.read_text();before=sha(p)
    a=s.index('  for(int k=0;k<count+length-1;k++){');b=s.index('  if(finite)prefix=zero_from;',a)
    s=s[:a]+'''  int k=0,n=count+length-1;
  for(;k+4<=n;k+=4){
   uint32x4_t bits=vandq_u32(vreinterpretq_u32_f32(vld1q_f32(y+k)),vdupq_n_u32(0x7fffffffu));
   if(vcmphiq_n_u32(bits,0x7f7fffffu)){finite=0;break;}
  }
  if(finite)for(;k<n;k++){union{float f;unsigned u;}bits={.f=y[k]};if((bits.u&0x7fffffffu)>0x7f7fffffu){finite=0;break;}}
'''+s[b:];p.write_text(s)
    write(ROOT/'ZERO_TAIL_VEC_IDENTITY.json',dict(before_sha256=before,after_sha256=sha(p),
        policy='Same finite/normal guard semantics with four-lane raw-bit checks; no libc memcpy in the sample scan.'))

if __name__=='__main__':main()
