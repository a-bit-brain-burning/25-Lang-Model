"""Skip provably neutral padded-zero products, retaining all zero edge cases."""
import shutil
from build_resources import ROOT,write,sha

def main():
    source=ROOT/'native_view_reuse';dest=ROOT/'native_zero_tail';dest.mkdir(exist_ok=False)
    for name in ('src','assets','frontend_patched','upstream'):
        shutil.copytree(source/name,dest/name)
    for name in ('bin','toolchain','frontend_source','bwe_fast','third_party'):
        (dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in (dest/'assets').glob('*.S'):
        p.write_text(p.read_text().replace(str(source),str(dest)))
    p=dest/'upstream/dnn/nndsp.c';s=p.read_text();before=sha(p)
    old='''static void adaconv_xcorr_mve(const float*x,const float*y,float*out,int length,int count,int arch){
 int i=0;'''
    new='''static int adaconv_normal4(float32x4_t x){
 uint32x4_t bits=vandq_u32(vreinterpretq_u32_f32(x),vdupq_n_u32(0x7fffffffu));
 /* Only normal finite accumulators may omit additions of signed zero. */
 return !vcmphiq_n_u32(vsubq_n_u32(bits,0x00800000u),0x7effffffu);
}
static void adaconv_xcorr_mve(const float*x,const float*y,float*out,int length,int count,int arch,int zero_from){
 int prefix=length;
 if(zero_from>=0&&zero_from<length){
  int finite=1;
  for(int k=0;k<count+length-1;k++){
   unsigned bits;memcpy(&bits,y+k,4);if((bits&0x7fffffffu)>0x7f7fffffu){finite=0;break;}
  }
  if(finite)prefix=zero_from;
 }
 int i=0;'''
    assert old in s;s=s.replace(old,new,1)
    # Preserve the original full operation sequence in the scalar/vector tail.
    a=s.index('static void adaconv_xcorr_mve(');b=s.index('\nstatic void scale_kernel',a)
    part=s[a:b].replace('j+2<=length','j+2<=prefix').replace('if(j<length){','if(j<prefix){')
    part=part.replace('   d=vaddq_f32(d,vmulq_f32(vld1q_f32(yp+12),v));\n  }','''   d=vaddq_f32(d,vmulq_f32(vld1q_f32(yp+12),v));
   xp++;yp++;j++;
  }
  if(j<length&&!(adaconv_normal4(a)&&adaconv_normal4(b)&&adaconv_normal4(c)&&adaconv_normal4(d))){
   /* Includes signed zero/subnormal/nonfinite sums: execute every old term. */
   for(;j<length;j++,xp++,yp++){
    float32x4_t v=vdupq_n_f32(*xp);
    a=vaddq_f32(a,vmulq_f32(vld1q_f32(yp),v));
    b=vaddq_f32(b,vmulq_f32(vld1q_f32(yp+4),v));
    c=vaddq_f32(c,vmulq_f32(vld1q_f32(yp+8),v));
    d=vaddq_f32(d,vmulq_f32(vld1q_f32(yp+12),v));
   }
  }''',1)
    assert 'adaconv_normal4(a)'in part;s=s[:a]+part+s[b:]
    s=s.replace('channel_buffer0, ADACONV_MAX_KERNEL_SIZE, overlap_size, arch);',
        'channel_buffer0, ADACONV_MAX_KERNEL_SIZE, overlap_size, arch, kernel_size);')
    s=s.replace('channel_buffer1, ADACONV_MAX_KERNEL_SIZE, frame_size, arch);',
        'channel_buffer1, ADACONV_MAX_KERNEL_SIZE, frame_size, arch, kernel_size);')
    p.write_text(s)
    write(ROOT/'ZERO_TAIL_IDENTITY.json',dict(before_sha256=before,after_sha256=sha(p),
        policy='Only omit OPUS_CLEAR padding products when all source samples are finite and all four vector accumulators are normal finite. Zero, subnormal, nonfinite and short-tail cases retain original products/additions.',
        weights_changed=False,new_approximations=False))

if __name__=='__main__':main()
