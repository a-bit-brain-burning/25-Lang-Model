"""Target-side differential regression for adaptive-convolution zero padding."""
import json
import subprocess
from build_resources import ROOT,WORK,NATIVE,CC,run,write,sha,elf_memory

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='zero_tail_edge_v1');parser.add_argument('--native-root',default='native_zero_tail');args=parser.parse_args()
    out=ROOT/'build'/args.tag;out.mkdir(exist_ok=False)
    old=(ROOT/'native_view_reuse/upstream/dnn/nndsp.c').read_text()
    new=(ROOT/args.native_root/'upstream/dnn/nndsp.c').read_text()
    tail=old[old.index('static __attribute__((noinline)) void adaconv_xcorr4_tail'):old.index('\nunsigned long long bwe_adaconv_outer16')]
    old=old[old.index('static void adaconv_xcorr_mve'):old.index('\nstatic void scale_kernel')].replace('adaconv_xcorr_mve','old_xcorr')
    new=new[new.index('static int adaconv_normal4'):new.index('\nstatic void scale_kernel')].replace('adaconv_xcorr_mve','new_xcorr')
    source=out/'kernel_test.c'
    source.write_text('''#include "bench.h"
#include <arm_mve.h>
#include <stdint.h>
#include <string.h>
'''+tail+old+new+'''
static uint32_t seed=20261031;
static uint32_t next(void){seed=seed*1664525u+1013904223u;return seed;}
static float bits(uint32_t u){float f;memcpy(&f,&u,4);return f;}
int main(void){
 float kernel[32],input[192],before[160],after[160];unsigned cases=0;
 const uint32_t edges[]={0,0x80000000u,1,0x80000001u,0x007fffffu,0x807fffffu,
  0x00800000u,0x80800000u,0x3f800000u,0xbf800000u,0x7f800000u,0xff800000u,0x7fc12345u};
 const unsigned counts[]={1,3,4,15,16,17,40,80,120,160};
 for(unsigned test=0;test<2048;test++){
  unsigned prefix=test%33,count=counts[test%10];
  for(unsigned i=0;i<32;i++)kernel[i]=i<prefix?(float)((int)(next()%2001)-1000)*.000125f:0.f;
  for(unsigned i=0;i<count+31;i++)input[i]=test<1024?(float)((int)(next()%20001)-10000)*.000125f:bits(edges[next()%(sizeof(edges)/4)]);
  if(test%7==0)for(unsigned i=0;i<prefix;i++)kernel[i]=bits(edges[(test+i)%10]);
  old_xcorr(kernel,input,before,32,count,0);new_xcorr(kernel,input,after,32,count,0,prefix);
  if(memcmp(before,after,count*4)){bench_metric("failed_case",test);bench_metric("prefix",prefix);bench_metric("count",count);return 2;}
  cases++;
 }
 bench_metric("zero_padding_bit_exact_cases",cases);bench_puts("N25_ZERO_TAIL_EDGE_PASS\\n");return 0;
}
''')
    flags=['-O3','-std=c11','-ffreestanding','-fno-builtin','-fno-fast-math','-ffp-contract=off',
        '-ffunction-sections','-fdata-sections','-fno-unwind-tables','-fno-asynchronous-unwind-tables',
        '-I'+str(NATIVE/'src')]
    objects=[];commands=[]
    inputs=[NATIVE/'src'/name for name in ('startup.S','bench.c','syscalls.c')]+[source]
    extra=ROOT/args.native_root/'src/bwe_adaconv16.S'
    if extra.exists():inputs.append(extra)
    for i,p in enumerate(inputs):
        obj=out/(str(i)+'.o');objects.append(obj);commands.append(run([CC,*flags,'-c',p,'-o',obj],out/(str(i)+'.log')))
    elf=out/'firmware.elf'
    commands.append(run([CC,'-nostdlib','-nostartfiles','-Wl,--gc-sections','-Wl,--build-id=none',
        '-Wl,--defsym=ARENA_BYTES=16','-Wl,--defsym=STACK_BYTES=32768','-T',NATIVE/'src/m55.ld',
        *objects,'-Wl,--start-group','-lm','-lc','-lgcc','-Wl,--end-group','-o',elf],out/'link.log'))
    report=ROOT/'reports'/args.tag;report.mkdir(exist_ok=False)
    command=[str(WORK/'local_qemu_20261030/bin/qemu-system-arm'),'-M','mps3-an547','-cpu','cortex-m55',
        '-accel','tcg,thread=single','-nographic','-serial','none','-monitor','none',
        '-semihosting-config','enable=on,target=native','-kernel',str(elf)]
    result=subprocess.run(command,cwd=report,capture_output=True,text=True,timeout=60)
    output=result.stdout+result.stderr;(report/'qemu.log').write_text(output)
    passed=result.returncode==0 and'N25_ZERO_TAIL_EDGE_PASS'in output
    write(report/'RESULT.json',dict(passed=passed,elf_sha256=sha(elf),commands=commands,
        command=command,source_sha256=sha(source),cases=2048,returncode=result.returncode))
    print(output);assert passed

if __name__=='__main__':main()
