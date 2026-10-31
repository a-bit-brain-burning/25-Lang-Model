"""M55 differential tests of old/new packed kernels, including FP edge bits."""
import subprocess
from build_resources import ROOT,NATIVE,WORK,CC,run,write,sha

def main():
    out=ROOT/'build/multiple_load_edges_v1';out.mkdir(exist_ok=False)
    original=ROOT/'native_spanish_lean/src';modified=ROOT/'native_multiple_load/src'
    inputs=[NATIVE/'src'/x for x in ('startup.S','bench.c','syscalls.c')]
    for filename,symbol,replacement in [('packed_matrix8.S','packed_matrix8','reference_matrix8'),
        ('packed_matrix_pair.S','packed_matrix_pair','reference_matrix_pair'),
        ('bwe_packed_matrix8.S','bwe_packed_matrix8','reference_bwe_matrix8')]:
        p=out/('reference_'+filename);p.write_text((original/filename).read_text().replace(symbol,replacement))
        inputs += [p,modified/filename]
    test=out/'test.c';test.write_text('''#include "bench.h"
#include <stdint.h>
#include <string.h>
typedef struct{const int8_t*x,*w;const float*scales,*bias;float*y;unsigned blocks,tiles;float scale;unsigned accumulate;}Single;
typedef struct{const int8_t*x0,*x1,*w;const float*scales,*bias;float*y0,*y1;unsigned blocks,tiles;float scale0,scale1;unsigned accumulate;}Pair;
extern void packed_matrix8(Single*),reference_matrix8(Single*),bwe_packed_matrix8(Single*),reference_bwe_matrix8(Single*);
extern void packed_matrix_pair(Pair*),reference_matrix_pair(Pair*);
static int8_t weights[65536] __attribute__((aligned(16))),x0[2048] __attribute__((aligned(16))),x1[2048] __attribute__((aligned(16)));
static float scale[80],bias[80],a[80],b[80],c[80],d[80];
static uint32_t seed=31;static uint32_t next(void){seed=seed*1664525u+1013904223u;return seed;}
static float bits(uint32_t value){union{uint32_t u;float f;}v={.u=value};return v.f;}
int main(void){
 static const unsigned blocks_list[]={1,2,3,4,5,8,12,24,48};
 static const uint32_t specials[]={0,0x80000000u,0x00800000u,0x80800000u,0x3f800000u,0xbf000000u,0x7f800000u,0xff800000u,0x7fc12345u};
 unsigned checks=0;
 for(unsigned test=0;test<256;test++){
  unsigned blocks=blocks_list[test%9],tiles=test%10+1,rows=tiles*8;
  for(unsigned i=0;i<blocks*tiles*128;i++)weights[i]=(int8_t)next();
  for(unsigned i=0;i<blocks*16;i++){x0[i]=(int8_t)next();x1[i]=(int8_t)next();}
  for(unsigned i=0;i<rows;i++){
   scale[i]=test<128?(float)((int)(next()%1001)-500)*.001f:bits(specials[(test+i)%9]);
   bias[i]=(float)((int)(next()%1001)-500)*.001f;a[i]=b[i]=c[i]=d[i]=(float)((int)(next()%1001)-500)*.001f;
  }
  float factor=test<128?.000123f:bits(specials[test%9]);
  Single old={x0,weights,scale,test&1?bias:NULL,a,blocks,tiles,factor,test&2?1:0};Single current=old;current.y=b;
  reference_matrix8(&old);packed_matrix8(&current);
  if(memcmp(a,b,rows*4)){bench_metric("failed_matrix8_case",test);return 1;}checks++;
  for(unsigned i=0;i<rows;i++)a[i]=b[i]=c[i]=d[i]=(float)((int)(next()%1001)-500)*.001f;
  Pair op={x0,x1,weights,scale,test&1?bias:NULL,a,c,blocks,tiles*2,factor,-factor,test&2?1:0};Pair np=op;np.y0=b;np.y1=d;
  reference_matrix_pair(&op);packed_matrix_pair(&np);
  if(memcmp(a,b,rows*4)||memcmp(c,d,rows*4)){bench_metric("failed_pair_case",test);return 2;}checks++;
  old=(Single){x0,weights,scale,NULL,a,blocks,tiles,0,0};current=old;current.y=b;
  reference_bwe_matrix8(&old);bwe_packed_matrix8(&current);
  if(memcmp(a,b,rows*4)){bench_metric("failed_bwe_case",test);return 3;}checks++;
 }
 bench_metric("multiple_load_bit_exact_checks",checks);bench_memory_report();bench_puts("N25_MULTIPLE_LOAD_KERNELS_PASS\\n");return 0;
}
''');inputs.append(test)
    flags=['-O3','-std=c11','-ffreestanding','-fno-builtin','-fno-fast-math','-ffp-contract=off',
        '-ffunction-sections','-fdata-sections','-fno-unwind-tables','-fno-asynchronous-unwind-tables','-I'+str(NATIVE/'src')]
    commands=[];objects=[]
    for i,p in enumerate(inputs):
        obj=out/(str(i)+'.o');objects.append(obj);commands.append(run([CC,*flags,'-c',p,'-o',obj],out/(str(i)+'.log')))
    elf=out/'firmware.elf'
    commands.append(run([CC,'-nostdlib','-nostartfiles','-Wl,--gc-sections','-Wl,--build-id=none',
        '-Wl,--defsym=ARENA_BYTES=16','-Wl,--defsym=STACK_BYTES=32768','-T',NATIVE/'src/m55.ld',*objects,
        '-Wl,--start-group','-lm','-lc','-lgcc','-Wl,--end-group','-o',elf],out/'link.log'))
    report=ROOT/'reports/multiple_load_edges_v1';report.mkdir(exist_ok=False)
    command=[str(WORK/'local_qemu_20261030/bin/qemu-system-arm'),'-M','mps3-an547','-cpu','cortex-m55',
        '-accel','tcg,thread=single','-nographic','-serial','none','-monitor','none',
        '-semihosting-config','enable=on,target=native','-kernel',str(elf)]
    result=subprocess.run(command,cwd=report,capture_output=True,text=True,timeout=60)
    output=result.stdout+result.stderr;(report/'qemu.log').write_text(output)
    passed=result.returncode==0 and'N25_MULTIPLE_LOAD_KERNELS_PASS'in output
    write(report/'RESULT.json',dict(passed=passed,checks=768,elf_sha256=sha(elf),command=command,commands=commands,returncode=result.returncode))
    print(output);assert passed

if __name__=='__main__':main()
