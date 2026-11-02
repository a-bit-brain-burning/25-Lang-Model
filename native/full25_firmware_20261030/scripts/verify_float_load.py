"""Target differential check of the unchanged FP32 BWE matrix expression."""
import subprocess
from build_resources import ROOT,NATIVE,WORK,CC,run,write,sha

def function(path,name):
    text=path.read_text();a=text.index('static inline void sgemv(');b=text.index('static inline void sparse_sgemv8x4',a)
    return text[a:b].replace('sgemv(',name+'(',1)

def main():
    out=ROOT/'build/float_matrix_edges_v1';out.mkdir(exist_ok=False)
    original=ROOT/'native_multiple_load/bwe_fast/dnn/vec.h';current=ROOT/'native_float_load/bwe_fast/dnn/vec.h'
    source=out/'test.c';source.write_text('''#include "bench.h"
#include <arm_mve.h>
#include <stdint.h>
#include <string.h>
typedef struct{const float*w,*x;float*out;unsigned tiles,cols,stride_bytes;}BWESgemv16;
extern void bwe_sgemv16(BWESgemv16*);
'''+function(original,'reference_sgemv')+function(current,'current_sgemv')+'''
static float weights[32768] __attribute__((aligned(16))),input[384],a[80],b[80];
static uint32_t seed=73;static uint32_t next(void){seed=seed*1664525u+1013904223u;return seed;}
static float bits(uint32_t value){union{uint32_t u;float f;}x={.u=value};return x.f;}
int main(void){
 const unsigned rows_list[]={1,3,4,15,16,17,31,32,64,80},cols_list[]={1,2,3,4,5,15,16,17,42,128,342};
 const uint32_t edges[]={0,0x80000000u,1,0x80000001u,0x007fffffu,0x807fffffu,0x00800000u,0x80800000u,
   0x3f800000u,0xbf800000u,0x7f800000u,0xff800000u,0x7fc12345u,0x7fa12345u};
 for(unsigned test=0;test<512;test++){
  unsigned rows=rows_list[test%10],cols=cols_list[test%11],stride=rows+(test%3?0:4);
  for(unsigned i=0;i<cols*stride;i++)weights[i]=test<256?(float)((int)(next()%2001)-1000)*.000125f:bits(edges[next()%14]);
  for(unsigned i=0;i<cols;i++)input[i]=test<256?(float)((int)(next()%2001)-1000)*.000125f:bits(edges[next()%14]);
  for(unsigned i=0;i<80;i++)a[i]=b[i]=bits(0x7fc56789u);
  reference_sgemv(a,weights,rows,cols,stride,input);current_sgemv(b,weights,rows,cols,stride,input);
  if(memcmp(a,b,sizeof(a))){bench_metric("failed_case",test);bench_metric("rows",rows);bench_metric("cols",cols);return 1;}
 }
 bench_metric("float_matrix_bit_exact_cases",512);bench_memory_report();bench_puts("N25_FLOAT_MATRIX_EDGE_PASS\\n");return 0;
}
''')
    inputs=[NATIVE/'src'/x for x in ('startup.S','bench.c','syscalls.c')]+[source,ROOT/'native_float_load/src/bwe_sgemv16.S']
    flags=['-O3','-std=c11','-ffreestanding','-fno-builtin','-fno-fast-math','-ffp-contract=off',
        '-ffunction-sections','-fdata-sections','-fno-unwind-tables','-fno-asynchronous-unwind-tables','-I'+str(NATIVE/'src')]
    objects=[];commands=[]
    for i,p in enumerate(inputs):
        obj=out/(str(i)+'.o');objects.append(obj);commands.append(run([CC,*flags,'-c',p,'-o',obj],out/(str(i)+'.log')))
    elf=out/'firmware.elf'
    commands.append(run([CC,'-nostdlib','-nostartfiles','-Wl,--gc-sections','-Wl,--build-id=none',
        '-Wl,--defsym=ARENA_BYTES=16','-Wl,--defsym=STACK_BYTES=32768','-T',NATIVE/'src/m55.ld',*objects,
        '-Wl,--start-group','-lm','-lc','-lgcc','-Wl,--end-group','-o',elf],out/'link.log'))
    report=ROOT/'reports/float_matrix_edges_v1';report.mkdir(exist_ok=False)
    command=[str(WORK/'local_qemu_20261030/bin/qemu-system-arm'),'-M','mps3-an547','-cpu','cortex-m55',
        '-accel','tcg,thread=single','-nographic','-serial','none','-monitor','none',
        '-semihosting-config','enable=on,target=native','-kernel',str(elf)]
    result=subprocess.run(command,cwd=report,capture_output=True,text=True,timeout=60)
    output=result.stdout+result.stderr;(report/'qemu.log').write_text(output)
    passed=result.returncode==0 and'N25_FLOAT_MATRIX_EDGE_PASS'in output
    write(report/'RESULT.json',dict(passed=passed,cases=512,elf_sha256=sha(elf),source_sha256=sha(source),
        command=command,commands=commands,returncode=result.returncode))
    print(output);assert passed

if __name__=='__main__':main()
