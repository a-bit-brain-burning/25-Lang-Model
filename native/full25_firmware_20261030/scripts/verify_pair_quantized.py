"""Differential test of direct interleaved quantization, including zero-row paths."""
import subprocess
from build_resources import ROOT,NATIVE,WORK,CC,run,write,sha

def main():
    out=ROOT/'build/pair_quantized_edges_v1';out.mkdir(exist_ok=False)
    original=ROOT/'native_float_load/src';current=ROOT/'native_pair_interleaved/src'
    inputs=[NATIVE/'src'/x for x in ('startup.S','bench.c','syscalls.c')]
    for prefix,folder in [('reference',original),('current',current)]:
        p=out/(prefix+'.c');defines='#define FAST_I8_ACT_SPLIT16 0\n'
        if prefix=='reference':defines+='#define packed_matrix_pair reference_packed_matrix_pair\n#define packed_dot4_pair reference_packed_dot4_pair\n'
        p.write_text(defines+'#include "'+str(folder/'fast_i8_dense.h')+'"\n#include "'+str(folder/'fast_i8_pair.h')+'"\n'+
            'int '+prefix+'_pair(const float*x0,const float*x1,const int8_t*w,const float*s,const float*b,float*y0,float*y1,unsigned ci,unsigned co,int add){return fast_i8_dense_pair(x0,x1,w,s,b,y0,y1,ci,co,add);}\n')
        inputs.append(p)
        for filename,symbol in [('packed_matrix_pair.S','packed_matrix_pair'),('packed_dot4_pair.S','packed_dot4_pair')]:
            if prefix=='reference':
                p=out/('reference_'+filename);p.write_text((folder/filename).read_text().replace(symbol,'reference_'+symbol))
            else:p=folder/filename
            inputs.append(p)
    test=out/'test.c';test.write_text('''#include "bench.h"
#include <stdint.h>
#include <string.h>
extern int reference_pair(const float*,const float*,const int8_t*,const float*,const float*,float*,float*,unsigned,unsigned,int);
extern int current_pair(const float*,const float*,const int8_t*,const float*,const float*,float*,float*,unsigned,unsigned,int);
static int8_t weights[180224] __attribute__((aligned(16)));
static float x0[2048],x1[2048],scales[80],bias[80],a[80],b[80],c[80],d[80];
static uint32_t seed=97;static uint32_t next(void){seed=seed*1664525u+1013904223u;return seed;}
static float bits(uint32_t u){union{uint32_t u;float f;}v={.u=u};return v.f;}
int main(void){
 const unsigned dimensions[]={1,7,15,16,17,20,31,32,128,192,328,688,768,1120,1344,2048};
 for(unsigned test=0;test<512;test++){
  unsigned ci=dimensions[test%16],co=(test%10+1)*8,blocks=(ci+15)/16;
  for(unsigned i=0;i<blocks*16*co;i++)weights[i]=(int8_t)next();
  unsigned kind=test%8;
  for(unsigned i=0;i<ci;i++){
   x0[i]=kind==0||kind==1?0.f:(float)((int)(next()%20001)-10000)*.000125f;
   x1[i]=kind==0||kind==2?0.f:(float)((int)(next()%20001)-10000)*.000125f;
   if(kind==3){x0[i]*=1e-30f;x1[i]*=1e-30f;}
   if(kind==4){x0[i]*=1e30f;x1[i]*=1e30f;}
  }
  if(kind==5)x0[0]=bits(0x7fc12345u);if(kind==6)x1[ci-1]=bits(0x7f800000u);
  for(unsigned i=0;i<80;i++){scales[i]=.000123f;bias[i]=(float)((int)(next()%2001)-1000)*.000125f;a[i]=b[i]=c[i]=d[i]=bits(0x7fc56789u);}
  const float*bs=test&1?bias:NULL;int add=test&2?1:0;
  int ra=reference_pair(x0,x1,weights,scales,bs,a,c,ci,co,add),rb=current_pair(x0,x1,weights,scales,bs,b,d,ci,co,add);
  if(ra!=rb||memcmp(a,b,sizeof(a))||memcmp(c,d,sizeof(c))){bench_metric("failed_case",test);bench_metric("ci",ci);bench_metric("co",co);return 1;}
 }
 bench_metric("pair_quantization_bit_exact_cases",512);bench_memory_report();bench_puts("N25_PAIR_QUANTIZATION_EDGE_PASS\\n");return 0;
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
    report=ROOT/'reports/pair_quantized_edges_v1';report.mkdir(exist_ok=False)
    command=[str(WORK/'local_qemu_20261030/bin/qemu-system-arm'),'-M','mps3-an547','-cpu','cortex-m55',
        '-accel','tcg,thread=single','-nographic','-serial','none','-monitor','none',
        '-semihosting-config','enable=on,target=native','-kernel',str(elf)]
    result=subprocess.run(command,cwd=report,capture_output=True,text=True,timeout=60)
    output=result.stdout+result.stderr;(report/'qemu.log').write_text(output)
    passed=result.returncode==0 and'N25_PAIR_QUANTIZATION_EDGE_PASS'in output
    write(report/'RESULT.json',dict(passed=passed,cases=512,elf_sha256=sha(elf),command=command,commands=commands,returncode=result.returncode))
    print(output);assert passed

if __name__=='__main__':main()
