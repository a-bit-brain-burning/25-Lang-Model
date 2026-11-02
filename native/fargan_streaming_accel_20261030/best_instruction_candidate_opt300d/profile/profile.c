/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Guest instruction events, not cycles. One vCPU only. */
#include "qemu-plugin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version=QEMU_PLUGIN_VERSION;
typedef struct Counter {
    uint64_t pc,hits,start,stop,cold;
    unsigned size;
    char *disas;
    struct Counter *next;
} Counter;
#define BUCKETS 65536
static Counter *table[BUCKETS];
static uint64_t begin_pc,end_pc;
static unsigned started,stopped;
static const char *output;

static void quote(FILE*f,const char*s){
    fputc('"',f);
    for(;*s;s++){if(*s=='"')fputc('"',f);fputc(*s,f);}
    fputc('"',f);
}
static void dump(void){
    FILE*f=fopen(output,"w");
    if(!f){perror("profile fopen");abort();}
    fprintf(f,"pc,size,request_instruction_events,cold_instruction_events,disassembly\n");
    for(unsigned b=0;b<BUCKETS;b++)for(Counter*c=table[b];c;c=c->next){
        if(!c->stop&&!c->cold)continue;
        fprintf(f,"0x%"PRIx64",%u,%"PRIu64",%"PRIu64",",c->pc,c->size,c->stop-c->start,c->cold);
        quote(f,c->disas);fputc('\n',f);
    }
    fclose(f);
}
static void begin(unsigned cpu,void*unused){
    (void)cpu;(void)unused;
    if(started)return;
    started=1;
    for(unsigned b=0;b<BUCKETS;b++)for(Counter*c=table[b];c;c=c->next)c->cold=c->start=c->hits;
}
static void end(unsigned cpu,void*unused){
    (void)cpu;(void)unused;
    if(stopped)return;
    if(!started){fprintf(stderr,"profile end before begin\n");abort();}
    stopped=1;
    for(unsigned b=0;b<BUCKETS;b++)for(Counter*c=table[b];c;c=c->next)c->stop=c->hits;
    dump();
}
static void translate(qemu_plugin_id_t id,struct qemu_plugin_tb*tb){
    (void)id;
    for(size_t i=0;i<qemu_plugin_tb_n_insns(tb);i++){
        struct qemu_plugin_insn*insn=qemu_plugin_tb_get_insn(tb,i);
        uint64_t pc=qemu_plugin_insn_vaddr(insn);
        unsigned b=(pc>>1)%BUCKETS;
        Counter*c;
        for(c=table[b];c&&c->pc!=pc;c=c->next){}
        if(!c){
            c=calloc(1,sizeof(*c));if(!c)abort();
            c->pc=pc;c->size=qemu_plugin_insn_size(insn);
            c->disas=qemu_plugin_insn_disas(insn);
            c->next=table[b];table[b]=c;
        }
        if(pc==begin_pc)qemu_plugin_register_vcpu_insn_exec_cb(insn,begin,QEMU_PLUGIN_CB_NO_REGS,NULL);
        if(pc==end_pc)qemu_plugin_register_vcpu_insn_exec_cb(insn,end,QEMU_PLUGIN_CB_NO_REGS,NULL);
        qemu_plugin_register_vcpu_insn_exec_inline(insn,QEMU_PLUGIN_INLINE_ADD_U64,&c->hits,1);
    }
}
static void finish(qemu_plugin_id_t id,void*unused){
    (void)id;(void)unused;
    fprintf(stderr,"PROFILE_MARKERS started=%u stopped=%u begin=0x%"PRIx64" end=0x%"PRIx64"\n",started,stopped,begin_pc,end_pc);
}
QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,const qemu_info_t*info,int argc,char**argv){
    if(!info->system_emulation||strcmp(info->target_name,"arm")||info->system.smp_vcpus!=1)return -1;
    for(int i=0;i<argc;i++){
        if(!strncmp(argv[i],"begin=",6))begin_pc=strtoull(argv[i]+6,NULL,0);
        else if(!strncmp(argv[i],"end=",4))end_pc=strtoull(argv[i]+4,NULL,0);
        else if(!strncmp(argv[i],"out=",4))output=argv[i]+4;
        else return -1;
    }
    if(!begin_pc||!end_pc||!output)return -1;
    qemu_plugin_register_vcpu_tb_trans_cb(id,translate);
    qemu_plugin_register_atexit_cb(id,finish,NULL);
    return 0;
}
