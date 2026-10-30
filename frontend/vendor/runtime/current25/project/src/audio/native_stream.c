/* Streaming ABI adapter around the frozen phase2.7B core. No new algorithm. */
#include "native.c"

typedef struct {
    TetContext context;
    TetStats stats;
    const float *basis,*bank;
    int dimensions,bands,length,ready;
    float pcm_gain;
} TetStream;

size_t tetts_stream_bytes(void){return sizeof(TetStream);}

int tetts_stream_initialize(void *memory,int dim,int bands,int length,const float *basis,
                            const float *bank,uint32_t seed,float pcm_gain){
    if(!memory||dim<2||dim>32||bands<3||bands>5||length<8||length>256)return -1;
    TetStream *s=memory;memset(s,0,sizeof(*s));initialize(&s->context,seed,bands);
    s->dimensions=dim;s->bands=bands;s->length=length;s->basis=basis;s->bank=bank;s->pcm_gain=pcm_gain;
    return 0;
}

int tetts_stream_frame(void *memory,const float *begin,const float *end,int count,
                        float *output,int16_t *pcm){
    if(!memory||count<1||count>160)return -1;
    TetStream *s=memory;TetContext *c=&s->context;TetStats *stats=&s->stats;int length=s->length;
    if(!s->ready){int r=build_filter(begin,s->basis,s->dimensions,length,c->filter_a,c->causal,stats);if(r)return r;s->ready=1;}
    int r=build_filter(end,s->basis,s->dimensions,length,c->filter_b,c->causal,stats);if(r)return r;
    source_block(c,begin,end,s->dimensions,s->bands,s->bank,count,0,stats);
    for(int n=0;n<count;n++){
        c->history[c->history_pos]=c->source[n];c->history[c->history_pos+length]=c->source[n];
        const float *past=c->history+c->history_pos+length;float a=0,b=0;
        for(int k=0;k<length;k++){a+=c->filter_a[k]*past[-k];b+=c->filter_b[k]*past[-k];}
        float alpha=(float)n/HOP;output[n]=a+(b-a)*alpha;
        if(!isfinite(output[n]))return -3;
        float x=output[n]*s->pcm_gain*32768.f;
        if(x>=32768.f||x< -32768.f)stats->clipped_pcm++;
        pcm[n]=(int16_t)lrintf(clip(x,-32768.f,32767.f));
        c->history_pos++;if(c->history_pos==length)c->history_pos=0;
    }
    stats->fir_mac+=(uint64_t)2*length*count;stats->generated_samples+=count;
    memcpy(c->filter_a,c->filter_b,(size_t)length*sizeof(float));
    return 0;
}
