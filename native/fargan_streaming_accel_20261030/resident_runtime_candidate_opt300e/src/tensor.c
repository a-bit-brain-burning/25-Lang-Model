#include "tensor.h"
#include <arm_mve.h>
extern void profile_dequant(unsigned n);
static float f32(const unsigned char *p) {
    union {uint32_t u;float f;} a;
    a.u=(uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
    return a.f;
}
static float f16(const unsigned char *p) {
    unsigned u=(unsigned)p[0]|((unsigned)p[1]<<8), exponent=(u>>10)&31,mantissa=u&1023;
    float v;
    if(exponent==0)v=(float)mantissa*(1.0f/16777216.0f);
    else {
        union {uint32_t u;float f;} a;
        a.u=((u&32768)<<16)|((exponent+112)<<23)|(mantissa<<13);
        return a.f;
    }
    return (u&32768)?-v:v;
}
float n25_value(const unsigned char *blob,const n25_tensor *t,uint32_t i) {
    const unsigned char *p=blob+t->offset;
    if(t->dtype==1)return f16(p+2*i);
    if(t->dtype==2)return f32(p+4*i);
    unsigned trailing=1;
    for(unsigned d=t->axis+1;d<t->rank;d++)trailing*=t->dims[d];
    unsigned channel=(i/trailing)%t->dims[t->axis];
    if(t->reserved){
        unsigned width=t->elements/t->dims[0],row=i/width,col=i%width,blocks=(width+15)/16;
        if(t->reserved==1)i=((row/8*blocks+col/16)*8+row%8)*16+col%16;
        else i=((row/4*blocks+col/16)*4+row%4)*16+col%16;
    }
    return (float)((int8_t)p[i])*f32(blob+t->scale_offset+4*channel);
}
void n25_unpack(const unsigned char *blob,const n25_tensor *t,float *out) {
    if(t->dtype==0&&t->axis==0&&!t->reserved){
        unsigned width=t->elements/t->dims[0];profile_dequant(t->dims[0]*(width/4));const int8_t*q=(const int8_t*)(blob+t->offset);
        for(unsigned row=0;row<t->dims[0];row++){
            float scale=f32(blob+t->scale_offset+row*4);unsigned i=0;
            for(;i+4<=width;i+=4)vst1q_f32(out+row*width+i,vmulq_n_f32(vcvtq_f32_s32(vldrbq_s32(q+row*width+i)),scale));
            for(;i<width;i++)out[row*width+i]=(float)q[row*width+i]*scale;
        }
    }else for(uint32_t i=0;i<t->elements;i++)out[i]=n25_value(blob,t,i);
}
void n25_linear(const unsigned char *blob,const n25_tensor *w,const n25_tensor *b,
                const float *x,float *y,unsigned rows) {
    unsigned co=w->dims[0],ci=w->dims[1];
    for(unsigned t=0;t<rows;t++)for(unsigned o=0;o<co;o++) {
        float sum=b?n25_value(blob,b,o):0;
        for(unsigned i=0;i<ci;i++)sum+=x[t*ci+i]*n25_value(blob,w,o*ci+i);
        y[t*co+o]=sum;
    }
}
void n25_conv1d(const unsigned char *blob,const n25_tensor *w,const n25_tensor *b,
               const float *x,float *y,unsigned steps,unsigned dilation,unsigned pad,unsigned groups) {
    unsigned co=w->dims[0],ci_group=w->dims[1],k=w->dims[2],co_group=co/groups;
    unsigned out_steps=steps+2*pad-dilation*(k-1);
    for(unsigned o=0;o<co;o++)for(unsigned t=0;t<out_steps;t++) {
        float sum=b?n25_value(blob,b,o):0;
        for(unsigned i=0;i<ci_group;i++)for(unsigned j=0;j<k;j++) {
            int at=(int)(t+j*dilation)-(int)pad;
            if(at>=0&&(unsigned)at<steps)
                sum+=x[((o/co_group)*ci_group+i)*steps+(unsigned)at]*n25_value(blob,w,(o*ci_group+i)*k+j);
        }
        y[o*out_steps+t]=sum;
    }
}
void n25_conv_transpose1d(const unsigned char *blob,const n25_tensor *w,const n25_tensor *b,
                         const float *x,float *y,unsigned steps,unsigned stride,unsigned pad) {
    unsigned ci=w->dims[0],co=w->dims[1],k=w->dims[2],out_steps=(steps-1)*stride-2*pad+k;
    for(unsigned o=0;o<co;o++)for(unsigned t=0;t<out_steps;t++) {
        float sum=b?n25_value(blob,b,o):0;
        for(unsigned i=0;i<ci;i++)for(unsigned j=0;j<k;j++) {
            int at=(int)(t+pad)-(int)j;
            if(at>=0&&(unsigned)at%stride==0&&(unsigned)at/stride<steps)
                sum+=x[i*steps+(unsigned)at/stride]*n25_value(blob,w,(i*co+o)*k+j);
        }
        y[o*out_steps+t]=sum;
    }
}
