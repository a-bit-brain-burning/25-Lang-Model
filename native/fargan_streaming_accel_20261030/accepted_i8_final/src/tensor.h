#ifndef NEW25_TENSOR_H
#define NEW25_TENSOR_H
#include <stdint.h>
/* File is little endian. Float IEEE754 binary32; this ABI is storage-quantized.
 * Activations and accumulation here are FP32; replace kernels for integer MAC.
 * ConvTranspose1d [in,out,k] uses scale axis 1, other matrices axis 0.
 */
typedef struct {
    uint32_t offset, scale_offset, elements, dims[3];
    uint8_t rank, dtype, axis, reserved; /* dtype 0=I8, 1=F16, 2=F32 */
} n25_tensor;
float n25_value(const unsigned char *blob,const n25_tensor *tensor,uint32_t index);
void n25_unpack(const unsigned char *blob,const n25_tensor *tensor,float *out);
void n25_linear(const unsigned char *blob,const n25_tensor *w,const n25_tensor *bias,
                const float *x,float *y,unsigned rows);
/* x/y channel-first [channel,time], groups must divide input/output channels. */
void n25_conv1d(const unsigned char *blob,const n25_tensor *w,const n25_tensor *bias,
               const float *x,float *y,unsigned steps,unsigned dilation,unsigned pad,unsigned groups);
void n25_conv_transpose1d(const unsigned char *blob,const n25_tensor *w,const n25_tensor *bias,
                         const float *x,float *y,unsigned steps,unsigned stride,unsigned pad);
#endif
