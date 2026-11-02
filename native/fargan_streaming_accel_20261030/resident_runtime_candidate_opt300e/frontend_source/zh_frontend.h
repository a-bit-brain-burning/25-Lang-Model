#ifndef NATIVE_ZH_FRONTEND_H
#define NATIVE_ZH_FRONTEND_H
#include "native.h"
int native_zh_encode(const Acoustic*,const char*,float*,int*,int*,float*,int32_t*,int32_t*,uint8_t*);
/* Numeric delivery topology produced from the current text entirely on ARM. */
int native_zh_policy(int,uint8_t*,int32_t(*)[2],int*,int32_t(*)[4],uint8_t*,int*);
void native_zh_reset(void);
#endif
