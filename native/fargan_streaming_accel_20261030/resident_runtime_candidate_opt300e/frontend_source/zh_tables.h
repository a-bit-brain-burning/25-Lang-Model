#ifndef NATIVE_ZH_TABLES_H
#define NATIVE_ZH_TABLES_H
#include <stdint.h>
#include <stddef.h>
/* table0 words; table1 characters. Returns0 found,1 absent,negative corrupt. */
int native_zh_lookup(int,const uint32_t*,int,unsigned char*,size_t,size_t*,int*);
int native_zh_varint(const unsigned char**,const unsigned char*,uint32_t*);
#endif
