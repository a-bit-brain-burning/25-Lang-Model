#include <stddef.h>
#include "../third_party/puff/puff.h"
int native_deflate_block(const unsigned char*src,size_t bytes,unsigned char*out,size_t cap){
 unsigned long in=(unsigned long)bytes,n=(unsigned long)cap;
 return puff(out,&n,src,&in)||in!=bytes||n!=cap?-1:0;
}
