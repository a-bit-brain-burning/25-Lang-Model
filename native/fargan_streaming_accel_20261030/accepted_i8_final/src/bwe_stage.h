#include <stdint.h>
void bwe_start(unsigned total_input,unsigned nonpause_input,uint64_t request_start);
void bwe_push(const int16_t *input,unsigned count);
void bwe_finish(void);
void bwe_preview(void);
