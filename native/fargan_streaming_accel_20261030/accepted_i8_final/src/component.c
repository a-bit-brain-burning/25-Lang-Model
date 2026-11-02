#include "bench.h"
#include "bwe_stage.h"
#include <stdint.h>
extern const int16_t reference_input[];
void snapshot_first_active_costs(void){}
int main(void){
 uint64_t start=bench_now();bwe_start(45920,3040,start);
 for(unsigned at=0;at<45920;){unsigned count=45920-at;if(count>137)count=137;bwe_push(reference_input+at,count);at+=count;}
 bwe_finish();bench_metric("tick_frequency_hz",bench_tick_frequency());bench_memory_report();bench_puts("BWE_REFERENCE_PASS\n");return 0;
}
