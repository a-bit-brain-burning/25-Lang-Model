#include <stdint.h>
void bwe_start(unsigned total_input,unsigned nonpause_input,uint64_t request_start);
void bwe_push(const int16_t *input,unsigned count);
void bwe_finish(void);
void bwe_preview(void);
typedef void (*bwe_pcm_callback)(const int16_t*,unsigned);
void bwe_set_pcm_callback(bwe_pcm_callback callback);
void bwe_model_init(void);
void bwe_cancel(void);
const int16_t *bwe_diagnostic_pcm(unsigned *samples);
uint64_t bwe_first_any_ticks(void);
uint64_t bwe_first_active_ticks(void);
unsigned bwe_finish_count(void);
int bwe_has_active_speech(void);
