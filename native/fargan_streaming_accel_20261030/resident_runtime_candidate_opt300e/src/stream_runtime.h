#ifndef TTS_STREAM_RUNTIME_H
#define TTS_STREAM_RUNTIME_H
#include <stdint.h>
/* Native Han/punctuation/pinyin-hint text API. Numeric/Latin normalization,
 * identifier controls and other languages remain separate integration work. */
void tts_runtime_init(void);
void tts_begin_text_request(unsigned request_id,const char*text);
void tts_begin_request(unsigned request_id);
unsigned tts_pull20ms(int16_t output[480]);
void tts_finish_request(void);
void tts_reset_request(void);
void tts_cancel_request(void);
#endif
