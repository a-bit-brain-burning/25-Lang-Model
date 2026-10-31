#ifndef FIXED_INPUT_H
#define FIXED_INPUT_H
#include "../src/text_frontend.h"
#include "../src/language_norms.h"
extern N25TextFeatures n25_text_input;
#define PHONE_COUNT n25_text_input.count
#define input_phones n25_text_input.phones
#define input_pause n25_text_input.pause
#define input_phonetic n25_text_input.phonetic
#define input_context n25_text_input.context
#define LANGUAGE_ID n25_current_norm->language_id
#define PHONETIC_DIM n25_current_norm->phonetic_dim
#define CONTEXT_DIM n25_current_norm->context_dim
#define ACOUSTIC_DIM n25_current_norm->acoustic_dim
#define norm_phonetic_mean n25_current_norm->phonetic_mean
#define norm_phonetic_std n25_current_norm->phonetic_std
#define norm_context_mean n25_current_norm->context_mean
#define norm_context_std n25_current_norm->context_std
#define norm_mean n25_current_norm->mean
#define norm_std n25_current_norm->std
/* The legacy HiFi helpers are unreferenced in this FARGAN-only image. */
static const float resample_filter[1]={0};
#define RESAMPLE_TAPS 1
#define RESAMPLE_REMOVE 0
#endif
