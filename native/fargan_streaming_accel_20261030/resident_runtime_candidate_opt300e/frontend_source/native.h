#ifndef TTS_M4_NATIVE_H
#define TTS_M4_NATIVE_H
#include <stdint.h>
#include <stddef.h>
#define TTS_MAX_UNITS 256
#define TTS_MAX_WIDTH 320
#define TTS_MAX_FEATURES 80
#define TTS_MAX_CHANNELS 39
#define TTS_MAX_ROW 1024
/* Weights point directly into a read-only ROM image. All scratch is bounded. */
typedef struct {const int8_t*w;const float*scale,*bias;int rows,cols;} QMatrix;
typedef struct {const float*weight,*bias;int width;} Norm;
typedef struct {const Norm*norm;const QMatrix*temporal,*projection,*mix;int dilation;} Block;
typedef struct {const QMatrix*ih,*hh;} Direction;
typedef struct {int numeric_dim,width,hidden,layers,outputs,embedding_dim;const QMatrix*input;const Norm*norm;const QMatrix*head0,*head1;const int8_t*embedding;const float*embedding_scale;Direction directions[4];} Planner;
typedef struct {const char*language;int phonetic_dim,context_dim,width,spectral_dim,channels,blocks,max_frames,routed;const Block*block[4];const Block*core;const QMatrix*down,*up,*input;const Norm*norm;const QMatrix*acoustic0,*acoustic1,*duration0,*duration1,*skip0,*skip1,*boundary0,*boundary1;const float*means,*scales;float prior;const Planner*planner;} Acoustic;
extern const Acoustic*const native_models[];
extern const int native_model_count;
int native_cache_prepare(int model);
size_t native_cache_used_bytes(void);
int native_find_language(const char*name);
size_t native_workspace_bytes(void);
int native_planner(int model,int words,const float*numeric,const int32_t*characters,float*plan);
int native_acoustic(int model,int units,const float*features,const uint8_t*edge,float*controls,int32_t*duration);
int native_matrix(const QMatrix*m,int n,const float*x,float*out);
#endif
