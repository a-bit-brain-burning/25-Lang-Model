#ifndef N25_LANGUAGE_NORMS_H
#define N25_LANGUAGE_NORMS_H
#include <stdint.h>
typedef struct __attribute__((aligned(16))) {
 uint32_t language_id,phonetic_dim,context_dim,acoustic_dim;
 float phonetic_mean[36],phonetic_std[36],context_mean[34],context_std[34],mean[39],std[39];
} N25LanguageNorm;
_Static_assert(sizeof(N25LanguageNorm)==896,"Language normalization ABI");
extern const N25LanguageNorm n25_language_norms[25];
extern const N25LanguageNorm *n25_current_norm;
void n25_frontend_runtime_init(void);
#endif
