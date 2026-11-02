#ifndef N25_NATIVE_FOUR_FRONTEND_H
#define N25_NATIVE_FOUR_FRONTEND_H
#include <stdint.h>
#define N25_FOUR_PHONE_CAP 512
typedef struct {
 unsigned count,phonetic_dim,context_dim;
 uint32_t phones[N25_FOUR_PHONE_CAP],pause[N25_FOUR_PHONE_CAP];
 float phonetic[N25_FOUR_PHONE_CAP*36],context[N25_FOUR_PHONE_CAP*34];
} N25FourFeatures;
typedef struct {
 unsigned count,phonetic_dim,context_dim;
 uint32_t*phones,*pause;float*phonetic,*context;
}N25FourOutput;
int n25_four_frontend_to_view(const char*,const char*,N25FourOutput*);
/* Ordinary NFC text for the named eu_four families. Other input is rejected.
 * Numeric/identifier normalization and remaining language families are separate.
 */
int n25_four_frontend(const char *language,const char *text,N25FourFeatures *out);
#endif
