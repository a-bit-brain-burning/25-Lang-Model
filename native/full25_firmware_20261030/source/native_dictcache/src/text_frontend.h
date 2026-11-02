#ifndef N25_TEXT_FRONTEND_H
#define N25_TEXT_FRONTEND_H
#define N25_TEXT_PHONE_CAP 512
typedef struct{unsigned count,phones[N25_TEXT_PHONE_CAP],pause[N25_TEXT_PHONE_CAP];float phonetic[N25_TEXT_PHONE_CAP*36],context[N25_TEXT_PHONE_CAP*34];}N25TextFeatures;
int n25_text_frontend(const char*,N25TextFeatures*);
#endif
int native_zh_prewarm(void);
