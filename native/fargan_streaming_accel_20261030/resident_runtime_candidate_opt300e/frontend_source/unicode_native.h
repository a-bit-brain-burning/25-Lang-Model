#ifndef TTS_NATIVE_UNICODE_H
#define TTS_NATIVE_UNICODE_H
#include <stddef.h>
#include <stdint.h>
int native_utf8_read(const char**cursor,uint32_t*cp);
int native_utf8_write(uint32_t cp,char*out,size_t capacity);
int native_unicode_nfd(const char*text,uint32_t*out,int capacity);
int native_unicode_normalize(const char*text,char*out,size_t capacity);
int native_letter(uint32_t cp);
int native_unicode_nfc(const char*,char*,size_t);
#endif
