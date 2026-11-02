/* Optional generic OpenCC T2S normalization before the preserved frontend.
 * Tables originate from the installed Apache-2.0 OpenCC data, not test cases.
 */
#include "zh_kokoro.c"
#include "t2s_tables.h"

#define T2S_PHRASE_COUNT ((int)(sizeof(t2s_phrases)/sizeof(*t2s_phrases)))
static short t2s_first_head[256];
static short t2s_first_next[T2S_PHRASE_COUNT];
static int t2s_index_ready;
static void t2s_build_index(void){
    for(int i=0;i<256;i++)t2s_first_head[i]=-1;
    for(int i=0;i<T2S_PHRASE_COUNT;i++){
        unsigned char c=(unsigned char)t2s_phrases[i].key[0];
        t2s_first_next[i]=t2s_first_head[c];t2s_first_head[c]=(short)i;
    }
    t2s_index_ready=1;
}
int qb_t2s(const char *text, char *out, int capacity) {
    if (!text || !out || capacity < 1) return -20;
    if(!t2s_index_ready)t2s_build_index();
    int used=0;out[0]=0;
    for (const char *p=text; *p;) {
        const T2SPhrase *chosen=NULL;
        for (int i=t2s_first_head[(unsigned char)*p];i>=0;i=t2s_first_next[i]) {
            const T2SPhrase *phrase=&t2s_phrases[i];
            if ((!chosen || phrase->bytes > chosen->bytes) && !strncmp(p,phrase->key,phrase->bytes)) chosen=phrase;
        }
        if (chosen) {
            if (append(out,capacity,&used,chosen->value)) return -20;
            p+=chosen->bytes;continue;
        }
        uint32_t cp;
        if(native_utf8_read(&p,&cp)<1)return -4;
        unsigned lo=0,hi=sizeof(t2s_characters)/sizeof(*t2s_characters);
        while(lo<hi){unsigned mid=lo+(hi-lo)/2;if(t2s_characters[mid][0]<cp)lo=mid+1;else hi=mid;}
        if(lo<sizeof(t2s_characters)/sizeof(*t2s_characters)&&t2s_characters[lo][0]==cp)cp=t2s_characters[lo][1];
        char bytes[8]={0};int count=native_utf8_write(cp,bytes,sizeof(bytes)-1);
        if(count<1)return -4;bytes[count]=0;
        if(append(out,capacity,&used,bytes))return -20;
    }
    return used;
}
int qb_zh_kokoro_t2s(const char *text,char *out,int capacity){
    char normalized[4096];int rc=qb_t2s(text,normalized,sizeof(normalized));
    return rc<0?rc:qb_zh_kokoro(normalized,out,capacity);
}
