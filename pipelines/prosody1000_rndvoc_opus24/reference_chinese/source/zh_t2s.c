/* Optional generic OpenCC T2S normalization before the preserved frontend.
 * Tables originate from the installed Apache-2.0 OpenCC data, not test cases.
 */
#include "zh_kokoro.c"
#include "t2s_tables.h"

int qb_t2s(const char *text, char *out, int capacity) {
    if (!text || !out || capacity < 1) return -20;
    int used=0;out[0]=0;
    for (const char *p=text; *p;) {
        const T2SPhrase *chosen=NULL;
        for (unsigned i=0;i<sizeof(t2s_phrases)/sizeof(*t2s_phrases);i++) {
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
