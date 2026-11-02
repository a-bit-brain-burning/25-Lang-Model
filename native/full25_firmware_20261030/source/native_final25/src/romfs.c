/* Read-only eSpeak resource filesystem over the firmware's compressed FEP data.
 * These linker wrappers never forward resource reads or allocations to a host.
 */
#include "romfs.h"
#include "fep_runtime.h"
#include "fep_assets.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <dirent.h>
#include "bench.h"
#include "storage.h"
#include "romfs_capacity.h"

#ifndef ROMFS_HEAP_BYTES
#define ROMFS_HEAP_BYTES 655360
#endif
#define HEAP_BYTES ROMFS_HEAP_BYTES
#define BLOCK_BYTES 262144
static unsigned char heap[HEAP_BYTES] __attribute__((section(".frontend_workspace"),aligned(16)));
#ifdef ROMFS_SHARED_WORKSPACE
static unsigned char *decode_work,*decode_output;
#else
static unsigned char decode_work[320000] __attribute__((section(".frontend_workspace"),aligned(16)));
static unsigned char decode_output[BLOCK_BYTES] __attribute__((section(".frontend_workspace"),aligned(16)));
#endif
typedef struct HeapBlock {size_t size;struct HeapBlock *next;unsigned used;} HeapBlock;
static HeapBlock *first;
static size_t heap_live,heap_peak,decode_peak;
static unsigned decodes,initialized;
static int cached_pack=-1,cached_part=-1;
static size_t cached_bytes;
static unsigned char warm_image[N25_ROMFS_WARM_CAPACITY] __attribute__((section(".frontend_workspace"),aligned(16)));
static const unsigned char*warm_files[N25_ROMFS_FILE_COUNT];
static unsigned warm_ready;
extern const unsigned char n25_rom_sha256[N25_ROMFS_FILE_COUNT][32];

int n25_romfs_set_workspace(void *buffer,size_t capacity){
#ifdef ROMFS_SHARED_WORKSPACE
    if(buffer&&(((uintptr_t)buffer&15)||capacity<320000+BLOCK_BYTES))return -1;
    decode_work=buffer;decode_output=buffer?(unsigned char*)buffer+320000:NULL;
    cached_pack=cached_part=-1;cached_bytes=0;return 0;
#else
    (void)buffer;(void)capacity;return -1;
#endif
}

typedef struct {const N25RomFile *file;size_t pos;unsigned open;} RomHandle;
static RomHandle handles[12];
struct audit_directory {char prefix[128],previous[256];unsigned cursor,open;struct dirent entry;};
static struct audit_directory directories[8];

static size_t aligned_size(size_t n){return n>SIZE_MAX-15?0:(n+15)&~(size_t)15;}
static size_t header_size(void){return aligned_size(sizeof(HeapBlock));}
int n25_romfs_initialize(void){
    if(initialized)return 0;
    first=(HeapBlock *)heap;first->size=sizeof(heap)-header_size();first->next=NULL;first->used=0;
    initialized=1;return 0;
}
void *__wrap_malloc(size_t bytes){
    n25_romfs_initialize();
    size_t n=aligned_size(bytes?bytes:1);
    if(!n){errno=ENOMEM;return NULL;}
    for(HeapBlock *b=first;b;b=b->next){
        if(b->used||b->size<n)continue;
        if(b->size-n>=header_size()+16){
            HeapBlock *next=(HeapBlock *)((unsigned char *)b+header_size()+n);
            next->size=b->size-n-header_size();next->next=b->next;next->used=0;
            b->size=n;b->next=next;
        }
        b->used=1;heap_live+=b->size+header_size();
        if(heap_live>heap_peak)heap_peak=heap_live;
        return (unsigned char *)b+header_size();
    }
    bench_metric("frontend_heap_allocation_failed_bytes",bytes);
    bench_metric("frontend_heap_live_at_failure_bytes",heap_live);
    errno=ENOMEM;return NULL;
}
void __wrap_free(void *p){
    if(!p)return;
    if((uintptr_t)p>=(uintptr_t)warm_image&&(uintptr_t)p<(uintptr_t)warm_image+sizeof(warm_image))return;
    for(unsigned i=0;i<n25_rom_file_count;i++)if(n25_rom_files[i].direct==p)return;
    HeapBlock *b=(HeapBlock *)((unsigned char *)p-header_size());
    if((unsigned char *)b<heap||(unsigned char *)b>=heap+sizeof(heap)||!b->used)abort();
    b->used=0;heap_live-=b->size+header_size();
    for(HeapBlock *at=first;at&&at->next;){
        if(!at->used&&!at->next->used){at->size+=header_size()+at->next->size;at->next=at->next->next;}
        else at=at->next;
    }
}
void *__wrap_calloc(size_t n,size_t bytes){
    if(bytes&&n>SIZE_MAX/bytes){errno=ENOMEM;return NULL;}
    void *p=__wrap_malloc(n*bytes);if(p)memset(p,0,n*bytes);return p;
}
void *__wrap_realloc(void *p,size_t bytes){
    if(!p)return __wrap_malloc(bytes);
    if(!bytes){__wrap_free(p);return NULL;}
    HeapBlock *b=(HeapBlock *)((unsigned char *)p-header_size());
    if(b->size>=bytes)return p;
    void *q=__wrap_malloc(bytes);if(!q)return NULL;
    memcpy(q,p,b->size);__wrap_free(p);return q;
}
char *__wrap_strdup(const char *s){size_t n=strlen(s)+1;char *p=__wrap_malloc(n);if(p)memcpy(p,s,n);return p;}
size_t n25_romfs_heap_peak(void){return heap_peak;}
size_t n25_romfs_decode_workspace_peak(void){return decode_peak;}
unsigned n25_romfs_decode_count(void){return decodes;}

static const char *relative_path(const char *path){
    static const char root[]="/espeak-ng-data";
    if(!path||strncmp(path,root,sizeof(root)-1))return NULL;
    path+=sizeof(root)-1;
    if(*path=='/')return path+1;
    return *path?NULL:path;
}
static const N25RomFile *file_by_name(const char *name){
    if(!name)return NULL;
    unsigned lo=0,hi=n25_rom_file_count;
    while(lo<hi){unsigned m=lo+(hi-lo)/2;if(strcmp(n25_rom_files[m].name,name)<0)lo=m+1;else hi=m;}
    return lo<n25_rom_file_count&&!strcmp(n25_rom_files[lo].name,name)?n25_rom_files+lo:NULL;
}
const void*n25_romfs_direct_file(const char*path,size_t*bytes){
    const N25RomFile*f=file_by_name(relative_path(path));
    if(!warm_ready||!f)return NULL;
    if(bytes)*bytes=f->bytes;
    return warm_files[f-n25_rom_files];
}
static int directory_exists(const char *name){
    size_t n=strlen(name);
    if(!n)return 1;
    for(unsigned i=0;i<n25_rom_file_count;i++)if(!strncmp(name,n25_rom_files[i].name,n)&&n25_rom_files[i].name[n]=='/')return 1;
    return 0;
}
int __wrap_stat(const char *path,struct stat *st){
    const char *rel=relative_path(path);if(!rel){errno=ENOENT;return -1;}
    const N25RomFile *f=file_by_name(rel);memset(st,0,sizeof(*st));
    if(f){st->st_mode=S_IFREG|0444;st->st_size=f->bytes;return 0;}
    if(directory_exists(rel)){st->st_mode=S_IFDIR|0555;return 0;}
    errno=ENOENT;return -1;
}
FILE *__wrap_fopen(const char *path,const char *mode){
    if(!mode||mode[0]!='r'){errno=EROFS;return NULL;}
    const N25RomFile *f=file_by_name(relative_path(path));
    if(!f){errno=ENOENT;return NULL;}
    for(unsigned i=0;i<sizeof(handles)/sizeof(handles[0]);i++)if(!handles[i].open){
        handles[i]=(RomHandle){f,0,1};return (FILE *)(handles+i);
    }
    errno=EMFILE;return NULL;
}
static RomHandle *handle(FILE *stream){
    for(unsigned i=0;i<sizeof(handles)/sizeof(handles[0]);i++)if(stream==(FILE *)(handles+i)&&handles[i].open)return handles+i;
    errno=EBADF;return NULL;
}
static int cache_part(unsigned pack,unsigned part){
    if(cached_pack==(int)pack&&cached_part==(int)part)return 0;
    if(pack>=FEP_PROBE_PACK_COUNT)return -1;
    FepPack p;FepSection s;char name[32];
    if(part&0x80000000u)snprintf(name,sizeof(name),"resource/%u",part&0x7fffffffu);
    else snprintf(name,sizeof(name),"part/%u",part);
    if(fep_open(&p,fep_probe_packs[pack].data,fep_probe_packs[pack].bytes)||fep_find(&p,name,&s))return -1;
    size_t peak=0;
    if(!decode_output||!decode_work||fep_decode(&s,decode_output,BLOCK_BYTES,decode_work,320000,&peak))return -1;
    if(peak>decode_peak)decode_peak=peak;
    cached_pack=pack;cached_part=part;cached_bytes=s.raw_bytes;decodes++;return 0;
}
static size_t read_bytes(void *out,size_t n,RomHandle *h){
    const N25RomFile *f=h->file;
    if(n>f->bytes-h->pos)n=f->bytes-h->pos;
    if(warm_ready){memcpy(out,warm_files[f-n25_rom_files]+h->pos,n);h->pos+=n;return n;}

    if(f->direct){memcpy(out,f->direct+h->pos,n);h->pos+=n;return n;}
    size_t written=0;
    while(written<n){
        size_t offset=f->offset+h->pos;
        unsigned part=f->part;
        if(!(part&0x80000000u)){part+=(unsigned)(offset/BLOCK_BYTES);offset%=BLOCK_BYTES;}
        if(cache_part(f->pack,part)||offset>=cached_bytes){errno=EIO;return written;}
        size_t take=cached_bytes-offset;if(take>n-written)take=n-written;
        memcpy((unsigned char *)out+written,decode_output+offset,take);written+=take;h->pos+=take;
    }
    return written;
}
int n25_romfs_prewarm_files(void){
    if(warm_ready)return 0;
    if(n25_rom_file_count!=N25_ROMFS_FILE_COUNT)return -1;
    size_t used=0;
    for(unsigned i=0;i<n25_rom_file_count;i++){
        const N25RomFile*f=n25_rom_files+i;
        if(f->direct)warm_files[i]=f->direct;
        else{
            used=(used+15)&~(size_t)15;
            if(used>sizeof(warm_image)||f->bytes>sizeof(warm_image)-used)return -1;
            warm_files[i]=warm_image+used;RomHandle handle={f,0,1};
            if(read_bytes(warm_image+used,f->bytes,&handle)!=f->bytes)return -1;
            used+=f->bytes;
        }
        unsigned char hash[32];storage_sha256(warm_files[i],f->bytes,hash);
        if(memcmp(hash,n25_rom_sha256[i],32))return -2;
    }
    warm_ready=1;
    bench_metric("all_language_G2P_files_prewarmed_SHA256",n25_rom_file_count);
    bench_metric("all_language_G2P_file_image_RAM_bytes",used);
    return 0;
}
size_t __wrap_fread(void *out,size_t size,size_t count,FILE *stream){
    RomHandle *h=handle(stream);if(!h||!size||!count)return 0;
    if(count>SIZE_MAX/size){errno=EOVERFLOW;return 0;}
    return read_bytes(out,size*count,h)/size;
}
int __wrap_fgetc(FILE *stream){RomHandle *h=handle(stream);unsigned char c;if(!h||read_bytes(&c,1,h)!=1)return EOF;return c;}
char *__wrap_fgets(char *out,int n,FILE *stream){
    if(n<=0)return NULL;
    int i=0,c=0;
    while(i+1<n&&(c=__wrap_fgetc(stream))!=EOF){out[i++]=(char)c;if(c=='\n')break;}
    if(!i&&c==EOF)return NULL;out[i]=0;return out;
}
int __wrap_fclose(FILE *stream){RomHandle *h=handle(stream);if(!h)return EOF;h->open=0;return 0;}
int __wrap_fseek(FILE *stream,long offset,int whence){
    RomHandle *h=handle(stream);if(!h)return -1;
    long long base=whence==SEEK_SET?0:whence==SEEK_CUR?h->pos:whence==SEEK_END?h->file->bytes:-1;
    long long to=base+offset;if(base<0||to<0||(unsigned long long)to>h->file->bytes){errno=EINVAL;return -1;}
    h->pos=(size_t)to;return 0;
}
long __wrap_ftell(FILE *stream){RomHandle *h=handle(stream);return h?(long)h->pos:-1;}
int __wrap_feof(FILE *stream){RomHandle *h=handle(stream);return !h||h->pos==h->file->bytes;}
void __wrap_rewind(FILE *stream){(void)__wrap_fseek(stream,0,SEEK_SET);}
int __wrap_ferror(FILE *stream){return handle(stream)?0:1;}
int __wrap_fflush(FILE *stream){(void)stream;return 0;}
char *__wrap_getenv(const char *name){(void)name;return NULL;}

DIR *__wrap_opendir(const char *path){
    const char *rel=relative_path(path);
    if(!rel||!directory_exists(rel)||strlen(rel)+2>sizeof(directories[0].prefix)){errno=ENOENT;return NULL;}
    for(unsigned i=0;i<sizeof(directories)/sizeof(directories[0]);i++)if(!directories[i].open){
        struct audit_directory *d=directories+i;memset(d,0,sizeof(*d));d->open=1;strcpy(d->prefix,rel);
        size_t n=strlen(rel);if(n){d->prefix[n]='/';d->prefix[n+1]=0;}
        return (DIR *)d;
    }
    errno=EMFILE;return NULL;
}
struct dirent *__wrap_readdir(DIR *directory){
    struct audit_directory *d=(struct audit_directory *)directory;
    if(!d||!d->open)return NULL;
    size_t n=strlen(d->prefix);
    while(d->cursor<n25_rom_file_count){
        const char *file=n25_rom_files[d->cursor++].name;
        if(strncmp(file,d->prefix,n))continue;
        file+=n;const char *slash=strchr(file,'/');size_t length=slash?(size_t)(slash-file):strlen(file);
        if(!length||length>=sizeof(d->entry.d_name))continue;
        memcpy(d->entry.d_name,file,length);d->entry.d_name[length]=0;
        if(!strcmp(d->previous,d->entry.d_name))continue;
        strcpy(d->previous,d->entry.d_name);return &d->entry;
    }
    return NULL;
}
int __wrap_closedir(DIR *directory){struct audit_directory *d=(struct audit_directory *)directory;if(!d||!d->open)return -1;d->open=0;return 0;}
