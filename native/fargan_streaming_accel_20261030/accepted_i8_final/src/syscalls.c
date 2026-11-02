#include "bench.h"
#include <sys/stat.h>
#include <errno.h>
#include <stddef.h>
void *_sbrk(ptrdiff_t increment){
 if(increment<0){errno=ENOMEM;return(void*)-1;}
 if(!increment){extern unsigned char __arena_start[];return __arena_start+bench_arena_mark();}
 void*p=bench_alloc((size_t)increment,8);if(!p){errno=ENOMEM;return(void*)-1;}return p;
}
int _write(int fd,const void*buf,size_t count){(void)fd;(void)buf;return(int)count;}
int _read(int fd,void*buf,size_t count){(void)fd;(void)buf;(void)count;return 0;}
int _close(int fd){(void)fd;return -1;}
int _fstat(int fd,struct stat*st){(void)fd;st->st_mode=S_IFCHR;return 0;}
int _isatty(int fd){(void)fd;return 1;}
int _lseek(int fd,int offset,int whence){(void)fd;(void)offset;(void)whence;return 0;}
int _getpid(void){return 1;}
int _kill(int pid,int signal){(void)pid;(void)signal;errno=EINVAL;return -1;}
_Noreturn void _exit(int status){bench_exit((unsigned)status);}
