/* All frontend resource access goes through romfs.c. Fail unimplemented I/O. */
#include <errno.h>
#include <sys/time.h>
int _open(const char *path,int flags,...){(void)path;(void)flags;errno=EROFS;return -1;}
int _unlink(const char *path){(void)path;errno=EROFS;return -1;}
int _gettimeofday(struct timeval *time,void *zone){(void)time;(void)zone;errno=ENOSYS;return -1;}
