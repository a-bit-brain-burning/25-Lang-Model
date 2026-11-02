/* Compile-only ABI for the ROM size probe. No filesystem is implemented. */
#ifndef ROM_PROBE_DIRENT_H
#define ROM_PROBE_DIRENT_H
typedef struct audit_directory DIR;
struct dirent { char d_name[256]; };
DIR *opendir(const char *);
struct dirent *readdir(DIR *);
int closedir(DIR *);
#endif
