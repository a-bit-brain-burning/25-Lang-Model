#include <stddef.h>
#include <zlib.h>
int native_deflate_block(const unsigned char*src,size_t bytes,unsigned char*out,size_t cap){
 z_stream stream={0};stream.next_in=(Bytef*)src;stream.avail_in=bytes;
 stream.next_out=out;stream.avail_out=cap;
 if(inflateInit2(&stream,-MAX_WBITS)!=Z_OK)return -1;
 int rc=inflate(&stream,Z_FINISH);
 int ok=rc==Z_STREAM_END&&stream.total_in==bytes&&stream.total_out==cap;
 inflateEnd(&stream);return ok?0:-1;
}
