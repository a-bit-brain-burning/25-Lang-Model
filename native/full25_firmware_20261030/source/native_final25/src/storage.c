#include "storage.h"
#include "fep_runtime.h"
#include "bench.h"
#include "../assets/model_index.h"
#include "../assets/storage_identity.h"
#include "../third_party/puff/puff.h"
#include <string.h>

extern const unsigned char storage_model_payload[],storage_model_index[];
extern const unsigned char storage_metadata_payload[],storage_metadata_index[];
#ifndef STORAGE_INCLUDE_METADATA_PROBE
#define STORAGE_INCLUDE_METADATA_PROBE 1
#endif
const unsigned char*n25_blob;
uint64_t storage_boot_ticks;
static void fail(const char*s){bench_puts(s);bench_puts("\n");bench_exit(71);}
static uint32_t u32(const unsigned char*p){return(uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static void decode(unsigned char*out,unsigned raw,const unsigned char*in,unsigned packed){
 unsigned long source=packed,destination=raw;
 int rc=puff(out,&destination,in,&source);
 if(rc||source!=packed||destination!=raw)fail("COMPRESSED_ASSET_DEFLATE_ERROR");
}
void storage_initialize(uint64_t boot){
 storage_boot_ticks=boot;if(n25_blob)fail("COMPRESSED_STORAGE_INIT_TWICE");
 /* Independent metadata reader probe. Release its entire temporary before
  * model placement; no production frontend functionality is implied. */
 uint64_t start=bench_now();
#if STORAGE_INCLUDE_METADATA_PROBE
 size_t initial=bench_arena_mark();
 if(memcmp(storage_metadata_index,"MET1",4)||u32(storage_metadata_index+4)!=STORAGE_METADATA_FILES||u32(storage_metadata_index+8)!=STORAGE_METADATA_RAW_BYTES)fail("METADATA_INDEX_HEADER");
 unsigned names_bytes=u32(storage_metadata_index+12);
 if(STORAGE_METADATA_INDEX_BYTES!=16+STORAGE_METADATA_FILES*16+names_bytes)fail("METADATA_INDEX_EXTENT");
 unsigned char*metadata=bench_alloc(STORAGE_METADATA_RAW_BYTES,16);if(!metadata)fail("METADATA_ALLOC");
 decode(metadata,STORAGE_METADATA_RAW_BYTES,storage_metadata_payload,STORAGE_METADATA_PACKED_BYTES);
 bench_metric("storage_metadata_decode_host_ticks",bench_now()-start);
 bench_metric("storage_metadata_temporary_arena_bytes",bench_arena_mark()-initial);
 start=bench_now();unsigned char hashes[STORAGE_METADATA_FILES][32];unsigned previous=0;
 for(unsigned i=0;i<STORAGE_METADATA_FILES;i++){
  const unsigned char*record=storage_metadata_index+16+i*16;
  unsigned offset=u32(record),length=u32(record+4),name_offset=u32(record+8),name_length=u32(record+12);
  if(offset!=previous||offset>STORAGE_METADATA_RAW_BYTES||length>STORAGE_METADATA_RAW_BYTES-offset||name_offset>names_bytes||name_length>names_bytes-name_offset)fail("METADATA_FILE_EXTENT");
  storage_sha256(metadata+offset,length,hashes[i]);if(memcmp(hashes[i],storage_metadata_sha256[i],32))fail("METADATA_FILE_SHA256");previous=offset+length;
 }
 if(previous!=STORAGE_METADATA_RAW_BYTES)fail("METADATA_INCOMPLETE");
 bench_metric("storage_metadata_sha256_host_ticks",bench_now()-start);
 start=bench_now();if(bench_write_file("metadata.sha256.bin",hashes,sizeof(hashes)))fail("METADATA_SHA_WRITE");
 if(bench_write_file("metadata.original.bin",metadata,STORAGE_METADATA_RAW_BYTES))fail("METADATA_RAW_WRITE");
 bench_metric("storage_metadata_probe_export_host_ticks",bench_now()-start);
 if(bench_arena_rewind(initial))fail("METADATA_REWIND");
 bench_metric("storage_metadata_files_SHA256_passed",STORAGE_METADATA_FILES);
 bench_metric("storage_metadata_live_bytes_after_probe",bench_arena_mark()-initial);
#else
 /* This archive was an independent identity/export probe. Model descriptors,
  * normalization constants and vocabulary are separately compiled inputs;
  * no inference operation ever reads these decompressed JSON documents. */
 bench_metric("storage_metadata_archive_probe_in_firmware",0);
#endif

 start=bench_now();
 unsigned char*model=bench_alloc(MODEL_BYTES,16);if(!model)fail("MODEL_DECODE_ALLOC");
 size_t mark=bench_arena_mark(),peak=0;void*workspace=bench_alloc(320000,16);
 if(!workspace)fail("MODEL_LZMA_WORKSPACE_ALLOC");
 FepSection section={.codec=1,.packed=storage_model_payload,.packed_bytes=2292658,
   .raw_bytes=MODEL_BYTES,.crc32=4182391410u,.window=262144};
 if(fep_decode(&section,model,MODEL_BYTES,workspace,320000,&peak))fail("MODEL_LZMA_DECODE");
 if(bench_arena_rewind(mark))fail("MODEL_LZMA_REWIND");
 bench_metric("storage_model_decode_host_ticks",bench_now()-start);
 bench_metric("storage_model_LZMA2_workspace_peak",peak);
 bench_metric("storage_model_LZMA2_bytes",section.packed_bytes);
 start=bench_now();unsigned char hash[32];storage_sha256(model,MODEL_BYTES,hash);
 if(memcmp(hash,storage_model_sha256,32))fail("MODEL_SHA256");
 bench_metric("storage_model_sha256_host_ticks",bench_now()-start);
 if(bench_write_file("model.sha256.bin",hash,sizeof(hash)))fail("MODEL_SHA_WRITE");
 unsigned char rom_hash[32];storage_sha256(storage_tensor_rom_bank,MODEL_ROM_BANK_BYTES,rom_hash);
 if(memcmp(rom_hash,storage_rom_bank_sha256,32))fail("MODEL_ROM_BANK_SHA256");
 bench_metric("storage_exact_weight_ROM_bank_bytes",MODEL_ROM_BANK_BYTES);
 n25_blob=model;
 bench_metric("storage_model_SHA256_passed",1);bench_metric("storage_model_persistent_RAM_bytes",MODEL_BYTES);
 bench_metric("storage_live_arena_before_inference",bench_arena_mark());
 bench_metric("storage_peak_arena_before_inference",bench_arena_peak());
 bench_metric("storage_initialization_total_host_ticks",bench_now()-boot);
 bench_metric("storage_request_time_decode_calls",0);
}
