from pathlib import Path
import subprocess,sys,argparse
ROOT=Path(__file__).resolve().parent;WORK=Path('/home/fomak/ffb')
ASSETS=WORK/'new25_opus_bwe24_20261030/repo/new25_postprocess/assets/opus_bwe'
parser=argparse.ArgumentParser();parser.add_argument('--tag',default='v1');parser.add_argument('--mode',choices=['chain','component'],default='chain');parser.add_argument('--activation-bits',choices=['8','16'],default='16');parser.add_argument('--arena-bytes',type=int,default=6000000);args=parser.parse_args()
names=['dnn/osce.c','dnn/osce_features.c','dnn/nnet.c','dnn/parse_lpcnet_weights.c','dnn/freq.c','dnn/lpcnet_tables.c','celt/kiss_fft.c','celt/pitch.c']
sources=[ROOT/'upstream/dnn/nndsp.c',ROOT/'src/quality.c',ROOT/'src/packed_dot8.S',ROOT/'src/packed_matrix8.S',ROOT/'src/packed_dot4_pair.S',ROOT/'src/packed_matrix_pair.S',ROOT/'src/tensor.c',ROOT/'src/bwe_stage.c',ROOT/'src/syscalls.c',ROOT/'assets/model_index.c',ROOT/'assets/weights.S',ROOT/'src/storage.c',ROOT/'src/storage_sha256.c',ROOT/'third_party/puff/puff.c']+[ASSETS/'upstream'/n for n in names]+[ROOT/'bwe_fast/dnn/nnet_default.c',ROOT/'bwe_fast/assets/bbwenet_data_rowmajor.c']
if args.mode=='component':
 sources=[p for p in sources if p.name not in ['quality.c','tensor.c','model_index.c','weights.S']]+[ROOT/'src/component.c',ROOT/'assets/reference_input.S']
defines=['OPUS_BUILD','ENABLE_OSCE','ENABLE_OSCE_BWE','DISABLE_LACE','DISABLE_NOLACE','DISABLE_DEBUG_FLOAT','VAR_ARRAYS','HAVE_LRINT','HAVE_LRINTF','SKIP_CONFIG_H','SUPPRESS_PERF_WARNINGS','OPUS_BWE_SCALAR','NDEBUG','GRU_INPUT_PAIR=1','FAST_BWE_FIR=1']
defines.append('FAST_I8_ACT_SPLIT16='+str(int(args.activation_bits=='16')))
s=(ROOT/'build.py').read_text();s=s.replace("'-I',str(ROOT/'src'),","'-I',str(ROOT/'src'),"+','.join(repr('-I'+str(ASSETS/'upstream'/d))for d in ['','include','dnn','celt','silk'])+',')
s=s.replace("'-Wextra','-Werror'", "'-Wextra','-Werror','-Wno-unused-function','-Wno-unused-variable','-Wno-unused-but-set-variable','-Wno-unused-parameter'")
s=s.replace("cmd=[str(cc),*flags,'-c'","cmd=[str(cc),*flags,*(['-Wno-error']if source.is_relative_to(Path("+repr(str(ASSETS))+"))or source.name in ['bbwenet_data_rowmajor.c','nnet_default.c','nndsp.c']else []),'-c'")
script=ROOT/'build_native.py';script.write_text(s)
cmd=[sys.executable,str(script),'--output',str(ROOT/'build'/args.tag),'--arena-bytes',str(args.arena_bytes),'--stack-bytes','131072','--opt','2','--no-latest']
for p in sources:cmd+=['--source',str(p)]
for d in defines:cmd+=['--define',d]
subprocess.run(cmd,check=True)
