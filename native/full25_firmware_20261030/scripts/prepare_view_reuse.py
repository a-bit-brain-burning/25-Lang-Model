"""Reuse model-lifetime float views that already exist; no additional cache."""
import shutil
from build_resources import ROOT,write,sha

def main():
    source=ROOT/'native_bwe_packed';dest=ROOT/'native_view_reuse';dest.mkdir(exist_ok=False)
    for name in ('src','assets','frontend_patched'):shutil.copytree(source/name,dest/name)
    for name in ('bin','toolchain','frontend_source','bwe_fast','upstream','third_party'):
        (dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for p in (dest/'assets').glob('*.S'):
        p.write_text(p.read_text().replace(str(source),str(dest)))
    p=dest/'src/quality.c';before=sha(p);s=p.read_text()
    old='''unsigned co=w->dims[0],ci=w->dims[1];size_t mark=bench_arena_mark();float*bias=alloc(co);
  for(unsigned o=0;o<co;o++)bias[o]=value(b,o);'''
    new='''unsigned co=w->dims[0],ci=w->dims[1];size_t mark=bench_arena_mark();
  const float*bias=tensor_view(b);
  if(!bias){float*temporary=alloc(co);for(unsigned o=0;o<co;o++)temporary[o]=value(b,o);bias=temporary;}'''
    assert old in s;s=s.replace(old,new,1)
    old='''unsigned co=w->dims[0],ci=w->dims[1];size_t mark=bench_arena_mark();float*weights=alloc(w->elements);n25_unpack(n25_blob,w,weights);
 float*bias=alloc(co);for(unsigned o=0;o<co;o++)bias[o]=value(b,o);'''
    new='''unsigned co=w->dims[0],ci=w->dims[1];size_t mark=bench_arena_mark();
 const float*weights=tensor_view(w),*bias=tensor_view(b);
 if(!weights){float*temporary=alloc(w->elements);n25_unpack(n25_blob,w,temporary);weights=temporary;}
 if(!bias){float*temporary=alloc(co);for(unsigned o=0;o<co;o++)temporary[o]=value(b,o);bias=temporary;}'''
    assert old in s;s=s.replace(old,new,1);p.write_text(s)
    write(ROOT/'VIEW_REUSE_IDENTITY.json',dict(before_sha256=before,after_sha256=sha(p),
        added_persistent_allocations=0,weight_arithmetic_unchanged=True,
        policy='Reuse existing immutable tensor_view buffers; same old temporary decode when a view is absent.'))

if __name__=='__main__':main()
