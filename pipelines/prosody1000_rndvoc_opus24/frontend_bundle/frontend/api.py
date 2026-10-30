"""25-language text-to-features frontend; no acoustic weights/audio/templates.

Resources and frozen per-language rules are vendored from the existing FEP
release. The old network is never constructed or loaded. PT/IT use the original
phoneme inventory expected by the NEW25 training annotations.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import atexit
import hashlib
import importlib
import importlib.util
import json
import shutil
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
VENDOR=HERE/'vendor'
_SHARED={}


def read(p):return json.loads(Path(p).read_text())


def load_adapter(path,expected):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen frontend adapter changed: '+str(path))
    spec=importlib.util.spec_from_file_location('_new25_frontend_'+expected[:16],path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class TextFrontend:
    def __init__(self,root=None,model_dir=None):
        self.root=Path(root or VENDOR).resolve()
        self.model_dir=Path(model_dir or HERE.parent/'int8/assets').resolve()
        key=(str(self.root),str(self.model_dir))
        if key in _SHARED:
            self.__dict__=_SHARED[key].__dict__
            return
        sys.path.insert(0,str(self.root/'runtime/current25'))
        sys.path.insert(0,str(self.root))
        from frontend_resources import Resources
        self.index=read(self.root/'FRONTEND_INDEX.json')
        self.entries={e['short']:e for e in self.index['models']}
        self.layout=read(self.root/'FEATURE_LAYOUT.json')
        self.trainer=SimpleNamespace(load_adapter=load_adapter)
        self.resources=Resources(self)
        atexit.register(shutil.rmtree,self.resources.path,ignore_errors=True)
        self.sessions={}
        self.vocabulary=read(self.model_dir/'vocabulary.json')
        self.normalization=read(self.model_dir/'normalization.json')
        self._bootstrap_no_tr()
        _SHARED[key]=self

    def _bootstrap_no_tr(self):
        # Apply the original no/tr extension to an isolated frozen DE module.
        # No runtime.loader/old model bootstrap or generated sentence fixture.
        de=self.resources.module('de')
        dedicated=de.frozen_module('de','eu_four')
        de.get_frontend('de',self.resources.library,self.resources.data)
        ext=importlib.import_module('new_frontend')
        original_import=ext.importlib.import_module
        fake_loader=SimpleNamespace(ModelRuntime=lambda *a,**kw:SimpleNamespace(
            _system=lambda locale:(None,None,None,SimpleNamespace(encode=lambda *a,**kw:None))))
        def selected(name,*args,**kwargs):
            if name=='runtime.loader':return fake_loader
            if name=='_tts25_frozen_de.frontend.eu_four':return dedicated
            return original_import(name,*args,**kwargs)
        # Python's explicit from-import of runtime.loader needs an importable
        # package, so run the exact extension body with only its bootstrap import
        # removed and a frontend-only ModelRuntime in its global namespace.
        import inspect
        if not hasattr(ext,'_new25_original_module'):
            ext._new25_original_module=ext.module
        body=inspect.getsource(ext._new25_original_module.__wrapped__)
        body=body.replace('from runtime.loader import ModelRuntime','ModelRuntime = _frontend_only_ModelRuntime')
        namespace=dict(vars(ext));namespace['_frontend_only_ModelRuntime']=fake_loader.ModelRuntime
        namespace['importlib']=SimpleNamespace(import_module=selected)
        exec(compile(body,'<frozen-no-tr-frontend-extension>','exec'),namespace)
        module=namespace['module']()
        ext.module=lambda:module

    def _session(self,language):
        if language not in self.entries:raise ValueError('Unsupported language: '+language)
        if language in self.sessions:return self.sessions[language]
        entry=self.entries[language];bundle=self.root/entry['bundle']
        spec=copy.deepcopy(self.layout['systems'][language]['spec'])
        metadata=read(bundle/'frontend_metadata.json')
        for key in ['adapter_path','lexicon_path']:
            if metadata.get(key):metadata[key]=str((bundle/metadata[key]).resolve())
        if language in ['pt','it']:
            metadata,adapter=self.resources.base_adapter(language,spec,read(bundle/'source_frontend_metadata.json'))
        else:
            metadata,adapter=self.resources.system(entry,spec,metadata)
        self.sessions[language]=(metadata,adapter)
        return metadata,adapter

    def _system(self,locale):
        language=next(e['short']for e in self.index['models']if e['language']==locale)
        metadata,adapter=self._session(language)
        return(None,self.layout['systems'][language]['spec'],metadata,adapter,None,None)

    def encode(self,text,language):
        if not isinstance(text,str)or not text.strip():raise ValueError('Nonempty ordinary text required')
        metadata,adapter=self._session(language)
        encoded=adapter.encode(language,text,metadata)
        phones=[str(p)for p in encoded.get('ipa',encoded.get('phones'))]
        norm=self.normalization[language]
        phonetic=np.asarray(encoded['phonetic_features'],dtype=np.float32)
        context=np.asarray(encoded['context_features'],dtype=np.float32)
        if phonetic.shape!=(len(phones),len(norm['phonetic_mean']))or context.shape!=(len(phones),len(norm['context_mean'])):
            raise ValueError('Frontend feature dimensions disagree with NEW25 normalization')
        if not np.isfinite(phonetic).all()or not np.isfinite(context).all():raise ValueError('Nonfinite frontend output')
        features=dict(phones=np.array([self.vocabulary.get(p,1)for p in phones],dtype=np.int64),
            phonetic=phonetic,context=context,pause=np.array([p.lower()in ['sil','sp','spn','<sil>']for p in phones],dtype=np.float32))
        return features,dict(language=language,text=text,ipa=phones,unknown_phone_count=int((features['phones']==1).sum()),
            ordinary_text_only=True,reference_audio_or_durations_used=False,resource_package=self.resources.manifest['source_candidate_id'])

    def close(self):
        # eSpeak and frozen adapters cache one process-wide library/data root.
        # Keep that directory alive for subsequent instances; atexit removes it.
        pass

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
