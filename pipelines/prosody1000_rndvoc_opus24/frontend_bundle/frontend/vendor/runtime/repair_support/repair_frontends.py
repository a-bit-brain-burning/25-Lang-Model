"""Text-only candidates, with explicit whole-utterance fallback on unsupported G2P."""
import copy,hashlib,importlib.util
from common import HERE,BASE,LOCALES,sha
class Frontend:
 def __init__(self,run,language):
  self.language=language;self.system=run._system(LOCALES[language]);self.impl=None
  if language=='it':
   from it_frontend import CandidateFrontend
   self.impl=CandidateFrontend(run);binding=sha(HERE/'it_frontend.py')
  elif language=='pt':
   source=BASE/'original_model_diagnosis_20261023/cross_language/corrected_frontend_v2.py'
   spec=importlib.util.spec_from_file_location('_grouped11_pt_candidate',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   self.impl=module.CandidateFrontend(run);binding=sha(source)
  elif language=='bg':
   from bg_frontend import Frontend as BG
   self.impl=BG(run);binding=self.impl.signature
  else:binding=self.system[2]['frontend_signature']
  self.signature=hashlib.sha256((binding+sha(__file__)).encode()).hexdigest()
 def encode(self,language,text,metadata=None):
  assert language==self.language
  fallback=None
  if self.impl is None:f=self.system[3].encode(language,text,self.system[2])
  else:
   try:f=self.impl.encode(language,text)
   except ValueError as error:
    fallback=str(error);f=self.system[3].encode(language,text,self.system[2])
  f=copy.deepcopy(f);f['metadata'].update(frontend_signature=self.signature,automatic_not_gold=True,
   profile='grouped11_native_MFA_repair',unsupported_candidate_fallback_to_original=fallback)
  return f

