"""Independent fresh ordinary-text features from the selected CEA342e API."""
import hashlib
import json
import sys
import numpy as np
from build_resources import ROOT,WORK,write,sha
from prepare_romance_live import write_cases
TEXTS={
 'bg':['Добро утро.','Това е нова книга.','Тя говори ясно?','В града, с приятели.','Гласът е ясен.'],
 'en':['Hello, how are you?','The little garden is quiet.','This is another test.','We can hear a clear voice.',"It's a sunny day.",'Walked singing softly.'],
 'fr':['Bonjour, comment allez-vous ?','La voix est claire.',"L'enfant parle.",'Le chat dort, mais la fille chante.','Nous parlons puis nous partons.'],
 'it':['Ciao, come stai?','Il piccolo giardino è tranquillo dopo la pioggia.','Una bella casa.','La voce è chiara.','Questo è un nuovo libro.'],
 'lt':['Labas rytas.','Tai yra naujas balsas.','Gera diena, graži kalba.','Lietuva yra graži.','Mano balsas skamba.','Sveiki, draugai.','Vaikas skaito knygą.','Tai puikus testas?'],
 'pt':['Olá, como estás?','O menino fala com a menina.','A letra o é uma vogal.','Eu ouço a voz.','O teste é claro.'],
 'ro':['Salut, prieteni!','Fata citește o carte.','Vocea este clară.','Un test simplu.','O mașină nouă.','Acest algoritm funcționează.'],
 'sl':['Pozdrav, kako si?','To je nova knjiga.','Lepa beseda, nov glas.','Prst je majhen.','V mestu, s prijatelji.'],
}
def main():
    source=WORK/'new25_clarity_repair_20261030/repo/pipelines/prosody1000_rndvoc_opus24'
    sys.path.insert(0,str(source));import text_worker
    out=ROOT/'host_reference/fresh_texts';out.mkdir(exist_ok=False)
    cases=json.loads((ROOT/'generated/FINAL25_FEATURE_CASES.json').read_text())['cases'];failures=[]
    for language,texts in TEXTS.items():
        for i,text in enumerate(texts):
            directory=out/(language+'_'+str(i))
            try:
                plan=text_worker.plan(text,language,directory)
                assert len(plan['chunks'])==1
                path=plan['chunks'][0]['file'];z=np.load(path)
                arrays={k:np.asarray(z[k],dtype='<u4'if k in('phones','pause')else'<f4')for k in('phones','pause','phonetic','context')}
                cases.append(dict(language=language,text=text,count=len(z['phones']),dim=z['phonetic'].shape[1],
                    context_dim=z['context'].shape[1],reference_npz=path,
                    sha256={k:hashlib.sha256(value.tobytes()).hexdigest()for k,value in arrays.items()}))
                print(language,i,len(z['phones']),flush=True)
            except Exception as error:
                failures.append(dict(language=language,text=text,error=str(error)));print(language,i,'SOURCE_REJECTED',str(error)[:160],flush=True)
    write(ROOT/'generated/FRESH_TEXT_FEATURE_CASES.json',dict(cases=cases,source_rejected=failures,
        generator_source_sha256=sha(source/'text_worker.py'),native_fixture_use=False,
        scope='Hashes are test-only. Native firmware receives raw text and contains no sentence feature answers.'))
    write_cases(cases)
if __name__=='__main__':main()
