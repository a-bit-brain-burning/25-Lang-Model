"""Synthesize a complete utterance from cached compatible frontend features."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from runtime import load


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--language',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    a,v=load(args.package)
    sys.path.insert(0,str(args.package.resolve()/'runtime'))
    from new25_hifigan.inference import write_pcm16
    with np.load(args.features,allow_pickle=False)as z:
        features={k:z[k].copy()for k in ['phones','phonetic','context','pause']}
    result=a.predict(language=args.language,**features)
    wave=v.synthesize(result.normalized[None],args.language)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_pcm16(args.output,wave)
    print(json.dumps(dict(output=str(args.output),frames=len(result.normalized),seconds=len(result.normalized)*.01,
                          reference_durations_used=False,host_FP32_reconstruction=True)))


if __name__=='__main__':main()
