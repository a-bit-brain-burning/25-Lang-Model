"""Offline ordinary text -> 25-language frontend -> current INT8 -> WAV."""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language',required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text')
    group.add_argument('--text-file',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--save-features',type=Path)
    args=parser.parse_args()
    from frontend import TextFrontend
    start=time.perf_counter()
    with TextFrontend()as frontend:
        features,metadata=frontend.encode(args.text if args.text is not None else args.text_file.read_text(encoding='utf-8'),args.language)
    frontend_seconds=time.perf_counter()-start
    if args.save_features is not None:
        args.save_features.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.save_features,**features)
    sys.path.insert(0,str(ROOT/'int8'))
    from runtime import load
    from new25_hifigan.inference import write_pcm16
    acoustic,vocoder=load(ROOT/'int8')
    result=acoustic.predict(language=args.language,**features)
    audio=vocoder.synthesize(result.normalized[None],args.language)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_pcm16(args.output,audio)
    metadata.update(samples=audio.shape[-1],sample_rate=16000,predicted_duration_frames=int(result.durations.sum()),
        frontend_seconds=frontend_seconds,host_total_seconds=time.perf_counter()-start,
        current_model='all_int8',execution='host FP32 reference from INT8 storage',
        reference_audio_or_durations_used=False)
    args.output.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
