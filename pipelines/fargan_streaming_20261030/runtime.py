"""Current acoustic/prosody -> released FARGAN 4750 -> original Opus BWE24."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'repo'))
sys.path.insert(1, str(WORK / 'tts_stream_runtime_20261030'))
import numpy as np
import torch
from new25_fargan import FARGANVocoder
from new25_acoustic.inference import AcousticPredictor
from acoustic_cursor import AcousticCursor
from cea_lazy_condition_cursor import LazyConditionCursor
from bwe_stream import BWE24Stream
from sync_cea342e_20261030 import cea_controls
from stream_dsp import pcm16


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FarganPipeline:
    def __init__(self, seed=0, identifier_prosody="natural"):
        if identifier_prosody not in ("natural", "original"):
            raise ValueError("Unknown identifier prosody mode")
        start = time.perf_counter()
        self.identifier_prosody = identifier_prosody
        self.legacy = cea_controls
        stage = WORK / 'new25_screenreader_20261030/unified_zh_prosody_v2'
        sys.path.insert(0, str(stage))
        from network import UnifiedAcoustic
        from model import unpack
        acoustic_package = stage / 'step1000_acoustic_package'
        normalization = WORK / 'new25_quality_continue_20261029/baseline/int8/assets/normalization.json'
        vocabulary = stage / 'assets/vocabulary.json'
        values = unpack(acoustic_package)
        state = {key.removeprefix('acoustic.'): torch.from_numpy(value)
                 for key, value in values.items() if key.startswith('acoustic.')}
        assert len(state) == 136
        vocab = json.loads(vocabulary.read_text())
        model = UnifiedAcoustic(len(vocab))
        model.load_state_dict(state, strict=True)
        self.acoustic = AcousticPredictor(model, json.loads(normalization.read_text()), vocab)
        self.vocoder = FARGANVocoder.from_pretrained(ROOT / 'repo/models/fargan')
        from new25_fargan.inference_optim import freeze_weight_norm
        self.optimization = freeze_weight_norm(self.vocoder)
        assert self.vocoder.streaming is True
        self.base_encode = self.acoustic.model.encode
        self.bwe = BWE24Stream()
        self.seed = seed
        self.busy = False
        self.last_report = None
        self.first_effective_prefix = None
        self.on_pcm16 = None
        self.load_seconds = time.perf_counter() - start
        files = [acoustic_package / 'weights.safetensors', acoustic_package / 'quantization.json',
                 normalization, vocabulary, ROOT / 'repo/models/fargan/vocoder_adapter_fp16.safetensors',
                 ROOT / 'repo/models/fargan/model_config.json', ROOT / 'repo/models/fargan/EXPORT.json']
        self.identity = dict(files={str(path): sha(path) for path in files},
            acoustic_stage_updates=1000, acoustic_global_updates=3000,
            FARGAN_step=4750, acoustic_parameters=1637644,
            FARGAN_parameters=817525, adapter_parameters=1891412,
            acoustic_tensors=136, FARGAN_adapter_tensors=91,
            streaming=True, parameter_storage='original FARGAN FP16; original acoustic INT8',
            arithmetic='FP32 with original quantization-style excitation noise',
            HiFi_weights_loaded=False, RND_weights_loaded=False, training_updates=0)
        self.identity['inference_optimization'] = self.optimization
        controls_identity = ROOT / 'sync_cea342e_20261030/CONTROL_SOURCE_IDENTITY.json'
        self.identity['frontend_and_identifier_controls'] = dict(
            commit='cea342e5f3ee6d2efd228da78b2f3cf895062118',
            source=str(WORK / 'new25_clarity_repair_20261030/repo/pipelines/prosody1000_rndvoc_opus24'),
            identity_file=str(controls_identity), identity_sha256=sha(controls_identity),
            identifier_prosody_mode=self.identifier_prosody, isolated_package_imports=True,
            source_RND_or_HiFi_loader_imported=False)

    def prepare(self, features, meta, language):
        self.acoustic.model.encode = self.base_encode
        regular, u, c, digit = cea_controls.install(self.acoustic, features, meta, 'natural') \
            if language == 'zh' else ([], [], [], [])
        flow = cea_controls.install_flow(self.acoustic, features, meta) if language == 'zh' else []
        active = language == 'zh' and self.identifier_prosody != 'original' and bool(meta.get('identifier_units'))
        connection = cea_controls.install_connection(self.acoustic, features, meta) if active else []
        return dict(regular=regular, U=u, C=c, first_digit=digit, continuous_code_flow=flow,
                    identifier_prosody_mode=self.identifier_prosody,
                    connected_identifier_gaps=connection, identifier_pitch=[],
                    identifier_pitch_requires_complete_acoustic_unit=active,
                    HiFi_logmel_energy_control_applied=False,
                    energy_note='Vocoder-specific 80D log-mel offsets are not applied to 20D FARGAN features')

    @staticmethod
    def _finalize_duration_controls(controls, features, meta, durations):
        # The same evidence correction performed by cea342e Engine.acoustic_piece.
        for d in controls['first_digit']:
            start = d['phone_range'][0]; gap = []; i = start-1
            while i >= 0 and features['pause'][i]:
                gap.append(i); i -= 1
            actual = int(durations[gap].sum())*10
            if actual != d['after_gap_ms']:
                d['emphasis_stage_gap_ms'] = d['after_gap_ms']; d['after_gap_ms'] = actual
        for record in controls['U']:
            unit = next(v for v in meta['identifier_units'] if v['symbol'] == 'U' and
                        [v['start'], v['end']] == record['phone_range'])
            for direction, key in [(-1, 'preceding_gap_ms'), (1, 'following_gap_ms')]:
                neighbor = next((v for v in meta['identifier_units'] if v['code_id'] == unit['code_id'] and
                                 v['symbol_index'] == unit['symbol_index']+direction), None)
                if neighbor:
                    a, b = (neighbor['end'], unit['start']) if direction < 0 else (unit['end'], neighbor['start'])
                    actual = int(durations[a:b].sum())*10
                    if actual != record[key]:
                        record['previous_stage_'+key] = record[key]; record[key] = actual

    @torch.inference_mode()
    def _predict_corrected(self, features, meta, language, controls):
        start = time.perf_counter()
        result = self.acoustic.predict(language=language, **features)
        predict_seconds = time.perf_counter()-start
        pitch_start = time.perf_counter()
        if controls['identifier_pitch_requires_complete_acoustic_unit']:
            controls['identifier_pitch'] = cea_controls.apply_pitch(
                result, features, meta, self.acoustic.normalization[language])
        controls['complete_acoustic_prediction_host_seconds'] = predict_seconds
        controls['identifier_pitch_apply_host_seconds'] = time.perf_counter()-pitch_start
        self._finalize_duration_controls(controls, features, meta, result.durations)
        return result

    def _raw(self, values):
        clipped = int(np.count_nonzero((values > 1) | (values < -1)))
        raw = pcm16(values)
        if self.on_pcm16 is not None:
            self.on_pcm16(raw)
        return raw, clipped

    @torch.inference_mode()
    def full_unit(self, features, meta, language):
        controls = self.prepare(features, meta, language)
        result = self._predict_corrected(features, meta, language, controls)
        values = self.vocoder.synthesize_offline(result.normalized[None], language, seed=self.seed)[0].numpy()
        raw, clips = self._raw(values)
        self.bwe.reset()
        output = np.concatenate((self.bwe.push(raw), self.bwe.finish()))
        self.acoustic.model.encode = self.base_encode
        return output, dict(mode='original_offline_FARGAN_reference', physical_frames=len(result.normalized),
            phone_durations=result.durations.tolist(), PCM16_samples=len(raw), PCM24_samples=len(output),
            source_sample_start16=400, source_sample_end16=len(result.normalized)*160-240,
            clipped_PCM16_samples=clips, BWE_quantizer_clips=self.bwe.quantizer_clips,
            controls=controls, identity=self.identity, training_updates=0)

    def stream_unit(self, features, meta, language, *, acoustic_chunk=16, mel_chunk=16):
        if self.busy:
            raise RuntimeError('One active request per pipeline')
        if acoustic_chunk < 1:
            raise ValueError('Positive acoustic chunk required')
        self.busy = True
        self.last_report = None
        self.first_effective_prefix = None
        start = time.perf_counter()
        total16 = total24 = clips = yields = 0
        first_any = first_active = None
        first_progress = None
        activity = np.empty(0, np.int16)
        activity_offset = 0
        prefix = []
        try:
            controls = self.prepare(features, meta, language)
            acoustic = LazyConditionCursor(self.acoustic, features, meta, language,
                active=controls['identifier_pitch_requires_complete_acoustic_unit'],
                raw_chunk=acoustic_chunk)
            controls['identifier_pitch_requires_complete_group'] = controls['identifier_pitch_requires_complete_acoustic_unit']
            controls['identifier_pitch_requires_complete_acoustic_unit'] = False
            self._finalize_duration_controls(controls, features, meta, acoustic.durations)
            if acoustic.frames < 5:
                raise ValueError('Original FARGAN valid-center contract requires at least five physical frames')
            request = self.vocoder.begin_stream(language, seed=self.seed)
            self.bwe.reset()
            first_phone = next((i for i, pause in enumerate(features['pause'])
                                if not pause and acoustic.durations[i]), len(acoustic.durations))
            nonpause16 = max(0, int(acoustic.durations[:first_phone].sum())*160-400)
            nonpause24 = (3*nonpause16+1)//2

            def observe(value):
                nonlocal total24, first_any, first_active, first_progress, activity, activity_offset, prefix
                if not len(value):
                    return
                total24 += len(value)
                if first_any is None:
                    first_any = time.perf_counter()-start
                    acoustic.first_pcm_decoded_frames = acoustic.raw.offset
                if first_active is not None:
                    return
                prefix.append(value.copy())
                activity = np.concatenate((activity, value))
                while len(activity) >= 480:
                    window = activity[:480].astype(np.float32)/32768
                    if activity_offset >= nonpause24 and np.mean(window*window) >= .0000316227766 and np.max(np.abs(window)) >= .01:
                        first_active = time.perf_counter()-start
                        self.first_effective_prefix = np.concatenate(prefix)
                        prefix = []
                        first_progress = dict(acoustic_frames_decoded=acoustic.raw.offset,
                            acoustic_frames_emitted=acoustic.offset,
                            identifier_groups_finalized=sum(g['done'] for g in acoustic.groups),
                            total_acoustic_frames=acoustic.frames, PCM16_samples_emitted=total16,
                            active_sample_offset24=activity_offset, PCM24_samples_available=total24)
                        return
                    activity = activity[240:].copy()
                    activity_offset += 240

            with torch.inference_mode():
                while acoustic.offset < acoustic.frames:
                    physical = acoustic.pull(acoustic_chunk)
                    values, request = self.vocoder.synthesize_chunk(physical[None], language, state=request,
                        final=acoustic.offset == acoustic.frames)
                    values = values[0].numpy()
                    if not len(values):
                        continue
                    raw, count = self._raw(values)
                    clips += count
                    total16 += len(raw)
                    output = self.bwe.push(raw)
                    observe(output)
                    if len(output):
                        yields += 1
                        yield output
                output = self.bwe.finish()
                observe(output)
                if len(output):
                    yields += 1
                    yield output
            controls['identifier_pitch'] = acoustic.pitch_records
            assert total16 == (acoustic.frames-4)*160
            assert total24 == (acoustic.frames-4)*240 and request.finished
            self.last_report = dict(status='CURRENT_ACOUSTIC_FARGAN_BWE_STREAM_COMPLETE',
                language=language, physical_frames=acoustic.frames, phone_durations=acoustic.durations.tolist(),
                PCM16_samples=total16, PCM24_samples=total24, model_yield_blocks=yields,
                first_PCM24_host_seconds=first_any, first_effective_PCM24_host_seconds=first_active,
                first_active_progress=first_progress, request_host_seconds=time.perf_counter()-start,
                model_load_host_seconds=self.load_seconds, FARGAN=request.report(), controls=controls,
                acoustic_stream=acoustic.report(), inference_optimization=self.optimization,
                source_sample_start16=400, source_sample_end16=acoustic.frames*160-240,
                original_valid_center=True, tail_padding_samples=0, teacher_prefill_frames=0,
                clipped_PCM16_samples=clips, BWE_quantizer_clips=self.bwe.quantizer_clips,
                BWE_flush_calls=self.bwe.flush_calls, identity=self.identity, training_updates=0,
                host_time_is_not_M55_cycles=True, quality_human_accepted=False)
        finally:
            self.acoustic.model.encode = self.base_encode
            self.busy = False

    def close(self):
        self.acoustic.model.encode = self.base_encode
        self.bwe.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _CorrectedConditionCursor:
    """Already corrected complete acoustic unit, consumed without recomputation.

    cea342e pitch uses identifier-wide medians. Hold the complete acoustic unit
    before any PCM is emitted; FARGAN/BWE remain stateful and packet streamed.
    """
    def __init__(self, result):
        self.result = result
        self.frames, self.dimension = result.normalized.shape
        self.durations = result.durations
        self.offset = self.calls = 0

    def pull(self, count):
        if type(count) is not int or count < 1:
            raise ValueError('Positive acoustic pull required')
        end = min(self.frames, self.offset+count)
        output = self.result.normalized[self.offset:end].copy()
        self.offset = end
        self.calls += 1
        return output

    def report(self):
        return dict(physical_frames=self.frames, emitted_frames=self.offset,
                    dimension=self.dimension, decoder_calls=1, cursor_pulls=self.calls,
                    encode_calls_this_request=1, weights_reloaded=False,
                    full_acoustic_unit_materialized_before_first_PCM=True,
                    reason='cea342e identifier-wide voiced log-F0 median must be known before pitch correction',
                    condition_cache_bytes=self.result.normalized.nbytes,
                    full_text_unit_encoding_retained=True, character_input_causal=False,
                    finished=self.offset == self.frames, training_updates=0)
