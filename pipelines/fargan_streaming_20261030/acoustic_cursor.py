"""One text-unit encode followed by persistent, demand-driven acoustic decode.

The unchanged encoder and learned prosody still consume a complete text unit.
Only audio-frame decoding is streamed. No weights, operators or normalization
are replaced, and no padding is added to the generated acoustic timeline.
"""
from __future__ import annotations

import numpy as np
import torch


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


class AcousticCursor:
    """Reuse one predictor's weights across independently owned requests.

    ``pull(count)`` returns at most ``count`` new normalized waveform-condition
    frames as CPU float32, while keeping the original GRU and causal-convolution
    states. Frame ownership/position arrays are built only for that pull.

    ``cancel()`` releases request tensors; ``reset()`` can start it again using
    the same model and stored phone features. A reset performs one new complete
    encode/prosody plan. ``close()`` permanently closes this cursor.
    """

    @torch.inference_mode()
    def __init__(self, predictor, features, language):
        self.predictor = predictor
        self.device = torch.device(predictor.device)
        self.total_encode_calls = 0
        self.generation = 0
        self.closed = False
        self.cancelled = False
        self._features = None
        self.encoded = self.prosody = self.state = None
        self.reset(features, language)

    @torch.inference_mode()
    def reset(self, features=None, language=None):
        from new25_acoustic.inference import LANGUAGES, MAX_FRAMES, predict_durations

        if self.closed:
            raise RuntimeError('Acoustic cursor is closed; create a new cursor')
        if features is None:
            if self._features is None:
                raise ValueError('Phone features are required for the first request')
            features = self._features
        language = self.language if language is None else language
        if language not in LANGUAGES:
            raise ValueError('Unsupported acoustic language')
        phones = np.asarray(features['phones'])
        if (phones.ndim != 1 or not 1 <= len(phones) <= MAX_FRAMES
                or phones.dtype.kind not in 'iu' or np.any(phones <= 0)
                or np.any(phones >= len(self.predictor.vocabulary))):
            raise ValueError('Expected nonpadding phone IDs from the current vocabulary')
        phones = phones.astype(np.int64, copy=True)
        norm = self.predictor.normalization[language]
        inputs = []
        stored = {'phones': phones}
        for name in ('phonetic', 'context'):
            value = np.asarray(features[name], dtype=np.float32)
            mean = np.asarray(norm[name+'_mean'], dtype=np.float32)
            std = np.asarray(norm[name+'_std'], dtype=np.float32)
            if value.shape != (len(phones), len(mean)) or not np.isfinite(value).all():
                raise ValueError('Invalid acoustic '+name+' features')
            value = value.copy()
            stored[name] = value
            prepared = (value-mean)/std
            if not np.isfinite(prepared).all():
                raise ValueError('Nonfinite normalized acoustic '+name+' features')
            inputs.append(torch.from_numpy(prepared)[None].to(self.device))
        pause = np.asarray(features['pause'])
        if pause.shape != phones.shape or not np.all((pause == 0) | (pause == 1)):
            raise ValueError('Expected the original binary pause array')
        stored['pause'] = pause.astype(np.float32, copy=True)
        lang = torch.tensor([LANGUAGES.index(language)], dtype=torch.long, device=self.device)
        with torch.autocast(self.device.type, enabled=False):
            encoded, prosody = self.predictor.model.encode(
                torch.from_numpy(phones)[None].to(self.device), *inputs, lang,
                torch.ones((1, len(phones)), device=self.device))
            durations = predict_durations(prosody, torch.from_numpy(stored['pause'])[None].to(self.device))[0]
        if not torch.isfinite(encoded).all() or not torch.isfinite(prosody).all():
            raise ValueError('Nonfinite acoustic encoder output')
        durations = durations.cpu().numpy().astype(np.int64, copy=True)
        frames = int(durations.sum())
        if frames < 1 or frames > MAX_FRAMES or np.any(durations < 0):
            raise ValueError('Predicted acoustic timeline is outside the supported range')
        # Retain phone-level plans, not O(total audio frames) layout arrays.
        self._features = stored
        self.language, self.norm, self.lang = language, norm, lang
        self.encoded, self.prosody = encoded, prosody
        self.durations, self.frames = durations, frames
        self._ends = np.cumsum(durations, dtype=np.int64)
        self._starts = self._ends-durations
        self._inverse = np.asarray([1/int(n) if n else 0. for n in durations], dtype=np.float32)
        self.dimension = len(norm['mean'])
        self.offset = self.calls = self.max_layout_frames = 0
        self.state = None
        self.cancelled = False
        self.generation += 1
        self.total_encode_calls += 1
        return self

    def _layout(self, begin, end):
        absolute = np.arange(begin, end, dtype=np.int64)
        owners = np.searchsorted(self._ends, absolute, side='right').astype(np.int64)
        relative = absolute-self._starts[owners]
        duration = self.durations[owners]
        position = np.empty((end-begin, 4), dtype=np.float32)
        position[:, 0] = relative.astype(np.float32)/np.maximum(duration-1, 1).astype(np.float32)
        position[:, 1] = self._inverse[owners]
        position[:, 2] = relative == 0
        position[:, 3] = relative+1 == duration
        self.max_layout_frames = max(self.max_layout_frames, end-begin)
        return torch.from_numpy(owners)[None].to(self.device), torch.from_numpy(position)[None].to(self.device)

    @torch.inference_mode()
    def pull(self, count):
        from new25_acoustic.inference import denormalize, waveform_condition

        if self.closed or self.cancelled:
            raise RuntimeError('Acoustic request is closed or cancelled; reset or create a new request')
        if type(count) is not int or count < 1:
            raise ValueError('Acoustic pull count must be a positive integer')
        if self.offset == self.frames:
            return np.empty((0, self.dimension), dtype=np.float32)
        end = min(self.frames, self.offset+count)
        owners, position = self._layout(self.offset, end)
        with torch.autocast(self.device.type, enabled=False):
            prediction, state = self.predictor.model.decode(
                self.encoded, self.prosody, owners, position, self.lang, self.dimension, self.state)
            result = waveform_condition(denormalize(prediction, self.norm), self.norm)
        if (not torch.isfinite(result).all() or not torch.isfinite(state['gru']).all()
                or any(not torch.isfinite(item).all() for item in state['conv'])):
            raise ValueError('Nonfinite acoustic decoder output/state')
        # LocalBlock returns views into concatenated history+input. Own compact
        # copies so a request never retains the backing storage of an old chunk.
        self.state = {'gru': state['gru'].detach().clone(),
                      'conv': [item.detach().clone() for item in state['conv']]}
        output = result[0].detach().cpu().numpy().copy()
        self.offset = end
        self.calls += 1
        return output

    @property
    def finished(self):
        return not self.closed and not self.cancelled and self.offset == self.frames

    def cancel(self):
        if not self.closed:
            self.cancelled = True
            self.encoded = self.prosody = self.state = None

    def close(self):
        self.cancel()
        self.closed = True

    def report(self):
        return dict(language=self.language, physical_frames=self.frames, emitted_frames=self.offset,
                    dimension=self.dimension, decoder_calls=self.calls, encode_calls_this_request=1,
                    total_encode_calls=self.total_encode_calls, generation=self.generation,
                    decoder_state_bytes=_tensor_bytes(self.state),
                    encoded_prosody_bytes=_tensor_bytes((self.encoded, self.prosody)),
                    decoder_state_shapes=None if self.state is None else
                        dict(gru=list(self.state['gru'].shape), conv=[list(x.shape) for x in self.state['conv']]),
                    decoder_state_owns_storage=True, peak_layout_frames=self.max_layout_frames,
                    full_audio_layout_cached=False, cancelled=self.cancelled, closed=self.closed,
                    finished=self.finished, weights_reloaded=False, training_updates=0,
                    full_text_unit_encoding_retained=True, character_input_causal=False)
