"""Published Lite skeleton using pinned official modules; explicit mel-grid synthesis."""
import torch

from ._vendor.models.rndvoc_22k import RNDVocoder22k
from .frontend import aligned_istft


class RNDVoCLite(RNDVocoder22k):
    def forward(self, mel):
        assert mel.ndim == 3 and mel.shape[1] == 80 and torch.isfinite(mel).all()
        # All learned operations below match the pinned upstream forward.
        init_mag = (self.PhiT @ torch.exp(mel)).abs().clamp_min(1e-5)
        init_spec = torch.stack([init_mag, torch.zeros_like(init_mag)], dim=-1)
        null_x = self.null_enc(init_spec).transpose(1, 3).contiguous()
        x = null_x
        for block in self.null_module_list:
            x = block(x)
        null_mag, null_pha = self.null_dec((self.alpha * null_x + x).transpose(1, 3).contiguous())
        null_filter = torch.eye(self.n_fft//2+1, dtype=null_mag.dtype, device=null_mag.device) - self.PhiTPhi
        null_filtered = torch.einsum('fk,bkt->bft', null_filter, null_mag).abs().clamp_min(1e-5)
        out_mag = null_filtered + init_mag
        re, im = out_mag * torch.cos(null_pha), out_mag * torch.sin(null_pha)
        logamp = torch.log(torch.sqrt(re.square() + im.square()) + 1e-7)
        phase = torch.atan2(im, re)
        wav = aligned_istft(torch.complex(re, im), torch.hann_window(self.win_size, device=mel.device))
        return logamp, phase, re, im, wav


def parameter_report(model):
    return {'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'parameter_tensors': len(list(model.parameters())),
            'nonlearned_buffer_elements': sum(p.numel() for p in model.buffers()),
            'fp16_parameter_bytes': sum(p.numel()*2 for p in model.parameters()),
            'whole_process_ram_verified': False,
            'paper_reported_lite_parameters': 710000,
            'official_lite_checkpoint_or_config_found': False}
