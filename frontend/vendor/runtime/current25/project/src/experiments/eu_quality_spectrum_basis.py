"""Versioned linear/log-warped spectral math for independent numeric experiments.

Shared by extraction, physical spectral loss and native basis construction.
This does not register or migrate any production feature/checkpoint contract.
"""
import hashlib
import math

import numpy as np
from scipy.fft import dct
import torch
from torch import nn

VERSION = 'experimental_logwarp_dct_v1'
BINS = 513
SAMPLE_RATE = 16000


def validate(dim, knee):
    if dim not in (24, 32):
        raise ValueError('Independent DSP supports spectral dimensions24/32')
    if knee is not None and (not np.isfinite(knee) or knee <= 0):
        raise ValueError('Warp knee must be finite positive or None for linear')


def shape_basis(dim=32, knee=700.):
    """(dim-1,513) log-power shape; coefficients are DCT/sqrt(513)."""
    validate(dim, knee)
    hz = np.linspace(0, SAMPLE_RATE/2, BINS)
    location = np.arange(BINS, dtype=np.float64) if knee is None else (BINS-1)*np.log1p(hz/knee)/np.log1p((SAMPLE_RATE/2)/knee)
    return np.sqrt(2)*np.cos(np.pi/BINS*np.arange(1,dim)[:,None]*(location[None]+.5))


def reconstruction_basis(dim=32, knee=700.):
    """Explicit name for the physical log-power loss's shape matrix."""
    return shape_basis(dim,knee)


def encode_shape(spectrum, dim=32, knee=700.):
    """True WORLD linear power513 -> shape; no energy/DC channel here."""
    validate(dim, knee)
    spectrum=np.asarray(spectrum,dtype=np.float64)
    if spectrum.ndim!=2 or spectrum.shape[1]!=BINS or not len(spectrum) or not np.isfinite(spectrum).all() or (spectrum<0).any():
        raise ValueError('Expected nonnegative finite WORLD513 spectrum')
    logarithm=np.log(np.maximum(spectrum,1e-12))
    if knee is not None:
        hz=np.linspace(0,SAMPLE_RATE/2,BINS)
        warped_hz=knee*np.expm1(np.linspace(0,np.log1p((SAMPLE_RATE/2)/knee),BINS))
        logarithm=np.array([np.interp(warped_hz,hz,row) for row in logarithm])
    return dct(logarithm,type=2,norm='ortho',axis=1)[:,1:dim]/math.sqrt(BINS)


def native_basis(dim=32, knee=700., taps=128):
    if taps not in (64,128,256):
        raise ValueError('Independent diagnostic taps64/128/256 required')
    cepstrum=np.fft.irfft(.5*shape_basis(dim,knee),n=1024,axis=1)
    cepstrum[:,1:512]*=2
    return np.ascontiguousarray(cepstrum[:,:taps].T,dtype=np.float32)


def spec(dim=32, knee=700., taps=128):
    basis=native_basis(dim,knee,taps)
    return {'version':VERSION,'spectral_dim':dim,'shape_channels':dim-1,'warp_knee_Hz':knee,
            'sample_rate':SAMPLE_RATE,'fft_size':1024,'bins':BINS,'taps':taps,
            'encoding':'DCT-II ortho(log power on uniform warped axis)[1:dim]/sqrt513',
            'DC':'waveform log mean-square energy replaces discarded spectral DC',
            'normalization':'trapezoid frequency power mean513 endpoints half, denominator512',
            'native_basis_float32_sha256':hashlib.sha256(basis.tobytes()).hexdigest()}


class PhysicalSpectrum(nn.Module):
    """Energy+shape -> normalized physical log power using matching feature axis.

    Input last dimension is dim: waveform log-energy followed by dim-1 shape.
    Native source/filter generation still requires corresponding native_basis.
    """
    def __init__(self, dim=32, knee=700.):
        super().__init__()
        self.dim,self.knee=dim,knee
        self.register_buffer('basis',torch.from_numpy(shape_basis(dim,knee).astype(np.float32)))
        weights=torch.ones(BINS);weights[[0,-1]]=.5
        self.register_buffer('log_weights',weights.log())
        self.logsum=math.log(BINS-1)

    def forward(self,spectral):
        if spectral.shape[-1]!=self.dim:
            raise ValueError('Wrong spectral dimension for fixed basis')
        shape=(spectral[...,1:]@self.basis).clamp(-80,80)
        normalization=torch.logsumexp(shape+self.log_weights,-1,keepdim=True)-self.logsum
        return spectral[...,:1]+shape-normalization
