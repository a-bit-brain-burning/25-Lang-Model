"""Original acoustic objective, with FP32 loss reductions."""
import torch
from torch.nn import functional as F

def weighted(x,w):
    return (x*w).sum()/w.expand_as(x).sum().clamp_min(1)

def acoustic_loss(pred,batch,phone_prediction):
    pred=pred.float();phone_prediction=phone_prediction.float()
    target=batch['target'];d=target.shape[-1]
    mask=batch['frame_mask'][:,:,None].float()
    w=mask.expand_as(pred).clone();w[:,:,0]*=batch['f0_valid'];w[:,:,-1]=0
    base=weighted(F.smooth_l1_loss(pred.float(),target,reduction='none'),w)
    dw=torch.minimum(w[:,1:],w[:,:-1])
    delta=weighted((pred[:,1:]-pred[:,:-1]-target[:,1:]+target[:,:-1]).abs(),dw)
    # Dynamic variance matching discourages the measured old oversmoothing;
    # applied to spectral shape, not unreliable/unvoiced F0 or VUV logits.
    n=mask.sum(1).clamp_min(1)
    def variance(z):
        m=(z*mask).sum(1)/n
        return (((z-m[:,None])**2)*mask).sum(1)/n
    gv=(torch.sqrt(variance(pred[:,:,2:d-6]).float()+1e-4)-
        torch.sqrt(variance(target[:,:,2:d-6])+1e-4)).abs().mean()
    voice=weighted(F.binary_cross_entropy_with_logits(pred[:,:,-1].float(),target[:,:,-1],reduction='none'),
                   batch['voice_weight']*batch['frame_mask'])
    pause=batch['pause'];pm=batch['phone_valid']*batch['phone_mask']
    duration=torch.log1p(batch['durations'].float())
    durpred=F.softplus(phone_prediction[:,:,0])*(1-pause)+F.softplus(phone_prediction[:,:,1])*pause
    dur=weighted(F.smooth_l1_loss(durpred,duration,reduction='none'),pm*(1+pause))
    presence=weighted(F.binary_cross_entropy_with_logits(phone_prediction[:,:,2].float(),
                   (batch['durations']>0).float(),reduction='none'),pm*pause)
    prosody=weighted((phone_prediction[:,:,[3,4]]-batch['phone_prosody']).abs(),
                     batch['phone_prosody_weight']*pm[:,:,None])
    total=base+.25*delta+.1*gv+.2*voice+.3*dur+.1*presence+.15*prosody
    return total,dict(acoustic=base,delta=delta,global_variance=gv,voice=voice,duration=dur,
                      pause=presence,phone_prosody=prosody)
