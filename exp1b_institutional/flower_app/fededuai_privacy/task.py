"""FedEduAI v19: student-example DP training; Flower 1.22.0 / Opacus 1.5.4.
Security scope: fixed roster/partition, one example per student, trusted local host.
Research evaluation metrics and non-DP comparison models are NOT DP releases.
"""
import json, os, time, secrets
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
get_noise_multiplier=lru_cache(maxsize=256)(get_noise_multiplier)


def make_net(dim, seed=0):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return nn.Sequential(nn.Linear(dim,32), nn.Tanh(), nn.Linear(32,2))


def weights(model):
    return [p.detach().cpu().numpy().copy() for p in model.state_dict().values()]


def set_weights(model, arrays):
    state=model.state_dict()
    if len(state)!=len(arrays): raise ValueError('Parameter count mismatch')
    for (k, old), a in zip(state.items(), arrays):
        if tuple(old.shape)!=tuple(a.shape) or not np.isfinite(a).all():
            raise ValueError('Invalid parameter shape/value')
        state[k]=torch.as_tensor(a,dtype=old.dtype)
    model.load_state_dict(state)


def probabilities(model, x):
    model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(x),2048):
            out.append(torch.softmax(model(torch.as_tensor(x[start:start+2048],dtype=torch.float32)),1)[:,1].numpy())
    return np.concatenate(out) if out else np.empty(0)


def local_train(arrays,x,y,cfg,history=None):
    """Keep the accountant across rounds; fresh SGD has no hidden optimizer state.
    DP noise/shuffle seeds are intentionally NOT set from published experiment seeds.
    secure_mode=False is a research RNG limitation, explicitly recorded.
    """
    if len(x)!=len(y) or len(x)<2: raise ValueError('Need at least two local students')
    if not np.isfinite(x).all() or not np.isin(y,[0,1]).all(): raise ValueError('Invalid training examples')
    model=make_net(x.shape[1],cfg['seed']);set_weights(model,arrays)
    optimizer=torch.optim.SGD(model.parameters(),lr=float(cfg['learning_rate']))
    # Never derive DP sampling/noise randomness from a published experiment seed.
    sample_generator=torch.Generator().manual_seed(secrets.randbits(63) if cfg['dp'] else
        int(cfg['seed'])*1000003+int(cfg.get('partition',0))*1009+int(cfg.get('round_number',0)))
    noise_generator=torch.Generator().manual_seed(secrets.randbits(63)) if cfg['dp'] and not cfg.get('secure_rng',False) else None
    loader=DataLoader(TensorDataset(torch.as_tensor(x,dtype=torch.float32),torch.as_tensor(y,dtype=torch.long)),
                      batch_size=min(int(cfg['batch_size']),len(x)),shuffle=True,generator=sample_generator)
    engine=None;sigma=None;start=time.perf_counter();steps=0
    use_clipping=bool(cfg['dp'] or cfg.get('clip_only',False) or cfg.get('poisson_only',False))
    if use_clipping:
        # Opacus Poisson sampler uses sample_rate=1/len(loader), including the last batch.
        q=1/len(loader)
        sigma=get_noise_multiplier(target_epsilon=float(cfg['epsilon']),target_delta=float(cfg['delta']),
            sample_rate=q,steps=len(loader)*int(cfg['rounds'])*int(cfg['local_epochs']),accountant='rdp')
        if cfg.get('clip_only',False) or cfg.get('poisson_only',False):sigma=0.0
        # poisson_only keeps the Opacus Poisson sampler but disables clipping via an unreachable norm bound.
        clip_norm=1e6 if cfg.get('poisson_only',False) else float(cfg['max_grad_norm'])
        engine=PrivacyEngine(accountant='rdp',secure_mode=bool(cfg.get('secure_rng',False)))
        if history:
            engine.accountant.history=[(float(a),float(b),int(c)) for a,b,c in history]
        model,optimizer,loader=engine.make_private(module=model,optimizer=optimizer,data_loader=loader,
            noise_multiplier=sigma,max_grad_norm=clip_norm,poisson_sampling=True,noise_generator=noise_generator)
    model.train()
    for _ in range(int(cfg['local_epochs'])):
        for xb,yb in loader:
            optimizer.zero_grad(set_to_none=True)
            # A sum divided by max(n,1) handles valid empty Poisson batches without NaN.
            loss=nn.functional.cross_entropy(model(xb),yb,reduction='sum')/max(len(yb),1)
            loss.backward();optimizer.step();steps+=1
    base=model._module if engine else model
    hist=engine.accountant.history if engine and cfg['dp'] else []
    eps=float(engine.get_epsilon(float(cfg['delta']))) if engine and cfg['dp'] else None
    if eps is not None and eps>float(cfg['epsilon'])+.03:
        raise RuntimeError('Privacy budget exceeded; refuse to export this update')
    return weights(base),hist,{'steps':steps,'seconds':time.perf_counter()-start,
        'epsilon':eps,'delta':float(cfg['delta']) if engine else None,'noise_multiplier':sigma,
        'accountant':'rdp' if engine else None,'secure_rng':bool(cfg.get('secure_rng',False))}


def data_root():
    # Outside the Flower application, so private shards are never included in its FAB.
    value=os.environ.get('FEDEDUAI_DATA_DIR')
    if not value: raise RuntimeError('Set FEDEDUAI_DATA_DIR to this host\'s private data directory')
    return Path(value)


def load_shard(partition):
    with np.load(data_root()/f'client_{int(partition)}.npz',allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def run_cfg(context):
    r=context.run_config
    return {'seed':int(r['seed']),'rounds':int(r['num-server-rounds']),
        'local_epochs':int(r['local-epochs']),'batch_size':int(r['batch-size']),
        'learning_rate':float(r['learning-rate']),'max_grad_norm':float(r['max-grad-norm']),
        'epsilon':float(r['epsilon']),'delta':float(r['delta']),
        'dp':bool(r['dp']),'clip_only':bool(r.get('clip-only',False)),'poisson_only':bool(r.get('poisson-only',False)),
        'secure_rng':bool(r['secure-rng'])}
