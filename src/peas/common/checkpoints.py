"""Safe model loading and full training state at Adam / L-BFGS boundaries."""
from __future__ import annotations
import copy
import os
import random
from pathlib import Path
import numpy as np
import torch
from peas.common.model import build_model

FORMAT_VERSION = 1

def portable_config(config, parent):
    """Store paths relative to a checkpoint/config file rather than a host."""
    config=copy.deepcopy(config)
    for section,key in (('paths','output_root'),('steady_state','checkpoint'),('evaluation','reference')):
        value=config[section].get(key)
        if value is not None and Path(value).is_absolute():
            config[section][key]=os.path.relpath(value,Path(parent).resolve())
    return config

def capture_rng():
    state = np.random.get_state()
    return {'python':random.getstate(),'numpy':[state[0],state[1].tolist(),state[2],state[3],state[4]],
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python'])
    n=state['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])

def read_checkpoint(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or 'model_state_dict' not in payload:
        raise ValueError('Checkpoint lacks model_state_dict')
    return payload

def save_checkpoint(path,model,state):
    path=Path(path)
    steady=model.config['steady_state'].get('checkpoint')
    payload={'format_version':FORMAT_VERSION,'config':portable_config(model.config,path.parent),
             'model_state_dict':model.transient.state_dict(),'training_state':state,
             'steady_checkpoint':os.path.relpath(steady,path.parent) if steady else None}
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp)
    tmp.replace(path)

def load_model(path, *, config=None, steady_checkpoint=None, device='cpu'):
    path=Path(path).resolve()
    payload=read_checkpoint(path)
    if payload.get('format_version')==FORMAT_VERSION:
        config=copy.deepcopy(payload['config'])
        for section,key in (('paths','output_root'),('steady_state','checkpoint'),('evaluation','reference')):
            value=config[section].get(key)
            if value is not None and not Path(value).is_absolute():
                config[section][key]=str((path.parent/value).resolve())
        stored=payload.get('steady_checkpoint')
        if steady_checkpoint is None and stored:
            candidate=path.parent/stored
            if candidate.is_file(): steady_checkpoint=str(candidate.resolve())
    elif config is None:
        raise ValueError('Old checkpoints require explicit --config and --steady-checkpoint for FEM assets')
    else:
        # Check old feature settings, rather than silently assigning a new Phi to weights.
        old=payload['config']
        from peas.common.config import native_config
        n=native_config(config)
        for section in ('features','model','domain','time'):
            for key,value in old[section].items():
                if section=='time' and key=='sampling': continue
                if key in n[section] and n[section][key]!=value:
                    raise ValueError(f'Old checkpoint and config disagree: {section}.{key}')
    if steady_checkpoint is not None:
        config=copy.deepcopy(config)
        config['steady_state']['checkpoint']=str(Path(steady_checkpoint).resolve())
    config['runtime']={**config['runtime'],'device':str(device)}
    model=build_model(config,device)
    model.transient.load_state_dict(payload['model_state_dict'],strict=True)
    return model,payload
