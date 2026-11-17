"""Frozen FEM fields; numerical gradients remain part of the PDE."""
from __future__ import annotations
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from peas.common.config import native_config, save_config
from peas.scenarios.canonical.steady import RadialFEMQInfinityProvider
from peas.scenarios.concentric.steady import CenteredFEMQInfinityProvider
from peas.scenarios.eccentric.steady import BisphericalFEMProvider, solve_q_infty_fem
from peas.common.utils import save_json

def validate_sphere_asset(path, c):
    g = c['problem']['domain']
    with np.load(path,allow_pickle=False) as d:
        if 'config' not in d: raise ValueError('Steady asset lacks geometry metadata')
        old = json.loads(str(d['config']))
        for key in ('rx_center','outer_center','rx_radius','outer_radius'):
            if not np.allclose(old['domain'][key],g[key],atol=1e-10,rtol=0):
                raise ValueError(f'Steady geometry mismatch: {key}')
        if not np.isclose(old['boundary']['kappa'],c['problem']['boundary']['kappa']):
            raise ValueError('Steady Robin coefficient mismatch')

def radial_provider(c):
    path = c['steady_state'].get('checkpoint')
    n = native_config(c)
    if not path: return RadialFEMQInfinityProvider(n)
    with np.load(path,allow_pickle=False) as data:
        saved = json.loads(str(data['parameters']))
        if saved != c['steady_state']['parameters']:
            raise ValueError('Radial FEM settings do not match asset')
        q = object.__new__(RadialFEMQInfinityProvider)
        q.outer_radius = saved['outer_radius']
        q.num_elements = saved['num_elements']
        q.mesh_power = saved['mesh_power']
        q.kappa = saved['robin_kappa']
        q.y_min = 1.0/q.outer_radius
        q._r = np.array(data['r'])
        q._q = np.array(data['q'])
        q._dq_dr = np.diff(q._q)/np.diff(q._r)
        return q

class FixedSteadyState:
    def __init__(self,c,device):
        self.config = c
        self.scene = c['scenario']
        n = native_config(c)
        path = c['steady_state'].get('checkpoint')
        if self.scene=='canonical': self.provider = radial_provider(c)
        elif self.scene=='concentric':
            self.provider = CenteredFEMQInfinityProvider(n,device,c['steady_state']['parameters'],path)
        elif self.scene=='eccentric':
            if not path: raise FileNotFoundError('Run prepare-steady or provide a steady checkpoint')
            validate_sphere_asset(path,c)
            self.provider = BisphericalFEMProvider(path,device)
        else:
            from peas.scenarios.pipe.physics import TorchFEMProvider, validate_qinf_checkpoint
            if not path: raise FileNotFoundError('Run prepare-steady or provide a steady checkpoint')
            validate_qinf_checkpoint(Path(path),n)
            p = c['steady_state']['parameters']
            self.provider = TorchFEMProvider(Path(path),device,p['gradient_method'],p['locator_candidates'])

    def value_and_derivative(self,coordinates):
        if self.scene=='canonical': return self.provider.value_and_dy(coordinates)
        return self.provider.value_and_grad(coordinates)

def prepare_steady(c, output):
    """Fresh field generation, retaining each original discretization."""
    c = copy.deepcopy(c)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    n = native_config(c,output)
    for path in n['outputs'].values(): Path(path).mkdir(parents=True,exist_ok=True)
    scene = c['scenario']
    if scene=='canonical':
        provider = RadialFEMQInfinityProvider(n)
        path = output/'steady.npz'
        np.savez_compressed(path,r=provider._r,q=provider._q,
                            parameters=json.dumps(c['steady_state']['parameters']))
        summary = provider.summary()
    elif scene=='concentric':
        provider = CenteredFEMQInfinityProvider(n,torch.device('cpu'),c['steady_state']['parameters'])
        path = output/'steady.npz'
        provider.save(path)
        summary = provider.summary()
    elif scene=='eccentric':
        summary, path = solve_q_infty_fem(n)
    else:
        from peas.scenarios.pipe.steady import solve
        n['project']['seed'] = c['steady_state']['parameters']['seed']
        summary = solve(n)
        path = Path(summary['checkpoint'])
    c['steady_state']['checkpoint'] = str(path.resolve()) if path else None
    save_json(output/'steady_summary.json',summary)
    save_config(c,output/'prepared.yaml')
    return c

def retain_steady_asset(c,run_dir):
    c = copy.deepcopy(c)
    source = c['steady_state'].get('checkpoint')
    if source:
        target = Path(run_dir)/'steady.npz'
        if Path(source).resolve()!=target.resolve(): shutil.copy2(source,target)
        c['steady_state']['checkpoint'] = str(target.resolve())
    return c
