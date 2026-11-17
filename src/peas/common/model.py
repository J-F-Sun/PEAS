"""Composed six-module PEAS model and Cartesian prediction contract."""
from __future__ import annotations
import numpy as np
import torch

class PEASModel:
    def __init__(self,config,input_adapter,features,transient,steady,ansatz,training):
        self.config = config
        self.input_adapter = input_adapter
        self.features = features
        self.transient = transient
        self.steady = steady
        self.ansatz = ansatz
        self.training = training

    def forward(self,t,xyz):
        coordinates = self.input_adapter.native_coordinates(xyz)
        u = self.transient(self.features(t,coordinates))
        Q, _ = self.steady.value_and_derivative(coordinates)
        return self.ansatz(Q,u)

    def predict(self,times,points, *, batch_size=None,physical_units=False):
        times = np.asarray(times,dtype=float)
        points = np.asarray(points,dtype=float)
        if physical_units: times,points = self.input_adapter.nondimensionalize(times,points)
        if times.ndim!=1 or not times.size or not np.isfinite(times).all() or np.any(times<0):
            raise ValueError('times must be a finite nonempty 1D array with t>=0')
        points = self.input_adapter.validate_points(points)
        size = int(batch_size or self.config['runtime']['prediction_batch_size'])
        if size<1: raise ValueError('prediction batch_size must be positive')
        device = next(self.transient.parameters()).device
        was_training = self.transient.training
        self.transient.eval()
        out = []
        with torch.no_grad():
            # Preserve the original pipe double-coordinate FEM lookup in inference.
            if self.config['scenario']=='pipe':
                p = self.config['steady_state']['parameters']
                Q,_,fallback = self.steady.provider.provider.value_and_grad(points,candidates=p['locator_candidates'])
                if np.any(fallback): raise ValueError('FEM locator fallback at prediction points')
            for i,point in enumerate(points):
                chunks=[]
                for start in range(0,len(times),size):
                    ts=times[start:start+size]
                    t=torch.tensor(ts[:,None],dtype=torch.float32,device=device)
                    xyz=torch.tensor(np.repeat(point[None],len(ts),axis=0),dtype=torch.float32,device=device)
                    if self.config['scenario']=='pipe':
                        u=self.transient(self.features(t,xyz)).cpu().numpy().ravel()
                        q=float(Q[i,0])*u
                    else: q=self.forward(t,xyz).cpu().numpy().ravel()
                    chunks.append(q)
                out.append(np.concatenate(chunks))
        self.transient.train(was_training)
        result=np.asarray(out)
        if not np.isfinite(result).all(): raise FloatingPointError('Nonfinite prediction')
        return result

    def predict_radii(self,times,radii,**kwargs):
        if self.config['scenario']!='canonical': raise ValueError('Radii API is canonical only')
        r=np.asarray(radii,dtype=float).reshape(-1)
        return self.predict(times,np.column_stack([r,np.zeros_like(r),np.zeros_like(r)]),**kwargs)

def build_model(config,device=None):
    from importlib import import_module
    from peas.common.config import validate_config
    from peas.common.utils import select_device
    validate_config(config)
    device = select_device(config['runtime']) if device is None else torch.device(device)
    scenario_module = 'concentric_radial' if config.get('representation') == 'radial' else config['scenario']
    return import_module('peas.scenarios.' + scenario_module).build(config, device)
