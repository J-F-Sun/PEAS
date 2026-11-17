"""Scenario physics objectives use the same Phi and frozen steady provider."""
from __future__ import annotations
from importlib import import_module
from peas.common.config import native_config

class PhysicsTraining:
    def __init__(self,c):
        self.config = c
        self.native = native_config(c)
        self.kernel = import_module('peas.scenarios.'+c['scenario']+'.physics_base' if c['scenario']=='canonical' else 'peas.scenarios.'+c['scenario']+'.physics')

    def make_batch(self,device, *, sizes=None, adaptive_points=None):
        if self.config['scenario']=='canonical':
            n = dict(self.native)
            if sizes is not None: n['training']={**n['training'],'batch_sizes':sizes}
            return self.kernel.make_batch(n,device)
        if self.config['scenario']=='pipe':
            return self.kernel.make_batch(self.native,device,sizes or self.native['training']['batch_sizes'])
        return self.kernel.make_batch(self.native,device,adaptive_points=adaptive_points,batch_sizes=sizes)

    def losses(self,model,batch):
        if self.config['scenario']=='canonical':
            return self.kernel.losses(model.transient,batch,self.native,model.steady.provider)
        return self.kernel.losses(model.transient,model.steady.provider,batch,self.native)

    def set_epoch(self,epoch):
        """Zero-based Adam epoch; retained objectives have no epoch state."""

    def state_dict(self):
        return {}

    def load_state_dict(self,state):
        if state:
            raise ValueError('Unexpected state for the original physics objective')

    def refresh_adaptive(self,model,device):
        return self.kernel.refresh_adaptive_points(model.transient,model.steady.provider,self.native,device)

    def seeded_batch(self,device,sizes,seed):
        if self.config['scenario']!='pipe': raise ValueError('Physical validation is configured only for pipe')
        return self.kernel.make_seeded_batch(self.native,device,sizes,seed)
