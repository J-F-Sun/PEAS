"""Prediction input validation, optional physical-unit scaling, and coordinate adaptation."""
from dataclasses import dataclass
import numpy as np
import torch

@dataclass(frozen=True)
class InputAdapter:
    config: dict

    @property
    def formulation(self):
        """Describe the pre-derived equation; does not transform a PDE."""
        return 'backward cumulative absorption: q_t = Laplacian(q)'

    def nondimensionalize(self, times, points):
        """Convert explicitly physical prediction inputs; training uses dimensionless data."""
        scales = self.config['problem']['nondimensionalization']
        a, D = scales['length_scale'], scales['diffusivity']
        return np.asarray(times)*D/a**2, np.asarray(points)/a

    def native_coordinates(self, xyz):
        if self.config['scenario']=='canonical':
            center = xyz.new_tensor(self.config['problem']['domain'].get('rx_center',[0,0,0]))
            return 1.0/torch.linalg.vector_norm(xyz-center,dim=1,keepdim=True).clamp_min(1.0)
        return xyz

    def base_descriptors(self, t, xyz):
        center = xyz.new_tensor(self.config['problem']['domain'].get('rx_center',[0,0,0]))
        d = torch.linalg.vector_norm(xyz-center,dim=1,keepdim=True)
        return torch.cat([xyz,t,1.0/d,t/d.square()],dim=1)

    def validate_points(self, points):
        p = np.asarray(points,dtype=float)
        if p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all():
            raise ValueError('points must be a finite nonempty (N,3) array')
        g = self.config['problem']['domain']
        d = np.linalg.norm(p-np.asarray(g.get('rx_center',[0,0,0])),axis=1)
        invalid = np.any(d < 1-1e-8)
        if self.config['scenario'] in ('concentric','eccentric'):
            invalid |= np.any(np.linalg.norm(p-np.asarray(g['outer_center']),axis=1)>g['outer_radius']+1e-8)
        if self.config['scenario']=='pipe':
            invalid |= np.any(np.linalg.norm(p[:,:2],axis=1)>g['pipe_radius']+1e-8)
            invalid |= np.any((p[:,2]<g['z_min']-1e-8)|(p[:,2]>g['z_max']+1e-8))
        if invalid: raise ValueError('Prediction points lie outside the fluid domain')
        return p
