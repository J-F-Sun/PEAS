"""Portable fitted spatial basis + original SciPy dense time interpolant."""
import numpy as np
import cloudpickle
from scipy.integrate import cumulative_trapezoid
from .geometry import sphere_surface


class FrozenModel:
    def __init__(self, solver, config, source, sigma):
        dense=solver.ansatz._model.steps[0][1];linear=solver.ansatz._model.steps[1][1]
        self.w=dense.weights.copy();self.b=dense.biases.copy()
        self.a=linear.weights.copy();self.ab=linear.biases.copy()
        self.v=solver._V_a.copy();self.coefficients=solver._coefficients_c
        self.config=config;self.source=np.array(source);self.sigma=sigma

    def basis(self,x,normal=None):
        from swimpde.utils import tanh_x
        z=x@self.w+self.b
        if normal is None:
            return (np.tanh(z)@self.a+self.ab)@self.v.T
        return ((tanh_x(z)*(normal@self.w))@self.a)@self.v.T

    def density(self,x,times):
        return self.basis(np.atleast_2d(x))@self.coefficients(np.atleast_1d(times))

    def rx_vector(self,n=None):
        g=self.config['geometry']
        if self.config['representation']=='radial':
            return self.basis(np.array([[g['rx_radius']]]),np.ones((1,1)))[0]*4*np.pi*g['rx_radius']**2*g['diffusivity']
        x,normals,weights=sphere_surface(g['rx_center'],g['rx_radius'],n or self.config['evaluation']['surface_points'])
        # normal of fluid domain is -normals, hence + radial derivative.
        return g['diffusivity']*(weights@self.basis(x,normals))

    def probability(self,times):
        times=np.asarray(times)
        # Observation only; no positivity/monotonicity correction.
        dense=np.unique(np.r_[np.linspace(0,self.config['time']['end'],self.config['evaluation']['integration_points']),times])
        flux=self.rx_vector()@self.coefficients(dense)
        q=cumulative_trapezoid(flux,dense,initial=0)
        return np.interp(times,dense,q),flux

    def save(self,path):
        with open(path,'wb') as f:cloudpickle.dump(self,f)

    @staticmethod
    def load(path):
        # Load only trusted locally produced artifacts.
        with open(path,'rb') as f:return cloudpickle.load(f)
