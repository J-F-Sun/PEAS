"""Minimal extensions to the vendored upstream boundary operator and radial Laplacian."""
import numpy as np
from swimpde.Ansatz import BoundaryCompliantAnsatz


class MixedBoundary:
    def __init__(self, domain, c):
        self.domain=domain;self.c=c
        # alpha*p + beta*normal_derivative(p) = 0.
        self.alpha=np.ones(len(domain.boundary_points))
        self.beta=np.zeros(len(self.alpha))
        outer=domain.labels=='outer'
        if c['scene']!='canonical':
            self.alpha[outer]=c['geometry']['kappa'];self.beta[outer]=c['geometry']['diffusivity']
        self.weight=c['numerics']['boundary_weight']

    def get_basis_space_bc(self,basis,domain,activation,activation_x,scale_bc=1):
        basis.activation=activation;phi=basis.transform(domain.boundary_points)
        basis.activation=activation_x;der=basis.transform(domain.boundary_points)
        dn=der*(domain.normal_vectors @ basis.weights)
        return self.weight*(self.alpha[:,None]*phi+self.beta[:,None]*dn)

    def get_bias_value(self):
        return self.weight*self.alpha[:,None]

    def get_target_bc(self,n_bc_points,n_outer_basis):
        return np.zeros((n_bc_points,n_outer_basis))

    def satisfied_by_constant(self):
        return False


class RadialAnsatz(BoundaryCompliantAnsatz):
    def evaluate_model_laplace(self,x):
        # Physical 3-D Laplacian acting on a radial density, without u=r*p.
        return super().evaluate_model_laplace(x)+2/x[:,0,None]*self.evaluate_model_gradient(x)[:,:,0]
