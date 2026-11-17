from dataclasses import dataclass
import numpy as np
from swimpde.Domain import Domain
from swimpde.Ansatz import Ansatz
from typing import Callable

@dataclass
class PoissonSolver:
    """
    Solver for the Poisson equation 
        𝚫u(x) = f(x) on 𝛀
        u(x) = g(x) on ∂𝛀

    Attributes:
    -----------
    domain: Domain
    f: Callable
        forcing function
    g: Callable 
        boundary condition
    ansatz: Ansatz 
        basis functions from which the solution will be built by linear combination
        (use BasicAnsatz for this solver)
    """
    domain: Domain
    f: Callable 
    g: Callable

    ansatz: Ansatz
    parameter_scaling: np.ndarray = None # constant 1 if kept at None

    def __post_init__(self):
        self._coefficients: np.ndarray = None 

    
    def fit(self):
        '''
        Approximate the solution of the Poisson problem by choosing the model parameters accordingly
        '''
        #initialize the model
        self.ansatz.init_model(self.domain)

        # fit the model to the forcing function
        self.ansatz.fit_model_laplace(self.domain.interior_points, self.f(self.domain.interior_points))

        # find a linear combination of basis functions that satisfies the PDE as well as possible
        if self.parameter_scaling is None:
            self.parameter_scaling = np.ones((self.domain.interior_points.shape[1], ))
        matrix_in = np.row_stack(
            [
                self.ansatz.evaluate_model_laplace(self.domain.interior_points, parameter_scaling=self.parameter_scaling),
                self.ansatz.evaluate_model(self.domain.boundary_points),
            ]
        )
        matrix_in = np.column_stack(
            [
                matrix_in,
                np.row_stack(
                    [np.zeros((self.domain.interior_points.shape[0], 1)), np.ones((self.domain.boundary_points.shape[0], 1))]
                ),
            ]
        )
        matrix_out = np.row_stack([self.f(self.domain.interior_points), self.g(self.domain.boundary_points)])
        last_layer_weights = np.linalg.lstsq(
            matrix_in, matrix_out, rcond=self.ansatz.regularization_scale
        )[0]

        self._coefficients = last_layer_weights

        return self

       
    def evaluate(self, x):
        '''
        Evaluate the solution
        '''
        return self.ansatz.evaluate_model(x) @ self._coefficients[:-1] + self._coefficients[-1]

    def evaluate_laplace(self, x):
        '''
        Evaluate the learned forcing function
        '''
        return self.ansatz.evaluate_model_laplace(x) @ self._coefficients[:-1] 
