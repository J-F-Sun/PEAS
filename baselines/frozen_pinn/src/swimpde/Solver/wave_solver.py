from dataclasses import dataclass
import numpy as np
from swimpde.Domain import Domain
from swimpde.Ansatz import Ansatz
from typing import Callable
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy


@dataclass
class WaveSolver:
    """
    Solver for the advection equation
    ∂^(2) u(x,t) / ∂^(2) t + ∂^(4) u(x,t)/ ∂^(4) x = f(t,x)

    with initial condition for u(t,x) and boundary condition (only zero dirichlet/neumann supported)

    Attributes:
    -----------
    domain: Domain
    ansatz: Ansatz
        basis functions from which the solution will be built by linear combination
        (use BoundaryCompliantAnsatz for this solver to ensure the boundary conditions are fulfilled)
    u0: Callable
        solution at time t0
    ut0: Callable
        time derivative of solution at time t0
    boundary_condition: str 
        boundary condition, one of "zero neumann"/"zero derivative" or "zero dirichlet"/"zero"
    forcing: Callable  
        forcing, a function of x and t
    c: float = 1 
        wave speed, constant
    regularization_scale: float
        regularization scale for computing the matrix inverse and solving least squares roblems
    ode_solver: str
        ode solver (to be used as 'method' in scipy.integrate.solve_ivp)

    """
    domain: Domain
    ansatz: Ansatz

    u0: Callable
    ut0: Callable

    boundary_condition: str 
    forcing: Callable
    c: float = 1

    regularization_scale: float = 1e-8
    ode_solver: str = 'DOP853' #'RK45'



    def __post_init__(self):
        # initialize internal parameters
        # ode solution for the time-dependent coefficients
        self._coefficients_c: Callable = None # time dependent coefficients, solution from solver

        # matrices for the ODE
        self._B: np.ndarray = None
        self._A: np.ndarray = None
        self._A_inv: np.ndarray = None
        self._A_one: np.ndarray = None
        self._V_a: np.ndarray = None

    
    def fit(self, t_span, rtol=1e-8, atol=1e-8, 
            svd_cutoff=None, time_blocks = 1,
            outer_basis=True, penalty=1e4):
        '''
        Approximate the solution of the advection problem by choosing the model parameters and time-dependent coefficients accordingly
        '''
        # set up the model for the ansatz function
        self.ansatz.init_model(self.domain, self.boundary_condition) 

        # define the ODE to be solved for the time-dependent coefficients
        def cd_t(t, cd):
            c, d = cd[:len(cd)//2, ], cd[len(cd)//2:, ]
            if outer_basis:
                c_t = d.ravel()
                rhs = self.forcing(self.domain.interior_points,t).reshape(-1) + self.c * c @ self._B.T
                d_t = self._A_inv @ rhs
                return np.concatenate([c_t, d_t])
            else:
                # # Compute the rhs of the augmented ODE
                # rhs_c_t = np.hstack((
                #     d @ self._A.T,
                #     -penalty * c @ self._A_boundary.T,
                #     -penalty * c @ self._C_boundary.T
                # ))
                # # Flatten the result directly
                # c_t = self.bc_inv @ rhs_c_t
                # c_t = c_t.ravel()

                # rhs_d_t = self.forcing(self.domain.interior_points,t).reshape(-1) - c @ self._B.T #- self.c * c @ self._A.T
                # d_t = self._A_inv @ rhs_d_t
                # d_t = d_t.ravel()

                # TODO: fix here as well
    
                return np.concatenate([c_t, d_t])

        # compute the matrices needed in the ODE
        self._init_matrices(svd_cutoff=svd_cutoff)

        # get the initial value for the ODE
        c_0 = self._get_cd0().reshape(-1)
        
        def event_func(t, y):
            # Define the event function to trigger when the absolute value of the solution exceeds a particular value
            return max(y) - 1e15

        event_func.terminal = True

        # solve the ODE
        solver = solve_ivp(fun=cd_t, t_span=t_span, y0=c_0, dense_output=True, method=self.ode_solver, rtol=rtol, atol=atol,events=event_func)
        self._coefficients_c = solver.sol 
        
        # Check if the integration was successful and the event was triggered
        if solver.status == 0:
            print("Integration successful.")
        else:
            print("Integration failed or terminated due to exceeding the maximum absolute value.")

        return self
    
    
    def evaluate(self, x_eval, t_eval):
        '''
        Evaluate the solution at given time and space points

        Parameters:
            x_eval: (n_eval, d), n_eval is the number of points, d is the dimension
            t_eval: (t, )
            
        Returns:
            sol_adv: (n, t)
        '''
        
        sol_cd = self._coefficients_c(t_eval).T
        sol_c = sol_cd[:, :sol_cd.shape[1]//2]
        sol_euler_bernoulli = self.ansatz.evaluate_model(x_eval) @ (self._V_a).T @ sol_c.T
        return sol_euler_bernoulli
        
    def _init_matrices(self, svd_cutoff=None):
        '''
        Set all matrices that occur in the ODE

        Parameters:
        rcond: regularization scale for inverse in gamma_A_inv
        '''
        self._A = self.ansatz.evaluate_model(self.domain.interior_points)
        self._A_boundary = self.ansatz.evaluate_model(self.domain.boundary_points)

        if svd_cutoff is None:
            svd_cutoff = self.regularization_scale * 10
        
        U_a, S_a, V_a = np.linalg.svd(self._A, full_matrices=False)
        idx_s = S_a / np.max(S_a) > svd_cutoff
        V_a = V_a[idx_s, :]
        
        self._V_a = V_a

        # Important matrices and their inverses
        self._A = self._A @ self._V_a.T
        self._A_boundary = self._A_boundary @ self._V_a.T
        self._C_boundary = self.ansatz.evaluate_model_laplace(self.domain.boundary_points) @ self._V_a.T
        self.bc = np.row_stack([self._A, self._A_boundary, self._C_boundary])
        self._A_inv = np.linalg.pinv(self._A, rcond = self.regularization_scale)
        self.bc_inv = np.linalg.pinv(self.bc, rcond = self.regularization_scale)
        # self._B =  (self.ansatz.evaluate_model_fourth_order_diff(self.domain.interior_points))
        self._B = (self.ansatz.evaluate_model_laplace(self.domain.interior_points))
        self._B =  (self._B).reshape((self._B.shape[0], -1)) @ (self._V_a).T # n_points, n_neurons, 1
        return self
    
    def _get_cd0(self):
        '''
        Initial condition of the time-dependent coefficients for the ODE solver
        Returns:
            c0: shape ((k+1)*2, ); initial condition for c (first k+1 entries) and d (= c_t, last k+1 entries)
        '''
        c0 = np.linalg.lstsq(np.row_stack([self._A_boundary, self._A]), self.u0(self.domain.all_points), self.regularization_scale)[0]
        d0 = np.linalg.lstsq(np.row_stack([self._A_boundary, self._A]), self.ut0(self.domain.all_points), self.regularization_scale)[0]
        cd0 = np.concatenate([c0, d0]).ravel()
        return cd0
    