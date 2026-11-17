from dataclasses import dataclass
import numpy as np
from swimpde.Domain import Domain
from swimpde.Ansatz import Ansatz
from typing import Callable
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy
from dataclasses import field
import copy
from scipy.stats import rv_discrete
import time 
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

@dataclass
class Diffusion_Solver:
    """
    Solver for the Diffusion equation
    ∂u(x,t)/∂t =  ∂^2u(x,t)/∂^2x  + f(x , t)

    with initial condition for u(x,t)

    Attributes:
    -----------
    domain: Domain
    ansatz: Ansatz
        basis functions from which the solution will be built by linear combination
        (use BoundaryCompliantAnsatz for this solver to ensure the boundary conditions are fulfilled)
    u0: Callable
        solution at time t0
    boundary_condition: str 
        boundary condition, one of "zero neumann"/"zero derivative" or "zero dirichlet"/"zero"
    forcing: Callable  
        forcing, a function of x and t
    regularization_scale: float
        regularization scale for computing the matrix inverse and solving least squares roblems
    ode_solver: str
        ode solver (to be used as 'method' in scipy.integrate.solve_ivp)
    boundary_condition_true: Callable
        Pass true function that computes the boundary condition
    """
    domain: Domain
    ansatz: Ansatz
    u0: Callable
    boundary_condition: str 
    forcing: Callable
    diff_coeff: float = 1
    regularization_scale: float = 1e-8
    ode_solver: str = 'DOP853' #'RK45'
    ansatz_collection: list = field(default_factory=list)
    c_collection: list = field(default_factory=list)
    svd_collecion: list = field(default_factory=list)
    scale_boundary_correction: float = 1000
    #args: list = None 
    boundary_condition_true: Callable = None

    def __post_init__(self):
        # initialize internal parameters
        # ode solution for the time-dependent coefficients
        self._coefficients_c: Callable = None # time dependent coefficients, solution from solver

        # matrices for the ODE
        self._B: np.ndarray[np.float64] = None
        self._A: np.ndarray[np.float64] = None
        self._A_inv: np.ndarray[np.float64] = None
        self._C: np.ndarray[np.float64] = None
        self._V_a: np.ndarray[np.float64] = None
       
    def evaluate(self, x_eval, t_eval, svd_on = True):
        '''
        Evaluate the solution at given time and space points

        Parameters:
            x_eval: (n_eval, d), n_eval is the number of points, d is the dimension
            t_eval: (t, )
            
        Returns:
            sol_diffusion: (n, t)
        '''
        
        if svd_on:
            sol_c = self._coefficients_c(t_eval.reshape((np.shape(t_eval)[0], ))).T
            sol_diffusion = self.ansatz.evaluate_model(x_eval) @ (self._V_a).T @ sol_c.T
        else:
            sol_c = self._coefficients_c(t_eval.reshape((np.shape(t_eval)[0], ))).T
            #sol_c = self._coefficients_c(t_eval).T
            sol_diffusion = self.ansatz.evaluate_model(x_eval) @ sol_c.T
        
        return sol_diffusion

    def evaluate_gradient(self, x_eval, t_eval):
        '''
        Evaluate the gradient of the solution at given time and space points

        Parameters:
            x_eval: (n_eval, d), n_eval is the number of points, d is the dimension
            t_eval: (t, )
            
        Returns:
            sol_diffusion: (n, t)
        '''
        sol_c = self._coefficients_c(t_eval.reshape((np.shape(t_eval)[0], ))).T
        grad_feature_matrix = self.ansatz.evaluate_model_gradient(x_eval)
        grad_feature_matrix = grad_feature_matrix.reshape((grad_feature_matrix.shape[0], grad_feature_matrix.shape[1]))
        gradient = grad_feature_matrix @ (self._V_a).T @ sol_c.T
        return gradient


    def _init_matrices(self, svd_cutoff, outer_basis=False, 
                       no_ode=False, svd_on=True, du_dt=None):
        '''
        Set all matrices that occur in the ODE

        Parameters:
        rcond: regularization scale for inverse in gamma_A_inv
        '''
        # self._A = self.ansatz.evaluate_model(self.domain.all_points) #CD
        self._A = self.ansatz.evaluate_model(self.domain.interior_points)
                
        # For Dirichlet Boundary conditions
        self._A_boundary =  self.ansatz.evaluate_model(self.domain.boundary_points)
        
        if svd_on:
            U_a, S_a, V_a = np.linalg.svd(self._A, full_matrices=False)
        
            # Truncate singular values and corresponding columns of U and rows of Vt
            mask = S_a / np.max(S_a) > svd_cutoff
            S_a = S_a[mask]
            U_a = U_a[:, mask]
            V_a = V_a[mask, :]
            self._V_a = V_a
            self._A = self._A @ self._V_a.T

            self._A_boundary = (self._A_boundary).reshape((self._A_boundary.shape[0], -1)) @ (self._V_a).T # n_points, n_neurons, 1


        if no_ode:
            if svd_on:
                # Laplacian 
                self._C = (self.ansatz.evaluate_model_laplace(self.domain.all_points)) @ (self._V_a).T #[:, :, 0]
            else:
                self._C = (self.ansatz.evaluate_model_laplace(self.domain.all_points)) #[:, :, 0]

        else:
            if svd_on:            
                # Calculate the generalized inverse of a matrix using its singular-value decomposition (SVD) and including all large singular values.
                self._A_inv = np.linalg.pinv(self._A, svd_cutoff)
                self._A_boundary_inv = np.linalg.pinv(np.row_stack([self._A, self._A_boundary]), rcond = svd_cutoff)
                self._C = (self.ansatz.evaluate_model_laplace(self.domain.interior_points)) @ (self._V_a).T

            else:
                # Calculate the generalized inverse of a matrix using its singular-value decomposition (SVD) and including all large singular values.
                self._A_inv = np.linalg.pinv(self._A, rcond = svd_cutoff)
                self._A_boundary_inv = np.linalg.pinv(np.row_stack([self._A, self._A_boundary]), rcond = svd_cutoff)
                self._C = (self.ansatz.evaluate_model_laplace(self.domain.interior_points))
            # Laplacian 

            def ODE_coefficients(t, c, outer_basis=outer_basis):
                """
                The ODE to be solved for the time-dependent coefficients
                """               
                f = self.forcing(self.domain.interior_points,t)
                laplacian_u = (c @ self._C.T)
                rhs = (laplacian_u + f).T
                
                if outer_basis:
                    c_t = self._A_inv @ rhs # this is without adding zero boundary condition
                else:
                # The following is by adding the zero boundary condition.
                    if self.boundary_condition_true is not None:
                        boundary_correction = self.scale_boundary_correction * (self.boundary_condition_true(self.domain.boundary_points, t).reshape(-1) - c @ self._A_boundary.T)
                    else:
                        boundary_correction = -c @ self._A_boundary.T
                    
                    if du_dt is not None: # If true time-derivative is known on the boundary
                        c_t = self._A_boundary_inv @ np.concatenate([rhs, du_dt(self.domain.boundary_points, t)])
                    else:
                        c_t = self._A_boundary_inv @ np.concatenate([rhs, boundary_correction])
                
                return c_t.ravel()
            self.ODE_coefficients = ODE_coefficients

        return self
    
    def _get_c0(self, initial_sol=None, outer_basis=False):
        '''
        Initial condition of the time-dependent coefficients for the ODE solver
        initial_sol =  Solution at the end of the previous time-block evaluated at all domain points
        Returns:
            c0: shape ((k+1)*2, ); initial condition for c (first k+1 entries) and d (= c_t, last k+1 entries)
        '''
        if initial_sol is None:
            if outer_basis:
                c0 = np.linalg.lstsq(self._A, self.u0(self.domain.interior_points), self.regularization_scale)[0]
                
            else:
                c0 = self._A_boundary_inv @ np.concatenate([self.u0(self.domain.interior_points), self.u0(self.domain.boundary_points)])
        else:
            if outer_basis:
                c0 = np.linalg.lstsq(self._A, initial_sol, self.regularization_scale)[0]
            else:
                c0 = self._A_inv @ initial_sol
        return c0
    
    def fit_no_ode_dirichlet_bc(self, data_init, svd_cutoff=None):
        '''
        Approximate the solution of the advection problem by choosing the model parameters and time-dependent coefficients accordingly
        '''
        # set up the model for the ansatz function
        self.ansatz.init_model(self.domain, self.boundary_condition) 
        
        # Perhaps redundant
        target = np.zeros_like((self.domain.interior_points))
        self.ansatz.fit_model(self.domain.interior_points, target)

        # Compute the matrices needed in the ODE
        self._init_matrices(svd_cutoff, no_ode=True)

        # Find coefficients of last layer
        # find a linear combination of basis functions that satisfies the PDE as well as possible
        basis_space = self._D + self._A * self._B - self.diff_coeff * self._C
        n_b = int(np.shape(self.domain.boundary_points)[0]/2)
        left = self.domain.boundary_points[:n_b,:]
        right = self.domain.boundary_points[n_b:,:]
        left_bc = self.ansatz.evaluate_model(left) @ self._V_a.T
        right_bc = self.ansatz.evaluate_model(right) @ self._V_a.T
        
        x_init = self.ansatz.evaluate_model(data_init[:, :2]) @ self._V_a.T
        matrix_in = np.row_stack(
            [
                 basis_space,
                 100 * left_bc,
                 100 * right_bc,
                 x_init
            ]
        )
        matrix_bias = np.row_stack([
            np.ones((basis_space.shape[0],1)),
            np.ones((left_bc.shape[0],1)),
            np.ones((right_bc.shape[0],1)),
            np.ones((data_init.shape[0],1)),
        ])
        matrix_in = np.column_stack([matrix_in,matrix_bias])

        matrix_out = np.hstack([
            np.zeros((basis_space.shape[0],)),
            np.zeros((left_bc.shape[0],)),
            np.zeros((right_bc.shape[0],)),
            data_init[:, 2]
        ])
        last_layer_weights = np.linalg.lstsq(
            matrix_in, matrix_out, rcond=self.ansatz.regularization_scale
        )[0]

        self._coefficients = last_layer_weights
        return self
    

    def evaluate_no_ode(self, xt_eval):
        '''
        Evaluate the solution at given time and space points

        Parameters:
            xt_eval: (n_eval, d), n_eval is the number of points, d is the dimension (including time)
            
        Returns:
            sol_adv: (n)
        '''
        
        sol_adv = self.ansatz.evaluate_model(xt_eval) @ (self._V_a).T @ self._coefficients[:-1] + self._coefficients[-1]
        return sol_adv
    

    def fit(self, t_span, rtol=1e-8, atol=1e-8, svd_cutoff=None, 
            outer_basis=True, time_blocks = 1, init_cond=None, 
            svd_on=True, du_dt=None):
        '''
        Approximate the solution of the advection problem by choosing the model parameters and time-dependent coefficients accordingly
        '''
        # set up the model for the ansatz function
        self.ansatz.init_model(self.domain, self.boundary_condition, initial_condition=init_cond) 

        # compute the matrices needed in the ODE
        if svd_cutoff is None:
            svd_cutoff = self.regularization_scale * 10
        self._init_matrices(svd_cutoff=svd_cutoff, 
                            outer_basis=outer_basis, 
                            svd_on=svd_on,
                            du_dt=du_dt)

        # get the initial value for the ODE
        c_0 = self._get_c0(outer_basis=outer_basis).reshape(-1)

        def event_func(t, y):
            # Define the event function to trigger when the absolute value of the solution exceeds a particular value
            return max(y) - 1e10

        event_func.terminal = False
        
        # solve the ODE
        ode_started = time.perf_counter()
        solver = solve_ivp(fun=self.ODE_coefficients, t_span=t_span, y0=c_0, dense_output=True, method=self.ode_solver, rtol=rtol, atol=atol,events=event_func)
        self._coefficients_c = solver.sol
        self._fit_info = dict(nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu, seconds=time.perf_counter()-ode_started) 
        
        print('Number of functon evaluations of the ODE solver: ', solver.nfev)
        # Check if the integration was successful and the event was triggered
        if solver.status != 0:
            print("Integration failed or terminated due to exceeding the maximum absolute value.")

        return self, solver.status

    def resample_data_points(self,gradient, p_distr_resampling, n_col = 1000):
        # We set the new interior and boundary points for the next time step by resampling
        rng = np.random.default_rng(seed=5) # Random seed
        
        # Compute probability distribution
        probabilities = p_distr_resampling(gradient).reshape(-1)
        cdf_values = cumtrapz(probabilities, self.domain.sample_points.reshape(-1,), initial=0)
        cdf_values /= cdf_values[-1]  # Normalize to make sure it goes from 0 to 1

        # Create an interpolation function for the inverse CDF
        inv_cdf_interp = interp1d(cdf_values, self.domain.sample_points.reshape(-1,), kind='quadratic', bounds_error=True)

        # Generate random samples from a uniform distribution
        uniform_samples = rng.random(n_col)

        # Use the inverse CDF interpolation function to map uniform samples to samples from the desired distribution
        x_collocation = inv_cdf_interp(uniform_samples)
        self.domain.interior_points = x_collocation.reshape((-1, 1))
        self.domain.set_all_points()


    def fit_time_blocks(self, t_span, rtol=1e-8, atol=1e-8, svd_cutoff=None, time_blocks = 1, prob_distr_resampling=None, n_col = 2000, outer_basis=True):
        '''
        Approximate the solution of the advection problem by choosing the model parameters and time-dependent coefficients accordingly
        '''
        def event_func(t, y):
            # Define the event function to trigger when the absolute value of the solution exceeds a particular value
            return max(y) - 1e10
        
        event_func.terminal = True # Terminate initial value problem if event_func is satisfied
        self.ansatz_collection = []
        self.c_collection = []
        self.svd_collecion = []
        
        t_block_size = (t_span[-1] - t_span[0])/time_blocks
        for i in range(time_blocks):
            # Solve the ODE for one time block
            t_block = [i * t_block_size, (i+1) * t_block_size]
            if i == 0:
                # set up the model for the ansatz function
                self.ansatz.init_model(self.domain, self.boundary_condition, initial_condition=None)
                self.ansatz_collection.append(copy.deepcopy(self.ansatz))

            else:
                # set up the model for the ansatz function: Pass previous solution as the target function
                self.resample_data_points(gradient=gradient, p_distr_resampling = prob_distr_resampling, n_col = n_col)
                sol_approx_interior = self.evaluate(self.domain.interior_points, t_block[0].reshape(-1,))
                sol_approx_all = self.evaluate(self.domain.all_points, t_block[0].reshape(-1,))
                self.ansatz.init_model(self.domain, self.boundary_condition, initial_condition=sol_approx_interior) #, initial_condition=initial_sol
                self.ansatz_collection.append(copy.deepcopy(self.ansatz))
            
            # Compute the matrices needed in the ODE (using boundary + interior/collocation points)
            if svd_cutoff is None:
                svd_cutoff = self.regularization_scale * 10
            self._init_matrices(svd_cutoff=svd_cutoff, outer_basis=outer_basis)

            # Store the SVD: Required for evaluating later in the time-blocking approach
            self.svd_collecion.append(copy.deepcopy(self._V_a))
            
            # Initialize coeffcients for the (re)-sampled weights
            if i == 0:
                c_0 = self._get_c0(outer_basis=outer_basis).reshape(-1)
                self._coefficients_c = c_0 # Dims of c_0 = num_svd (A: (interior + boundary pts) * svd, u_0 = (int + boundary_pts) * 1)
            else:
                c_0 = self._get_c0(initial_sol=sol_approx_all, outer_basis=outer_basis).reshape(-1) # Boundary + collocation
                
            # solve the ODE
            time_st = time.time()
            solver = solve_ivp(fun=self.ODE_coefficients, t_span=t_block, y0=c_0, dense_output=True, method=self.ode_solver, rtol=rtol, atol=atol,events=event_func)
            time_end = time.time()
            time_solver = time_end - time_st
            self._coefficients_c = solver.sol
            
            # Store the interpolant to evaluate afterwards
            if solver.status == 0:
                self.c_collection.append(copy.deepcopy(self._coefficients_c))

            else:
                print("Integration failed or terminated due to exceeding the maximum absolute value.")
                for j in range(i, time_blocks):
                    self.c_collection.append(lambda t : t * 100)
                break
            
            # Domain for evaluation:
            if i < time_blocks - 1:
                if solver.success:
                    gradient = np.abs(self.evaluate_gradient(self.domain.sample_points, t_block[1].reshape(-1,))) # Sample points
                else:
                    break
        return self, solver.status
    
    
    def evaluate_blocks(self, x_eval, t_eval, time_blocks = 1, solver_status=0):
        '''
        Evaluate the solution at given time and space points

        Parameters:
            x_eval: (n_eval, d), n_eval is the number of points, d is the dimension
            t_eval: (t, )
            
        Returns:
            sol_diffusion: (n, t)
        '''
        if solver_status == 0:
            t_block_size = (t_eval[-1] - t_eval[0])/time_blocks
            for i in range(time_blocks):
                if i < time_blocks - 1:
                    sol_c = self.c_collection[i](t_eval[(i*t_block_size <= t_eval) & (t_eval < (i+1)*t_block_size)]).T
                else:
                    sol_c = self.c_collection[i](t_eval[(i*t_block_size <= t_eval) & (t_eval <= (i+1)*t_block_size)]).T
                
                # Compute solution of diffusions equation using appropriate basis functions for the particular time-block
                sol_diffusion_block = self.ansatz_collection[i].evaluate_model(x_eval) @ self.svd_collecion[i].T @ sol_c.T
                if i == 0:
                    sol_diffusion = sol_diffusion_block
                else:
                    sol_diffusion = np.hstack((sol_diffusion, sol_diffusion_block))
        else:
            sol_diffusion = np.ones((np.shape(x_eval)[0], np.shape(t_eval)[0])) * 1000
        return sol_diffusion

