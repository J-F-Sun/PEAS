from dataclasses import dataclass
import numpy as np
from swimpde.Domain import Domain
from swimpde.Ansatz import Ansatz
from typing import Callable
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy
import copy

@dataclass
class AdvectionSolver:
    """
    Solver for the advection equation
    ∂u(x,t)/∂t = -c∂u(x,t)/∂x + f(t,x)

    with initial condition for u(t,x) and boundary condition (only zero dirichlet/neumann supported)

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
    boundary_condition: str 
    forcing: Callable
    c: float = 1

    regularization_scale: float = 1e-8
    ode_solver: str = 'DOP853' #'RK45'

    _coefficients_c = []


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
        self._C: np.ndarray = None


    def fit_time_blocks(self, t_span, rtol=1e-8, svd_cutoff=None, atol=1e-8, time_blocks = 1):
        '''
        Approximate the solution of the advection problem by choosing the model parameters and time-dependent coefficients accordingly
        '''
        # set up the model for the ansatz function
        self.ansatz.init_model(self.domain, self.boundary_condition) 

        # define the ODE to be solved for the time-dependent coefficients
        def c_t(t, c):
            rhs = - (self.c * (c @ self._B.T) + self.forcing(self.domain.all_points,t)).T
            c_t = self._A_inv @ rhs
            return c_t.reshape(-1)

        # compute the matrices needed in the ODE
        self._init_matrices(svd_cutoff=svd_cutoff)
        
        def event_func(t, y):
            # Define the event function to trigger when the absolute value of the solution exceeds a particular value
            return max(np.abs(y)) - 1e5
        
        event_func.terminal = True # Terminate initial value problem if event_func is satisfied
        t_block_size = (t_span[-1] - t_span[0])/time_blocks
        self._coefficients_c = []
        for i in range(time_blocks):
            t_block = [i * t_block_size, (i+1) * t_block_size]
            if i == 0:
                # get the initial value for the ODE
                c_0 = self._get_c0().reshape(-1)
            else:
                c_0 = self._coefficients_c[i-1](t_block[0]).reshape(-1)

            # solve the ODE
            solver = solve_ivp(fun=c_t, t_span=t_block, y0=c_0, dense_output=True, method=self.ode_solver, rtol=rtol, atol=atol,events=event_func)
            #self._coefficients_c.append(copy.deepcopy(solver.sol))
            
            if solver.status == 0:
                self._coefficients_c.append((copy.deepcopy(solver.sol))
)
            else:
                print("Integration failed or terminated due to exceeding the maximum absolute value.")
                for j in range(i, time_blocks):
                    self._coefficients_c.append(lambda t : t * 100)
                break
            
        return self, solver.status
    
    def evaluate_blocks(self, x_eval, t_eval, time_blocks = 1, solver_status=0):
        '''
        Evaluate the solution at given time and space points

        Parameters:
            x_eval: (n_eval, d), n_eval is the number of points, d is the dimension
            t_eval: (t, )
            
        Returns:
            sol_adv: (n, t)
        '''
        if solver_status == 0:
            t_block_size = (t_eval[-1] - t_eval[0])/time_blocks
            for i in range(time_blocks):
                if i < time_blocks - 1:
                    sol = self._coefficients_c[i](t_eval[(i*t_block_size <= t_eval) & (t_eval < (i+1)*t_block_size)]).T
                else:
                    sol = self._coefficients_c[i](t_eval[(i*t_block_size <= t_eval) & (t_eval <= (i+1)*t_block_size)]).T
                if i == 0:
                    sol_c = sol
            #sol_c = sol_c[:, :sol_c.shape[1]//2]
                else:
                    sol_c = np.vstack((sol_c, sol))
            sol_adv = self.ansatz.evaluate_model(x_eval) @ sol_c.T
        else:
            sol_adv = np.ones((np.shape(x_eval)[0], np.shape(t_eval)[0])) * 1000
        return sol_adv
    
    def fit_no_ode_periodic_bc(self, data_init, svd_cutoff=None):
        '''
        Approximate the solution of the advection problem by choosing the model parameters and time-dependent coefficients accordingly
        '''
        # set up the model for the ansatz function
        self.ansatz.init_model(self.domain, self.boundary_condition) 
        
        # Perhaps redundant
        target = np.zeros_like((self.domain.interior_points))
        self.ansatz.fit_model(self.domain.interior_points, target)

        # compute the matrices needed in the ODE
        self._init_matrices(svd_cutoff, no_ode=True)

        # Find coefficients of last layer
        # find a linear combination of basis functions that satisfies the PDE as well as possible
        basis_space = self.c * self._B + self._C
        n_b = int(np.shape(self.domain.boundary_points)[0]/2)
        left = self.domain.boundary_points[:n_b,:]
        right = self.domain.boundary_points[n_b:,:]
        c0_bc = (self.ansatz.evaluate_model(left) - self.ansatz.evaluate_model(right)) @ self._V_a.T
        c1_bc = (self.ansatz.evaluate_model_gradient(left)[:, :, 0] - self.ansatz.evaluate_model_gradient(right)[:, :, 0]) @ self._V_a.T         
        x_init = self.ansatz.evaluate_model(data_init[:, :2]) @ self._V_a.T
        matrix_in = np.row_stack(
            [
                 basis_space,
                 c0_bc,
                 c1_bc,
                 x_init
            ]
        )
        matrix_bias = np.row_stack([
            np.ones((basis_space.shape[0],1)),
            np.ones((c0_bc.shape[0],1)),
            np.zeros((c1_bc.shape[0],1)),
            np.ones((data_init.shape[0],1)),
        ])
        matrix_in = np.column_stack([matrix_in,matrix_bias])

        matrix_out = np.hstack([
            np.zeros((basis_space.shape[0],)),
            np.zeros((c0_bc.shape[0],)),
            np.zeros((c1_bc.shape[0],)),
            data_init[:, 2]
        ])
        last_layer_weights = np.linalg.lstsq(
            matrix_in, matrix_out, rcond=self.ansatz.regularization_scale
        )[0]

        self._coefficients = last_layer_weights
        return self
    

    def fit(self, t_span, rtol=1e-8, atol=1e-8, time_blocks = 1, svd_cutoff=None, svd_on = True):
        '''
        Approximate the solution of the advection problem by choosing the model parameters and time-dependent coefficients accordingly
        '''
        # set up the model for the ansatz function
        self.ansatz.init_model(self.domain, self.boundary_condition) 

        # define the ODE to be solved for the time-dependent coefficients
        def c_t(t, c):
            rhs = - (self.c * (c @ self._B.T) + self.forcing(self.domain.all_points,t)).T
            c_t = self._A_inv @ rhs
            return c_t.reshape(-1)

        # compute the matrices needed in the ODE
        self._init_matrices(svd_cutoff=svd_cutoff)

        # get the initial value for the ODE
        c_0 = self._get_c0().reshape(-1)
        
        def event_func(t, y):
            # Define the event function to trigger when the absolute value of the solution exceeds a particular value
            return max(y) - 1e5

        event_func.terminal = False

        # solve the ODE
        solver = solve_ivp(fun=c_t, t_span=t_span, y0=c_0, dense_output=True, method=self.ode_solver, rtol=rtol, atol=atol,events=event_func)
        self._coefficients_c = solver.sol 
        
        # Check if the integration was successful and the event was triggered
        """
        if solver.status == 0:
            print("Integration successful.")
        else:
            print("Integration failed or terminated due to exceeding the maximum absolute value.")
        """
        return self, solver.status
    
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
    

    def evaluate(self, x_eval, t_eval, solver_status=0, svd_on = True):
        '''
        Evaluate the solution at given time and space points

        Parameters:
            x_eval: (n_eval, d), n_eval is the number of points, d is the dimension
            t_eval: (t, )
            
        Returns:
            sol_adv: (n, t)
        '''
        if solver_status == 0:
            sol_c = self._coefficients_c(t_eval).T
            if svd_on:
                sol_adv = self.ansatz.evaluate_model(x_eval) @ (self._V_a).T @ sol_c.T
            else:
                sol_adv = self.ansatz.evaluate_model(x_eval) @ sol_c.T

        else:
            sol_adv = np.ones((np.shape(x_eval)[0], np.shape(t_eval)[0])) * 1000
        return sol_adv

        
    def _init_matrices(self, svd_cutoff=None, no_ode=False, svd_on = True):
        '''
        Set all matrices that occur in the ODE

        Parameters:
        rcond: regularization scale for inverse in gamma_A_inv
        '''
        self._A = self.ansatz.evaluate_model(self.domain.all_points)
        if svd_on:
            U_a, S_a, V_a = np.linalg.svd(self._A, full_matrices=False)
           
            if svd_cutoff is None:
                svd_cutoff = self.regularization_scale * 10
            idx_s = S_a / np.max(S_a) > svd_cutoff
            V_a = V_a[idx_s, :]
            
            self._V_a = V_a
            """
            U_a, S_a, V_a = np.linalg.svd(self._A)
            self._V_a = V_a
            """
            # Feature matrix
            self._A = self._A @ self._V_a.T
        
        if no_ode:
            # Partial derivative w.r.t x
            self._B =  (self.ansatz.evaluate_model_gradient(self.domain.all_points))[:, :, 0]
            self._B =  (self._B).reshape((self._B.shape[0], -1)) @ (self._V_a).T # n_points, n_neurons, 1

            # Partial derivative w.r.t t
            self._C =  (self.ansatz.evaluate_model_gradient(self.domain.all_points))[:, :, 1]
            self._C =  (self._C).reshape((self._C.shape[0], -1)) @ (self._V_a).T # n_points, n_neurons, 1
        else:
            self._B =  (self.ansatz.evaluate_model_gradient(self.domain.all_points))
            if svd_on:
                self._B =  (self._B).reshape((self._B.shape[0], -1)) @ (self._V_a).T # n_points, n_neurons, 1
        
        # CD: Calculate the generalized inverse of a matrix using its singular-value decomposition (SVD) and including all large singular values.
        self._A_inv = np.linalg.pinv(self._A, self.regularization_scale)
        
        # Adding labels and title
        """
        plt.semilogy(np.arange(len(S_a)), S_a, label="A")
        U_b, S_b, V_b = np.linalg.svd(self._B.reshape((self._B.shape[0], -1)))
        #U_b, S_b, V_b = np.linalg.svd(self._B)
        plt.semilogy(np.arange(len(S_b)), S_b, label="B")
        plt.ylabel('Singular Values')
        plt.title("Singular Values of data matrix and its derivative")
        # Displaying the legend and the plot
        plt.legend()
        plt.show()
        """
        #print("_A_inv: ", self._A_inv.shape)
        return self
    
    def _get_c0(self):
        '''
        Initial condition of the time-dependent coefficients for the ODE solver
        Returns:
            c0: shape ((k+1)*2, ); initial condition for c (first k+1 entries) and d (= c_t, last k+1 entries)
        '''
        c0 = np.linalg.lstsq(self._A, self.u0(self.domain.all_points), self.regularization_scale)[0]
        return c0