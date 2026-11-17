from .ansatz import Ansatz
from abc import ABC, abstractmethod
from typing import Callable, Union
from swimpde.utils import activations, activations_x, activations_xx, activations_xxx, activations_xxxx, parameter_samplers
from sklearn.pipeline import Pipeline
from swimnetworks import (Dense, Linear)
import numpy as np
from dataclasses import dataclass
import warnings


@dataclass
class BoundaryCompliantAnsatz(Ansatz):
    '''
    Ansatz composed of basis functions that all comply to a boundary condition.

    The individual basis functions have the form \phi_j(x) = \sum{a_ji * \psi_i(w_i*x + b_i)}. The \psi_i can be any activation function and are the same across all \phi_j. 
    The coefficients a_ij are chosen so that the desired boundary condition is fulfilled by all \phi_j.
    \psi_i are referred to as the inner basis, and \phi_j as the outer basis in this code.

    Attributes:
    -----------
    n_outer_basis: int
        number of (outer) basis functions \phi_j
    n_inner_basis: int 
        number of inner basis functions psi_i used to construct the outer basis
    activation: Union[Callable, str]
        activation function to use for the inner basis.
        If the function is not a predefined one (passed as string), the derivatives relevant to the solver and a parameter sampler need to be provided as callable, 
        otherwise they can be automatically deduced.
    activation_x: Union[Callable, str]
        first derivative of the activation function
    activation_xx: Union[Callable, str]
        second derivative of the activation function
    parameter_sampler: str
        parameter sampler to use in the SWIM algorithm (see the SWIM package for possible options)
    target_gen: Union[Callable, str]
        target values on the domain interior for constructing the outer basis. 
        Uses a dense layer to generate target values, if this is one of the known activation functions. 
        Uses all zeros with 10 random indices set to one if this option is 'ones'. 
        Can also be a callable to directly specify the sampling of the target values. 
    domain_margin_percent: float 
        If nonzero, use only interior points with a minimum distance (domain_margin_percent/100 * domain size) from the boundary to determinine the coefficients a_ij.
        Having this buffer zones avoids abrupt jumps for target values close to the boundary and the target values on the boundary.
        Note that the domain size is approximated by taking the largest distance between boundary points in the coordinate directions and may not reflect the actual domain size well for oddly shaped domains.
    random_state: int
        random state to be used in all processes that require randomness to allow reproducibility
    regularization_scale: float
        regularization scale used in least squares problems
    scale_bc: float
        Scaling factor for boundary conditions. (Particularly useful for periodic boundary conditions to ensure that scale_bc * (u_left - r_right) = 0 and similar for neumann BC)
    Note:
    -----
    This ansatz is to be used with time-dependent solvers to automatically ensure the compliance to boundary conditions across all time steps.

    '''

    n_outer_basis: int = 512 
    n_inner_basis: int = 128 

    activation: Union[Callable, str] = "tanh"
    activation_x: Union[Callable, str] = None
    activation_xx: Union[Callable, str] = None
    activation_xxx: Union[Callable, str] = None
    activation_xxxx: Union[Callable, str] = None
    parameter_sampler: str = "tanh"

    target_gen: Union[Callable, str] = "tanh" 
    target_gen_x: Callable = None
    target_gen_xx: Callable = None
    target_gen_xxxx: Callable = None
    domain_margin_percent: float = 5 

    random_state: int = 42
    regularization_scale: float = 1e-13
    scale_bc: float = 1.
    sample_randomly: bool = False

    def __post_init__(self):
        # induce the activation derivatives and sampler in case the activation function is a predefined one
        if isinstance(self.activation, str):
            try:
                if self.activation_x is None:
                    self.activation_x = activations_x[self.activation]
                if self.activation_xx is None:
                    self.activation_xx = activations_xx[self.activation]
                if self.activation_xxx is None:
                    self.activation_xxx = activations_xxx[self.activation]
                if self.activation_xxxx is None:
                    self.activation_xxxx = activations_xxxx[self.activation]

                if self.parameter_sampler is None:
                    self.parameter_sampler = self.activation
                
                self.activation = activations[self.activation]
            except KeyError:
                raise ValueError(f"Unknown activation {self.activation}.")
            
        # try to look up names from known activation functions, if the passed parameters are not callables
        if not isinstance(self.activation, Callable):
            try:
                self.activation = activations[self.activation]
            except KeyError:
                raise ValueError(f"Unknown activation {self.activation}.")
        if not isinstance(self.activation_x, Callable):
            try:
                self.activation_x = activations_x[self.activation_x]
            except KeyError:
                raise ValueError(f"Unknown activation_x {self.activation_x}.")
        if not isinstance(self.activation_xx, Callable):
            try:
                self.activation_xx = activations_xx[self.activation_xx]
            except KeyError:
                raise ValueError(f"Unknown activation_xx {self.activation_xx}.")

        if not isinstance(self.activation_xxx, Callable):
            try:
                self.activation_xxx = activations_xxx[self.activation_xxx]
            except KeyError:
                raise ValueError(f"Unknown activation_xxx {self.activation_xxx}.")

        if not isinstance(self.activation_xxxx, Callable):
            try:
                self.activation_xxxx = activations_xxxx[self.activation_xxxx]
            except KeyError:
                raise ValueError(f"Unknown activation_xxxx {self.activation_xxxx}.")
            
        # generating targets for the outer basis
        if not isinstance(self.target_gen, Callable):
            if self.target_gen in activations.keys():
                self.get_target = self._get_target_dense_layer
            elif self.target_gen == 'ones':
                self.get_target = self._get_target_ones 
            else:
                raise ValueError(f"Unknown target {self.target_gen}.")   

        else:
            self.get_target = self.target_gen   

        # internal model
        self._model = None
    

    def init_model(self, domain, boundary_condition, initial_condition = None):
        '''
        Build the model, constructing the (outer) basis so that it fulfills the boundary condition.

        Parameters:
        domain: Domain
            the domain the solver uses
        boundary_condition: str
            the boundary condition the basis should fulfill
            Can be either "zero neumenn"/"zero derivative" to require the derivative in normal direction to be zero at the boundary,
            or "zero dirichlet"/"zero" to require the basis functions to be zero at the boundary.
        '''
        
        # choose the correct boundary condition
        if not isinstance(boundary_condition, str):
            self._boundary_condition = boundary_condition
        elif boundary_condition.lower() in ["zero derivative", "zero neumann"]:
            if domain.normal_vectors is None:
                raise ValueError("Neumann boundary condition requires the normal_vectors property of the domain to be set")
            self._boundary_condition = ZeroNeumann()
        elif boundary_condition.lower() in ["zero", "zero dirichlet"]:
            self._boundary_condition = ZeroDirichlet()
        elif boundary_condition.lower() in ["non-zero dirichlet"]:
            self._boundary_condition = NonZeroDirichlet()
        elif boundary_condition.lower() in ["zero dirichlet second order"]:
            self._boundary_condition = ZeroDirichletSecondOrder()
        elif boundary_condition.lower() in ["periodic"]:
            self._boundary_condition = Periodic()
        elif boundary_condition.lower() in ["periodic strict"]:
            self._boundary_condition = PeriodicStrict()
        else:
            raise ValueError(f"Unknown boundary condition {boundary_condition}.")
        
        # determine the core interior points with a distance of at least (domain_margin_percent/100 * domain size) to the boundary
        if self.domain_margin_percent > 0:
            max_domain_size = np.max(np.max(domain.interior_points, axis=0) - np.min(domain.interior_points, axis=0))
            min_dist = self.domain_margin_percent / 100. * max_domain_size
            core_interior_points = []
            sol_core_int_points = []
            i = 0
            for x_i in domain.interior_points:
                far_enough = True
                for x_b in domain.boundary_points:
                    if np.linalg.norm(x_i - x_b) < min_dist:
                        far_enough = False
                        break
                if far_enough:
                    core_interior_points.append(x_i)
                    if initial_condition is not None:
                        sol_core_int_points.append(initial_condition[i])
                i = i + 1
            core_interior_points = np.array(core_interior_points)
            if initial_condition is not None:
                sol_core_int_points = np.array(sol_core_int_points)
            else:
                sol_core_int_points = None
        else:
            core_interior_points = domain.interior_points
            sol_core_int_points = initial_condition
            # initial_condition
            

        # build the targets for inner and outer basis
        #target_inner_basis_values = self.get_target(domain.interior_points, self.n_outer_basis, initial_condition=initial_condition) # no. of interior points * n_OBF
        target_inner_basis_values = self.get_target(domain.interior_points, self.n_inner_basis, initial_condition=initial_condition) # no. of interior points * n_OBF
        #target_outer_basis_values = self.get_target(core_interior_points, self.n_outer_basis, initial_condition=initial_condition) # no. of core interior points * n_OBF
        target_outer_basis_values = self.get_target(core_interior_points, self.n_outer_basis, initial_condition=sol_core_int_points) # no. of core interior points * n_OBF

        # Parameter sampler for the inner basis, drawing points in the domain without replacement to avoid duplicates
        #TODO fix
        def sample_no_replacement(x, y, rng):
            # only consider cases where the number of x pairs is >= the number of required samples!!
            # also, inefficient dummy implementation
            # also, only for tanh sampling rn
            n = x.shape[0]
            sx = np.arange(n)
            sy = np.arange(n)
            sxx, syy = np.meshgrid(sx, sy)
            # Stack the combinations and then filter out the pairs where x = y
            pairs = np.stack([sxx.ravel(), syy.ravel()], axis=1)
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]

            selected_pairs = rng.choice(pairs,
                                  size=self.n_inner_basis,
                                  replace=False)
            idx_from = selected_pairs[:, 0]
            idx_to = selected_pairs[:, 1]
            directions = x[idx_to, ...] - x[idx_from, ...]
            dists = np.linalg.norm(directions, axis=1, keepdims=True)**2
            dists = np.clip(dists, a_min=1e-10, a_max=None)

            scale = 0.5 * (np.log(1 + 1/2) - np.log(1 - 1/2))
            weights = (2 * scale * directions / dists).T
            biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1) - scale

            return weights, biases, idx_from, idx_to


        # inner basis functions \psi
        if self.parameter_sampler == 'tanh':
            param_sampler = sample_no_replacement #"tanh" 
        else:
            param_sampler = self.parameter_sampler
        
        inner_basis = Dense(
            layer_width=self.n_inner_basis,
            activation=self.activation,
            parameter_sampler= param_sampler,
            sample_uniformly=False,
            random_seed=self.random_state + 42,
            prune_duplicates=True
        )
        # CD: Why do we need this?? (Figured out) --> In get target, we don't assign the unifromly sampled weights and biases to the inner_basis object!
        # inner_basis.fit(domain.interior_points, target_inner_basis_values)
        inner_basis.fit(domain.sample_points, target_inner_basis_values)
        # To find: weights (n_IBF), from: target_inner_basis_values (n_int * n_OBF), interior points = (n_int, )

        # Construct the linear system for the outer weights
        basis_space = inner_basis.transform(core_interior_points) #CD: n_core * n_IBF
        
        #CD: n_boundary * n_IBF (E.g For 1d problem 2 * n_IBF)
        basis_space_bc = self._boundary_condition.get_basis_space_bc(inner_basis, domain, self.activation, self.activation_x, self.scale_bc) #292 * 4096
        #basis_space_bc = self._boundary_condition.get_basis_space_bc(inner_basis, domain, self.activation, self.activation_x, self.activation_xx, self.scale_bc) #292 * 4096

        matrix_in = np.row_stack([
            basis_space,
            basis_space_bc
        ])

        if isinstance(self._boundary_condition, ZeroDirichletSecondOrder):
            boundary_bias_value = self._boundary_condition.get_bias_value(int(basis_space_bc.shape[0]/2)) # Pass the shape to get bias values as 1 for u and 0 for higher order derivatives
        else:
            boundary_bias_value = self._boundary_condition.get_bias_value()

        if isinstance(self._boundary_condition, Periodic) or isinstance(self._boundary_condition, PeriodicStrict):
            # just fit the given periodic target values
            linear_layer = Linear(regularization_scale=self.regularization_scale)
            # linear_layer.fit(inner_basis.transform(domain.all_points), self.get_target(domain.all_points, self.n_outer_basis))
            ######################################################################################################################
            # take derivative into account
            
            matrix_a = inner_basis.transform(domain.all_points)

            inner_basis.activation = self.activation_x
            matrix_b = np.stack([
                (inner_basis.transform(domain.all_points)
                * inner_basis.weights[d, :]) 
                for d in range(domain.all_points.shape[1])
            ], axis=-1).squeeze() # squeeze only works for 1d

            inner_basis.activation = self.activation_xx
            matrix_c = (inner_basis.transform(domain.all_points) * np.linalg.norm(inner_basis.weights, axis=0, keepdims=True)**2)
            
            inner_basis.activation = self.activation_xxxx
            matrix_d = (inner_basis.transform(domain.all_points) * np.linalg.norm(inner_basis.weights, axis=0, keepdims=True)**4)

            inner_basis.activation = self.activation

            scale_zeroth = 1e0
            scale_first = 1e-4
            scale_second = 1e-6
            scale_fourth = 1e-8

            linear_layer.fit(
                np.concatenate(
                    [scale_zeroth * matrix_a,
                    scale_first * matrix_b,
                    scale_second * matrix_c,
                    scale_fourth * matrix_d],
                    axis=0
                ),
                np.concatenate(
                    [scale_zeroth * self.get_target(domain.all_points, self.n_outer_basis),
                    scale_first * self.target_gen_x(domain.all_points, self.n_outer_basis),
                    scale_second * self.target_gen_xx(domain.all_points, self.n_outer_basis),
                    scale_fourth * self.target_gen_xxxx(domain.all_points, self.n_outer_basis)],
                    axis=0
                )
            )
            # ###############################################################
            # # sanity check
            # dumpling_a = np.concatenate(
            #     [scale_zeroth * matrix_a,
            #     scale_first * matrix_b,
            #     scale_second * matrix_c,
            #     scale_fourth * matrix_d],
            #     axis=0
            # )
            # dumpling_b = np.concatenate(
            #     [scale_zeroth * self.get_target(domain.all_points, self.n_outer_basis),
            #     scale_first * self.target_gen_x(domain.all_points, self.n_outer_basis),
            #     scale_second * self.target_gen_xx(domain.all_points, self.n_outer_basis),
            #     scale_fourth * self.target_gen_xxxx(domain.all_points, self.n_outer_basis)],
            #     axis=0
            # )
            # dumpling = np.linalg.lstsq(dumpling_a,
            #     dumpling_b,
            #     rcond=self.regularization_scale
            # )
            # print("basis functions residual", dumpling[1])
            # print("my squared norm error", np.linalg.norm(dumpling_b - dumpling_a @ dumpling[0], axis=0)**2)
            # print("shape", dumpling_a.shape, dumpling_b.shape, dumpling[0].shape)

        else:
            matrix_bias = np.row_stack([
                np.ones((basis_space.shape[0],1)),
                np.ones((basis_space_bc.shape[0],1)) * boundary_bias_value
            ])
            matrix_in = np.column_stack([matrix_in,matrix_bias])

            matrix_out = np.row_stack([
                target_outer_basis_values, # CD: no. of core interior points * n_OBF
                #self._boundary_condition.get_target_bc(domain.boundary_points.shape[0], target_outer_basis_values.shape[1])
                self._boundary_condition.get_target_bc(basis_space_bc.shape[0], target_outer_basis_values.shape[1])
            ])

            # construct the linear layer by solving the linear system for its weights
            weights_biases = np.linalg.lstsq(matrix_in, matrix_out, rcond=self.regularization_scale)[0]

            # CD: Why do we need this? Do"t we do this above when we set the bias in matrix_in?
            # add a constant basis function (to outer basis functions) if the boundary condition allows it (not possible for Dirichlet conditions because it will destroy the zeros on the boundary of the other basis functions. This replaces the constant time-dependent coefficient in the time-dependent PDEs)
            if self._boundary_condition.satisfied_by_constant():
                weights_biases = np.column_stack([weights_biases, np.zeros(weights_biases.shape[0])])
                weights_biases[-1, -1] = 1

            # assemble the model
            linear_layer = Linear(regularization_scale=self.regularization_scale)
            # linear_layer.fit(np.row_stack([domain.boundary_points, core_interior_points]), np.zeros((np.row_stack([domain.boundary_points, core_interior_points]).shape[0],1))) # fit for initialization
            linear_layer.fit(domain.interior_points[:2, ...], np.zeros((2, 1))) # fit for initialization
            linear_layer.weights = weights_biases[:-1, :]
            linear_layer.biases = weights_biases[-1:, :]

        self._model = Pipeline([("base-dense", inner_basis), ("base-linear", linear_layer)])

        # now also map the functions to a basis we can use for integration, i.e. with good condition number
        #_, singular_values, v_matrix = np.linalg.svd(self._model.transform(domain.interior_points), full_matrices=False)
        #idx_s = singular_values / np.max(singular_values) > self.svd_cutoff
        #v_matrix = v_matrix[idx_s, ...]
        #self._model[1].weights = self._model[1].weights @ v_matrix.T
        #self._model[1].biases = self._model[1].biases @ v_matrix.T

        # ---------------- DEBUG BELOW --------------------------------------------------------
        """
        rank2 = np.linalg.matrix_rank(matrix_in)
        matrix_out_model = np.row_stack([
            self._model.transform(core_interior_points),
            self._model.transform(domain.boundary_points)
        ])

        #diff = matrix_out - matrix_out_model
        # x = 1
        print("rank of lstsq: ", rank) # Rank of least squares
        print("rank2: ", rank2) # Rank of the data matrix
        print("dimension: ", matrix_in.shape)
        wb = np.row_stack((inner_basis.weights, inner_basis.biases))
        unique_num = np.unique(wb.T, axis=0).shape[0]
        print("number of unique (inner basis) weight/bias pairs: ", unique_num)

        self._debug_target_outer = target_outer_basis_values
        self._debug_target_inner = target_inner_basis_values
        self._debug_core_points = core_interior_points

        #sol, residuals, rank2, s = np.linalg.lstsq(matrix_in, matrix_out, rcond=self.regularization_scale)
        condition = np.linalg.cond(matrix_in)
        print('condition number of matrix_in: ', condition)
        x = 1
        """
 
    def evaluate_model(self, x):
        '''
        Evaluate the model.

        input shape: (n_points, d)
        output shape: (n_points, n_neurons)
            n_neurons is either n_outer_basis if no constant basis was added (dirichlet) or n_outer_basis + 1 including the constant
        '''
        self._model.steps[0][1].activation = self.activation
        phi = self._model.transform(x)    
        return phi

    def evaluate_model_gradient(self, x):
        '''
        Evaluate the gradient of the model.

        input shape: (n_points, d)
        output shape: (n_points, n_neurons, d)
        '''
        self._model.steps[0][1].activation = self.activation_x
        phi_x = np.stack([
            (self._model.steps[0][1].transform(x)
            * self._model.steps[0][1].weights[d, :]) 
            @ self._model.steps[1][1].weights
            for d in range(x.shape[1])
        ], axis=-1)    
        return phi_x

    def evaluate_model_laplace(self, x):
        '''
        Evaluate the laplace operator applied to the model.

        input shape: (n_points, d)
        output shape: (n_points, n_neurons)
        '''
        self._model.steps[0][1].activation = self.activation_xx
        return (self._model.steps[0][1].transform(x) * np.linalg.norm(self._model.steps[0][1].weights, axis=0, keepdims=True)**2) @ self._model.steps[1][1].weights


    def evaluate_model_fourth_order_diff(self, x):
        '''
        Evaluate the laplace operator applied to the model.

        input shape: (n_points, d)
        output shape: (n_points, n_neurons)
        '''
        self._model.steps[0][1].activation = self.activation_xxxx
        return (self._model.steps[0][1].transform(x) * np.linalg.norm(self._model.steps[0][1].weights, axis=0, keepdims=True)**4) @ self._model.steps[1][1].weights



    def fit_model(self, x, y):
        self._model.steps[0][1].activation = self.activation
        warnings.warn("Using 'fit'  a boundary condition ansatz has no effect. The ansatz is designed to meet certain boundary conditions and interacts with the PDE only through the external weights.")
    
    def fit_model_laplace(self, x, y):
        self._model.steps[0][1].activation = self.activation_xx
        warnings.warn("Using 'fit' on a boundary condition ansatz has no effect. The ansatz is designed to meet certain boundary conditions and interacts with the PDE only through the external weights.")

   
    def _get_target_dense_layer(self, x, k, initial_condition=None):
        '''
        Generate a target using a dense layer with uniformly sampled weights.
        Uses self.target_gen as activation function

        Parameters:
        x: domain points used to fit the dense layer, shape (n_points, d)
        k: number of neurons used in the dense layer

        Returns:
        target values at the points x, shape (n_points, k)! Verified!
        '''
        if initial_condition is None:
            sample_uniformly = True
            target = np.zeros((x.shape[0],1)) # Not used , only required as an argument!
        else:
            sample_uniformly = False
            target = initial_condition
        gen_layer = Dense(layer_width=k, activation=activations[self.target_gen], parameter_sampler=parameter_samplers[self.target_gen], 
                            sample_uniformly=sample_uniformly, random_seed=self.random_state + 4)
        gen_layer.fit(x, target)
        #gen_layer.fit(x, np.zeros((x.shape[0],1)))
        return gen_layer.transform(x)

    def _get_target_ones(self, x, k):
        '''
        Generate a set of target values consisting of zeros with 10 random indices set to 1 for each target

        Parameters:
        x: domain points for which to create target values, shape (n_points, d)
        k: number of targets to generate

        Returns:
        target values at the points x, shape  (n_points, k)
        '''
        functions = []
        rng = np.random.default_rng(self.random_state + 7254)
        for _ in range(k):
            f_target = np.zeros((x.shape[0], 1))
            f_target[rng.integers(low=0, high=f_target.shape[0], size=10)] = 1
            functions.append(f_target)
        return np.column_stack(functions)


class BoundaryCondition(ABC):
    '''
    Base class for boundary conditions used by the BoundaryCompliantAnsatz
    '''

    @abstractmethod
    def get_basis_space_bc(self, basis, domain, activation, activation_x, scale_bc):
        '''
        Transform the boundary points according to the boundary condition
        to generate the lhs of the linear system to satisfy the boundary condition
        '''
        pass

    @abstractmethod
    def get_bias_value(self):
        '''
        Return the value multiplied with the constant coefficient of the constructed (outer) basis functions.
        1 for Dirichlet to preserve the constant, and 0 for Neumann because it vanishes in the derivative.
        '''
        pass
    
    @abstractmethod
    def get_target_bc(self, n_bc_points, n_outer_basis):
        '''
        Return the target value of the (outer) basis or its derivative at the boundary
        to generate the rhs of the linear system to satisfy the boundary condition
        '''
        pass

    @abstractmethod
    def satisfied_by_constant(self):
        '''
        Return true if the boundary condition is compatible with adding a constant (outer) basis function,
        i.e. the constant function satisfies the boundary condition
        '''
        pass


class ZeroNeumann(BoundaryCondition):
    '''
    Zero Neumann boundary condition
    '''
        
    def get_basis_space_bc(self, basis, domain, activation, activation_x, scale_bc = 1):
        '''
        Evaluate the derivative of basis functions evaluated in normal direction at the boundary
        '''
        basis.activation = activation_x
        basis_space_boundary_x = basis.transform(domain.boundary_points)
        basis_space_x = np.stack([
            basis_space_boundary_x * basis.weights[k_dim, :]
            for k_dim in range(domain.n_dim)
        ])
        gradients = []
        for k in range(domain.boundary_points.shape[0]):
            gradients.append(domain.normal_vectors[k,:] @ basis_space_x[:, k, :])
        basis_space_x = np.row_stack(gradients)
        return basis_space_x
    
    def get_bias_value(self):
        return 0

    def get_target_bc(self, n_bc_points, n_outer_basis):
        '''
        Target value for the derivatives of the (outer) basis at the boundary
        '''
        return np.zeros((n_bc_points, n_outer_basis))
    
    def satisfied_by_constant(self):
        return True
    

class ZeroDirichlet(BoundaryCondition):
    '''
    Zero Dirichlet boundary condition
    '''

    def get_basis_space_bc(self, basis, domain, activation, activation_x, scale_bc = 1):
        '''
        Evaluate the basis functions at the boundary
        '''
        basis.activation = activation
        basis_space_boundary = basis.transform(domain.boundary_points)
        return basis_space_boundary
    
    def get_bias_value(self):
        return 1

    def get_target_bc(self, n_bc_points, n_outer_basis):
        '''
        Target values for the outer basis functions at the boundary
        '''
        return np.zeros((n_bc_points, n_outer_basis))
    
    def satisfied_by_constant(self):
        return False
    

class NonZeroDirichlet(BoundaryCondition):
    '''
    Non-Zero Dirichlet boundary condition
    '''

    def get_basis_space_bc(self, basis, domain, activation, activation_x, scale_bc = 1):
        '''
        Evaluate the basis functions at the boundary
        '''
        basis.activation = activation
        basis_space_boundary = basis.transform(domain.boundary_points)
        return basis_space_boundary
    
    def get_bias_value(self):
        return 1

    def get_target_bc(self, n_bc_points, n_outer_basis):
        '''
        Target values for the outer basis functions at the boundary
        '''
        return np.zeros((n_bc_points, n_outer_basis))
    
    def satisfied_by_constant(self):
        return False

class ZeroDirichletSecondOrder(BoundaryCondition):
    '''
    Zero Dirichlet boundary condition
    '''

    def get_basis_space_bc(self, basis, domain, activation, activation_x, activation_xx, scale_bc = 1):
        '''
        Evaluate the basis functions at the boundary
        '''
        #basis.activation = activation
        #basis_space_boundary = basis.transform(domain.boundary_points)
        #return basis_space_boundary
    
        basis.activation = activation
        basis_space_boundary = basis.transform(domain.boundary_points)
        #u_bc = scale_bc * (basis_space_boundary[0] - basis_space_boundary[1]).reshape(1, -1)
        # diff_dirichlet = scale_bc * (basis_space_boundary[0] - basis_space_boundary[1]).reshape(1, -1)
        # * np.linalg.norm(self._model.steps[0][1].weights, axis=0, keepdims=True)**2) 
        basis.activation = activation_xx
        basis_space_boundary_xx = basis.transform(domain.boundary_points)
        basis_space_xx = np.stack([
            basis_space_boundary_xx * np.linalg.norm(basis.weights[k_dim, :], axis=0, keepdims=True)**2 #basis.weights[k_dim, :]
            for k_dim in range(domain.n_dim)
        ])
        basis_space_xx = basis_space_xx.reshape((basis_space_xx.shape[1], basis_space_xx.shape[2]))
        """
        u_xx_bc = []
        for k in range(domain.boundary_points.shape[0]):
            u_xx_bc.append(basis_space_xx[:, k, :])#domain.normal_vectors[k,:] @ 
        basis_space_xx = np.row_stack(u_xx_bc)
        """

        # Left ounda
        return np.vstack((scale_bc * basis_space_boundary, scale_bc * basis_space_xx)) #diff # shape(1, 1)
    
    def get_bias_value(self, n_bc_points):
        b_u = np.ones((n_bc_points, 1))
        b_u_xx = np.zeros((n_bc_points, 1)) 
        return np.vstack((b_u, b_u_xx))

    def get_target_bc(self, n_bc_points, n_outer_basis):
        '''
        Target values for the outer basis functions at the boundary
        '''
        return np.zeros((n_bc_points, n_outer_basis))
    
    def satisfied_by_constant(self):
        return False


class Periodic(BoundaryCondition):
    '''
    Periodic boundary condition
    '''
        
    def get_basis_space_bc(self, basis, domain, activation, activation_x, scale_bc = 1):
        '''
        Evaluate the basis functions at the boundary and compute the difference of values at the boundary
        '''
        basis.activation = activation
        basis_space_boundary = basis.transform(domain.boundary_points)
        diff = scale_bc * (basis_space_boundary[0] - basis_space_boundary[1]).reshape(1, -1)
        # Left ounda
        return diff # shape(1, 1)
    
    
    def get_bias_value(self):
        return 0

    def get_target_bc(self, n_bc_points, n_outer_basis):
        '''
        Target value for the derivatives of the (outer) basis at the boundary
        '''
        return np.zeros((n_bc_points, n_outer_basis))
    
    def satisfied_by_constant(self):
        return True



class PeriodicStrict(BoundaryCondition):
    '''
    Periodic boundary condition
    '''
        
    def get_basis_space_bc(self, basis, domain, activation, activation_x, scale_bc = 1):
        '''
        Evaluate the basis functions at the boundary and compute the difference of values at the boundary
        '''
        basis.activation = activation
        basis_space_boundary = basis.transform(domain.boundary_points)
        diff_dirichlet = scale_bc * (basis_space_boundary[0] - basis_space_boundary[1]).reshape(1, -1)

        basis.activation = activation_x
        basis_space_boundary_x = basis.transform(domain.boundary_points)
        basis_space_x = np.stack([
            basis_space_boundary_x * basis.weights[k_dim, :]
            for k_dim in range(domain.n_dim)
        ])
        gradients = []
        for k in range(domain.boundary_points.shape[0]):
            gradients.append(basis_space_x[:, k, :])#domain.normal_vectors[k,:] @ 
        basis_space_x = np.row_stack(gradients)
        
        diff_neumann = scale_bc * (basis_space_x[0] - basis_space_x[1]).reshape(1, -1)
        diff = np.vstack((diff_dirichlet, diff_neumann))

        # Left ounda
        return diff # shape(1, 1)
    
    
    def get_bias_value(self):
        return 1

    def get_target_bc(self, n_bc_points, n_outer_basis):
        '''
        Target value for the derivatives of the (outer) basis at the boundary
        '''
        return np.zeros((n_bc_points, n_outer_basis))
    
    def satisfied_by_constant(self):
        return True