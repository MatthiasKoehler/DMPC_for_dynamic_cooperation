""" Auxiliary methods:
Define several methods used in the code of distributed MPC for dynamic cooperation.
"""
import casadi as cas
import numpy as np
import warnings
import matplotlib.pyplot as plt
import scipy
import cvxpy
import dill
from datetime import datetime
import gurobipy as gp
from concurrent.futures import ThreadPoolExecutor
import os


class Agent:
    """A discrete-time system with individual states and inputs, i.e. without dynamic coupling to other agents.
            
    Attributes:
    - id (int): Integer identifier. 
    - state_dim (int): Dimension of the state.
    - input_dim (int): Dimension of the input.
    - dynamics (casadi function): Discrete-time dynamics of the agent, i.e. x+ = f(x,u).
    - output_dim (int): Dimension of the output.
    - output_map (casadi function): Output map of the agents, i.e. y = h(x,u).
    - average_output_map (casadi function): Output map defining a different output used for averaging.
    - current_state (narray): Current state of the agent. Stored as a one-dimensional numpy array.
    - current_average_value (narray): Current average value of the agent.
    - t (int): Current time step associated with the current state of the agent.
    - neighbours (list of agents): Neighbours of the agent. 
    - stage_cost (casadi function): Function defining the stage cost (for tracking) of the agent which takes 'x', 'u', 'xT', 'uT' and return 'l', i.e. l(x, u, xT, uT), which is either called in that order or using these names,
    e.g. stage_cost(x=x[0:n], u=u[0:q], xT=xT[0:n], uT=uT[0:q])['l']
    - state_constraints (dict): Containing linear inequality constraints (A*x <= b with keys 'A' and 'b'), equality constraints (Aeq*x <= b with keys 'Aeq' and 'beq')
    - input_constraints (dict): Containing linear inequality constraints (A*u <= b with keys 'A' and 'b'), equality constraints (Aeq*u <= b with keys 'Aeq' and 'beq')
    - cooperation_constraints (dict): Containing constraints on the cooperation output and reference state and input.
        - 'Ay' (np.array): Matrix defining a pointwise-in-time affine constraint on the cooperation output, i.e. Ay@yT(k) <= by.
        - 'by' (np.array): Vector defining a pointwise-in-time affine constraint on the cooperation output, i.e. Ay@yT(k) <= by.
        - 'Ax' (np.array): Matrix defining a pointwise-in-time affine constraint on the cooperation state, i.e. Ax@xT(k) <= bx.
        - 'bx' (np.array): Vector defining a pointwise-in-time affine constraint on the cooperation state, i.e. Ax@xT(k) <= bx.
        - 'Au' (np.array): Matrix defining a pointwise-in-time affine constraint on the cooperation input, i.e. Au@uT(k) <= bu.
        - 'bu' (np.array): Vector defining the constraint on the cooperation input, i.e. Au@uT <= bu.
        - 'function' (cas.Function): Function that takes the cooperation decision variables and returns the constraint.
        - 'upper_bound' (list[np.array]): List containing the upper bounds for the constraint.
        - 'lower_bound' (list[np.array]): List containing the lower bounds for the constraint.
    - data (dict): Containing arbitrary data, e.g. trajectories.
    - coupling_constraints (list[casadi.Function]): Containing coupling constraints with neighbours defined as casadi.Functions. The constraint is defined such that this function is smaller than or equal to zero. Define coupling constraint pointwise-in-time.
    - nonlinear_constraints (list[casadi.Function]): Defining nonlinear constraints local to the agent.
    
    Class variables:
    - _counter (int): This is a class-level counter that increments each time an instance is generated and is used to assign the id of the agent.
    """
    
    _counter = 0  # class-level counter
    
    def __init__(self, id = None, dynamics = None, output_map = None, initial_time = 0, initial_state = None, neighbours = None, box_state_constraints=None, box_input_constraints=None, stage_cost=None, average_output_map=None, current_average_value=None, offset_cost=None, nonlinear_constraints=None, data=None):
        """
        Initialise an agent.
        
        Args:
        - dynamics (casadi Function): vector field of the system. (default is None)
        - output_map (casadi Function): function from state and input to output, i.e. y = h(x,u) (default output is outputs = states)
        - initial_time (int): initial time for internal time-keeping. (default 0)
        - initial_state (numpy array): initial state at the initial time. (default 0)
        - neighbours (list with agents): neighbours of this agent. (default None)
        - box_state_constraints (array): Contains a lower (first column) and upper bound (second column) for each state variable (rows).
        - box_input_constraints (array): Contains a lower (first column) and upper bound (second column) for each input variable (rows).
        - stage_cost (casadi Function): function from state and input to reals.
        - average_output_map (casadi Function): function from state and input to output used in average constraints. (default None)
        - current_average_value (array): initial value of the value that should satisfy the average constraint. (default None)
        - offset_cost (casadi Function): function penalising deviation (offset) from a desirable reference (default None)
        - nonlinear_constraints (list[casadi.Function]): Defining nonlinear constraints local to the agent. 
            Currently, only pointwise-in-time constraints on the state are allowed.
            The input must be named 'x' and the output 'g'. The constraint should be non-positive if and only if the state is feasible.
        """
        
        # Warn if id is provided, since this has been deprecated.
        if id is not None:
            warnings.warn(
                "The 'id' parameter is deprecated and will be ignored. "
                "Unique IDs are generated automatically.",
                DeprecationWarning
            )
        
        self.dynamics = dynamics
        
        if dynamics is not None:
            # Set values as provided.
            for var_name in dynamics.name_in():
                if var_name not in ['x', 'u']:
                    raise ValueError(f"Unknown input variable '{var_name}' in dynamics function.")
            if dynamics.name_out() != ['x+']:
                raise ValueError("The dynamics function should have a single output (vector) 'x+'.")
            if dynamics.size_in('x') != dynamics.size_out('x+'):
                raise ValueError("The dynamics function should have the same input and output dimension.")
            if dynamics.size_in('x')[1] != 1:
                raise ValueError("The state entering the dynamics function should be a column vector.")
            if dynamics.size_in('u')[1] != 1:
                raise ValueError("The input entering the dynamics function should be a column vector.")
            
            self.state_dim = dynamics.size_in('x')[0]
            self.input_dim = dynamics.size_in('u')[0]
            
            # Define a symbolic state, input and output.
            self._state = cas.SX.sym('x', self.state_dim)
            self._input = cas.SX.sym('u', self.input_dim)
            
        # Assign an id.
        Agent._counter += 1
        self.id = Agent._counter
        
        # Define the output map.
        if output_map is None and dynamics is not None:
            self.output_map = cas.Function('h', [self._state, self._input], [self._state], ['x', 'u'], ['y'])
            self._output = cas.SX.sym('y', self.output_dim)
        elif output_map is not None:
            self.output_map = output_map
            self._output = cas.SX.sym('y', self.output_dim)

        # Set constraints of the agent. If no box constraints are passed, the constraints are initialised to be empty.
        self.set_box_state_constraints(box_state_constraints=box_state_constraints)
        self.set_box_input_constraints(box_input_constraints=box_input_constraints) 
        
        # Initialize dictionaries for constraints on the cooperation output, state, and input.
        self.cooperation_constraints = {'Ay': None, 'by': None, 'Ax': None, 'bx': None, 'Au': None, 'bu': None, 'function': None, 'upper_bound': None, 'lower_bound': None}
        
        # Set the initial time (internal clock) and the initial state.
        self.t = initial_time
        self.current_state = initial_state
        
        if neighbours is None:    
            self.neighbours = []
        else:
            self.neighbours = neighbours
            
        self.stage_cost = stage_cost
        self.offset_cost = offset_cost
        self.average_output_map = average_output_map
        if current_average_value is not None:
            self.current_average_value = current_average_value
            
        self.nonlinear_constraints = nonlinear_constraints
        self.coupling_constraints = None
        
        if data is None:
            self.data = {}  # Initialise an empty dictionary.

    def __str__(self):
        return "Agent " + str(self.id)
    
    def __repr__(self):
        return "Agent() " + str(self.id)
    
    @property
    def cooperation_constraints(self):
        return self._cooperation_constraints

    @cooperation_constraints.setter
    def cooperation_constraints(self, constraints):
        required_keys = {'Ay', 'by', 'Ax', 'bx', 'Au', 'bu', 'function', 'upper_bound', 'lower_bound'}
        if not isinstance(constraints, dict):
            raise TypeError("cooperation_constraints must be a dictionary.")
        if not required_keys.issubset(constraints.keys()):
            missing_keys = required_keys - constraints.keys()
            raise ValueError(f"cooperation_constraints is missing required keys: {missing_keys}")
        self._cooperation_constraints = constraints
    
    @property
    def id(self):
        return self._id
    @id.setter
    def id(self, id_value):
        """Set the ID which is an integer."""
        if isinstance(id_value, int):
            self._id = id_value
        else:
            raise TypeError("Please specify an integer as the ID.")
        
    @property
    def output_map(self):
        return self._output_map
    @output_map.setter
    def output_map(self, value):
        size_out = value.size_out('y')
        if size_out[1] > 1:
            raise ValueError('The output map should return a column vector.')
        self._output_map = value

    @property
    def output_dim(self):
        return self._output_map.size_out('y')[0]
    @output_dim.setter
    def output_dim(self, value):
        warnings.warn('output_dim is determined by output_map and cannot be set manually.', UserWarning)    
        
    @property
    def current_state(self):
        return self._current_state
    @current_state.setter 
    def current_state(self, state):
        if state is not None:
            if not (isinstance(state, np.ndarray) or isinstance(state, cas.DM)):
                raise TypeError("State must be a numpy array or casadi.DM.")
            # Check the state dimension.
            # The state should be saved as a two-dimensional array, even if a one-dimensional would suffice. It will be transformed into a column vector.
            # This increases compatibility with casadi's DM type, which only knows two-dimensional shapes.
            if len(state.shape) == 1 and state.shape[0] == self.state_dim:
                raise ValueError(f"State assignment failed. One-dimensional arrays are not allowed. Received: {state.shape}; expected: ({self.state_dim}, 1) or (1, {self.state_dim}).")
            elif state.shape == (self.state_dim, 1):
                self._current_state = state
            elif state.shape == (1, self.state_dim):
                self._current_state = state.T
            else:
                raise ValueError(f"State assignment failed, e.g. due to wrong dimensions. Received: {state.shape}; expected: ({self.state_dim}, 1) or (1, {self.state_dim}).")
        else:
            self._current_state = None
    @current_state.deleter
    def current_state(self):
        raise AttributeError("Do not delete, set state to 0.")
    
    @property
    def current_reference(self):
        return self._current_reference
    @current_reference.setter
    def current_reference(self, reference):
        # TODO: validation.
        self._current_reference = reference 
        
    @property
    def coupling_constraints(self):
        return self._coupling_constraints
    @coupling_constraints.setter
    def coupling_constraints(self, cstr_list):
        if cstr_list is not None and not isinstance(cstr_list, list):
            raise TypeError("coupling_constraints must be a list of dictionaries.") 
        if cstr_list is not None:
            for cstr in cstr_list:
                if not isinstance(cstr, cas.Function):
                    raise TypeError("The 'function' value must be a casadi.Function.")
        self._coupling_constraints = cstr_list
    
    @property
    def current_average_value(self):
        return self._current_average_value
    @current_average_value.setter
    def current_average_value(self, average_value):
        # TODO: validation.
        if np.shape(average_value)[0] == np.size(average_value):
            self._current_average_value = average_value
        else:
            raise AttributeError("The current average value needs to be a column vector.")
            
    def set_box_state_constraints(self, box_state_constraints=None):
        """
        Set constraints of the agents.
        
        Keyword arguments:
        - box_state_constraints (list): Contains a lower (first entry, 'lb') and upper bound (second entry, 'ub') for the state vector ('x'), i.e. lb <= x <= ub element-wise.
        """
        # Define state constraints.
        self.state_constraints = {"A": np.empty, "b": np.empty}
        if box_state_constraints is not None:
            self.state_constraints["A"] = np.vstack((-np.eye(self.state_dim), np.eye(self.state_dim)))
            self.state_constraints["b"] = np.vstack((-box_state_constraints[0]*np.ones((self.state_dim,1)), box_state_constraints[1]*np.ones((self.state_dim,1))))
            
    def set_box_input_constraints(self, box_input_constraints=None):
        """
        Set box input constraints for the agent.
        
        Keyword arguments:
        - box_input_constraints (list): Contains a lower (first entry, 'lb') and upper bound (second entry, 'ub') for the input vector ('u'), i.e. lb <= u <= ub element-wise.
        """
        # Define input constraints.
        self.input_constraints = {"A": np.empty, "b": np.empty}
        if box_input_constraints is not None:
            self.input_constraints["A"] = np.vstack((-np.eye(self.input_dim), np.eye(self.input_dim)))
            self.input_constraints["b"] = np.vstack((-box_input_constraints[0]*np.ones((self.input_dim,1)), box_input_constraints[1]*np.ones((self.input_dim,1))))


class Quadrotor(Agent):
    """
    A quadrotor agent with a 10-dimensional state and 3-dimensional input.

    State:
    - z1, z2, z3: Position in x, y, z (m)
    - theta: Pitch angle (rad)
    - phi: Roll angle (rad)
    - v1, v2, v3: Linear velocities in x, y, z (m/s)
    - omega_theta, omega_phi: Angular velocities (rad/s)

    Input:
    - u_theta: Pitch control input (arbitrary units)
    - u_phi: Roll control input (arbitrary units)
    - u_thrust: Vertical thrust input (arbitrary units)

    Output:
    - z1, z2, z3: Position in space

    Attributes:
    - h (float): Sampling time.
    - g (float): Gravitational acceleration.
    - d_phi (float): Damping for angular displacements.
    - d_omega (float): Damping for angular velocity feedback.
    - k_motor (float): Motor gain from input to angular acceleration.
    - k_thrust (float): Thrust coefficient.
    - method (str): Discretization method ('RK4', 'RK2', 'Euler').
    """

    def __init__(self, h, g=9.81, d_phi=8.0, d_omega=10.0, k_motor=10.0, k_thrust=0.91, method='Euler'):
        self.h = h
        self.g = g
        self.d_phi = d_phi
        self.d_omega = d_omega
        self.k_motor = k_motor
        self.k_thrust = k_thrust
        self.method = method

        x = cas.MX.sym('x', 10)  # State: [z1, z2, z3, theta, phi, v1, v2, v3, omega_theta, omega_phi]
        u = cas.MX.sym('u', 3)   # Input: [u_theta, u_phi, u_thrust]

        def f(x, u):
            z1, z2, z3 = x[0], x[1], x[2]
            theta, phi = x[3], x[4]
            v1, v2, v3 = x[5], x[6], x[7]
            omega_theta, omega_phi = x[8], x[9]

            u_theta, u_phi, u_thrust = u[0], u[1], u[2]

            return cas.vertcat(
                v1,                                          # z1_dot
                v2,                                          # z2_dot
                v3,                                          # z3_dot
                -self.d_phi * theta + omega_theta,           # theta_dot
                -self.d_phi * phi + omega_phi,               # phi_dot
                self.g * cas.tan(theta),                     # v1_dot
                self.g * cas.tan(phi),                       # v2_dot
                -self.g + self.k_thrust * u_thrust,          # v3_dot
                -self.d_omega * theta + self.k_motor * u_theta,  # omega_theta_dot
                -self.d_omega * phi + self.k_motor * u_phi       # omega_phi_dot
            )

        # RK4 dynamics for simulation
        k1 = f(x, u)
        k2 = f(x + (h/2) * k1, u)
        k3 = f(x + (h/2) * k2, u)
        k4 = f(x + h * k3, u)
        x_next_RK4 = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        dynamics_RK4 = cas.Function('dynamics_RK4', [x, u], [x_next_RK4], ['x', 'u'], ['x+'])

        # Select discretization method
        if method == 'RK4':
            x_next = x_next_RK4
        elif method == 'RK2':
            k1 = f(x, u)
            k2 = f(x + (h/2) * k1, u)
            x_next = x + h * k2
        elif method == 'Euler':
            x_next = x + h * f(x, u)
        else:
            raise ValueError("Unknown method. Choose 'RK4', 'RK2', or 'Euler'.")

        dynamics = cas.Function('dynamics', [x, u], [x_next], ['x', 'u'], ['x+'])

        # Define output map (position)
        output_map = cas.Function('output', [x, u], [x[0:3]], ['x', 'u'], ['y'])

        super().__init__(dynamics=dynamics, output_map=output_map)
        self.dynamics_RK4 = dynamics_RK4

    def compute_jacobians(self, xval=None, uval=None) -> tuple[np.ndarray, np.ndarray]:
        """Compute Jacobians of the dynamics."""
        x = cas.MX.sym('x', 10)
        u = cas.MX.sym('u', 3)
        dfdx = cas.Function('dfdx', [x, u], [cas.jacobian(self.dynamics(x, u), x)], ['x', 'u'], ['dfdx'])
        dfdu = cas.Function('dfdu', [x, u], [cas.jacobian(self.dynamics(x, u), u)], ['x', 'u'], ['dfdu'])

        if xval is not None and uval is not None:
            return dfdx(xval, uval), dfdu(xval, uval)
        return dfdx, dfdu


class Satellite(Agent):
    """
    A satellite agent with a 4-dimensional or 3-dimensional state and 2-dimensional input.

    State:
    - r: Orbital radius (m)
    - theta: Angular position (rad) (optional)
    - v_r: Radial velocity (m/s)
    - omega: Angular velocity (rad/s)

    Input:
    - F_r: Radial thrust (N)
    - F_theta: Tangential thrust (N)
    
    Output:
    - r: Orbital radius (m)
    - theta: Angular position (rad) (optional)
    
    Attributes:
    - h (float): Sampling time.
    - mu (float): Gravitational parameter of the central body.
    - m (float): Mass of the satellite in kg.
    - method (str): Discretization method ('RK4', 'RK2' or 'Euler').
    - dynamics_RK4 (casadi.Function): Discrete-time dynamics using RK4.
    - scaling (str): Whether to use SI units ('m') or scale the dynamics to kilometre ('km') or megametre ('Mm').
    - r0 (float): Offset for the radius in m used for scaling.
    - state_dim (int): Dimension of the state.
    - input_dim (int): Dimension of the input.
    - dynamics (casadi.Function): Discrete-time dynamics of the agent, i.e. x+ = f(x,u), which is called in that order.
    - output_map (casadi.Function): Output map of the agents, i.e. y = h(x,u), where the output is the position of the satellite.
    - sf (float): Scaling factor for scaling metres.
    
    Functions:
    - compute_jacobians: Compute the Jacobians either evaluated at a given state and input or symbolically.
    """

    def __init__(self, h, mu=3.986e14, m=200, method='Euler', scaling='m', r0 = 0.0, no_theta=False):
        """
        Initialise a satellite agent.

        Arguments:
            - h (float): Sampling time.
            - mu (float): Gravitational parameter of the central body (default: Earth's mu = 3.986e14 m^3/s^2).
            - m (float): Mass of the satellite in kg (default: 200 kg).
            - method (str): Discretization method ('RK4', 'RK2' or 'Euler'). Default is 'Euler'.
            - scaling (str): Whether to use SI units ('m') or scale the dynamics to kilometre ('km') or megametre ('Mm'). Default is 'm'.
            - r0 (float): Offset for the radius in m used for scaling. Default is 0.0.
            - no_theta (bool): If True, the angular position is not included in the state and output. Default is False.
        """
        if no_theta:
            x = cas.MX.sym('x', 4)  # Define a symbolic state [r, v_r, omega]
        else:
            x = cas.MX.sym('x', 4)  # Define a symbolic state [r, theta, v_r, omega]
        u = cas.MX.sym('u', 2)  # Define a symbolic input [F_r, F_theta]
        
        if scaling == 'm':
            self.mu = mu
        elif scaling == 'km':
            self.mu = mu / 1e9
            r0 = r0*1e-3
        elif scaling == 'Mm':
            self.mu = mu / 1e18
            r0 = r0*1e-6
        else:
            raise ValueError("Unknown scaling. Choose 'm', 'km', or 'Mm'.")
        self.scaling = scaling
        self.r0 = r0
        self.m = m

        def f(x, u, mu=self.mu, r0=self.r0, m=self.m, no_theta=no_theta):
            """
            Defines the continuous-time satellite dynamics.

            State:
            - x[0] = r: Orbital radius (m)
            - x[1] = theta: Angular position (rad) (not included if no_theta is True)
            - x[2] = v_r: Radial velocity (m/s) (x[1] if no_theta is True)
            - x[3] = omega: Angular velocity (rad/s) (x[2] if no_theta is True)

            Input:
            - u[0] = F_r: Radial thrust (N)
            - u[1] = F_theta: Tangential thrust (N)

            Returns:
            - Time derivatives of the state variables.
            """
            if no_theta:
                r, v_r, omega = x[0], x[1], x[2]
                F_r, F_theta = u[0], u[1]

                return cas.vertcat(
                v_r,                                                        # r_dot
                    (r + r0) * omega**2 - mu / (r + r0)**2 + F_r / m,       # v_r_dot
                    -2 * v_r * omega / (r + r0) + F_theta / (m * (r + r0))  # omega_dot
                )
            else:
                r, theta, v_r, omega = x[0], x[1], x[2], x[3]
                F_r, F_theta = u[0], u[1]

                return cas.vertcat(
                    v_r,                                                    # r_dot
                    omega,                                                  # theta_dot
                    (r + r0) * omega**2 - mu / (r + r0)**2 + F_r / m,       # v_r_dot
                    -2 * v_r * omega / (r + r0) + F_theta / (m * (r + r0))  # omega_dot
                )

        # Always make the RK4 discretization available for simulation.
        k1 = f(x, u)
        k2 = f(x + (h/2) * k1, u)
        k3 = f(x + (h/2) * k2, u)
        k4 = f(x + h * k3, u)
        x_next_RK4 = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        dynamics_RK4 = cas.Function('dynamics_RK4', [x, u], [x_next_RK4], ['x', 'u'], ['x+'])  

        if method == 'RK4':
            x_next = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        elif method == 'RK2':
            k1 = f(x, u)
            k2 = f(x + (h/2) * k1, u)
            x_next = x + h * k2
        elif method == 'Euler':  # Euler method
            x_next = x + h * f(x, u)
        else:
            raise ValueError("Unknown method. Choose 'RK4', 'RK2', or 'Euler'.")

        dynamics = cas.Function('dynamics', [x, u], [x_next], ['x', 'u'], ['x+'])  

        # Define the position as the output.
        if no_theta:
            output_map = cas.Function('output', [x, u], [x[0]], ['x', 'u'], ['y'])
        else:
            output_map = cas.Function('output', [x, u], [cas.vertcat(x[0], x[1])], ['x', 'u'], ['y']) 

        # Call the parent class' constructor.
        super().__init__(dynamics=dynamics, output_map=output_map)
        self.h = h
        self.method = method
        self.dynamics_RK4 = dynamics_RK4
        self.sf = 1e-6 if scaling == 'Mm' else 1e-3 if scaling == 'km' else 1
        
    def compute_jacobians(self, xval=None, uval=None) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Jacobians either evaluated at a given state and input or symbolically.
        
        Returns:
            numpy.ndarray: Jacobian with respect to the state.
            numpy.ndarray: Jacobian with respect to the input.
        """
        x = cas.MX.sym('x', self.state_dim)  # Define a symbolic state.
        u = cas.MX.sym('u', self.input_dim)  # Define a symbolic input.
        
        dfdx = cas.Function('dfdx', [x, u], [cas.jacobian(self.dynamics(x,u), x)], ['x', 'u'], ['dfdx'])
        dfdu = cas.Function('dfdu', [x, u], [cas.jacobian(self.dynamics(x,u), u)], ['x', 'u'], ['dfdx'])
        
        if xval is not None and uval is not None:
            dfdx = dfdx(xval, uval)
            dfdu = dfdu(xval, uval)
            
        return dfdx, dfdu
    
    
class Vessel(Agent):
    def __init__():
        raise NotImplementedError("The Vessel class is not implemented yet.")
    
    
def compute_orbital_radius(agent, T, scaling_factor):
    """Try finding the initial orbital radius such that coasting leads to a periodic orbit.
    
    TODO: Should be moved to the Satellite class.
    """
    sf = scaling_factor
    
    r = cas.MX.sym('r', 1, 1)

    objective = r

    constraints = []
    constraints_lb = []
    constraints_ub = []

    r0 = agent.r0

    x = cas.vertcat(r, 0.0, 0.0, cas.sqrt(agent.mu/(r + r0)**3))
    for _ in range(T):
        x = agent.dynamics(x, np.zeros((2,1)))

    constraints.append(x[1] - 2*cas.pi)
    constraints_lb.append(np.zeros((1,1)))
    constraints_ub.append(np.zeros((1,1)))

    nlp = {'x': r, 'f': objective, 'g': cas.vertcat(*constraints)}
    S = cas.nlpsol('S', 'ipopt', nlp)
    Pr_sol = S(x0=(6.9e6)*sf - r0, lbg=np.concatenate(constraints_lb), ubg=np.concatenate(constraints_ub))
    
    return float(Pr_sol['x']/sf)


def save_data(data, agents, filestamp:str = None):
    # Get the current date and time
    now = datetime.now()
    # Format the date and time as a string
    timestamp = now.strftime("%Y%m%d_%H_%M_%S")
    data['time_stamp'] = f"{timestamp}"
    if filestamp is None:
        filestamp = timestamp
    filename = "{}_data.dill".format(filestamp)  # Create the filename with the filestamp.
    file_path = "./data/{}".format(filename)  # Concatenate with the path.
    
    if 'cooperative_task' in data:
        if 'kwargs' in data['cooperative_task'] and 'agents' in data['cooperative_task']['kwargs']:
            data['cooperative_task']['kwargs'].pop('agents')
    
    for agent in agents:
        data['agents'][f'A{agent.id}']['cl_x'] = agent.cl_x
        data['agents'][f'A{agent.id}']['cl_u'] = agent.cl_u
        if hasattr(agent, 'r0'):
            data['agents'][f'A{agent.id}']['r0'] = agent.r0
        if 'neighbours' not in data['agents'][f'A{agent.id}']:
            data['agents'][f'A{agent.id}']['neighbours'] = []
        for neighbour in agent.neighbours:            
            data['agents'][f'A{agent.id}']['neighbours'].append(neighbour.id)
        data['agents'][f'A{agent.id}']['MPC_sol'] = agent.MPC_sol

    with open(file_path, 'wb') as f:
        dill.dump(data, f)
    
    # Restore agents in kwargs.
    data['cooperative_task']['kwargs']['agents'] = agents
    print(f'Saved data to {file_path} at {timestamp}.')
        

def set_cooperative_task_to_narrow_position_exchange(t:int, agents:list, weight:float=1.0, N=None, T=1, data=None, distance=0.5) -> int:
    """Design a cooperative task, where agents exchange their positions. All necessary ingredients are saved to the agents as attributes.
    
    In the narrow_position_exchange example, the positions are pre-assigned since the focus lies on navigating the narrow path.
    The narrow path is described by ellipsoidal constraints.
    """
    horizontal_length = data['cooperative_task']['horizontal_length']
    vertical_length = data['cooperative_task']['vertical_length'] 
    vertical_length_tightened = data['cooperative_task']['vertical_length_tightened']
    exponent = data['cooperative_task']['exponent']
        
    # Initialize the variables that each agent uses for cooperation.
    for idx, agent in enumerate(agents):
        q = agent.input_dim
        n = agent.state_dim
        p = agent.output_dim
        # Define the decision variables.
        yT = cas.MX.sym(f'A{agent.id}_yT', p*T)  # Define the T steps of the trajectory as decision variables.
        agent.named_cooperation_dec_vars = {f'A{agent.id}_yT': yT}
        uT = cas.MX.sym(f'A{agent.id}_uT', q*T, 1)  # input sequence of cooperation reference
        xT = cas.MX.sym(f'A{agent.id}_xT', n*T, 1)  # state sequence of cooperation reference
        agent.named_cooperation_dec_vars[f'A{agent.id}_uT'] = uT
        agent.named_cooperation_dec_vars[f'A{agent.id}_xT'] = xT
    
    # Define collision avoidance constraints.
    for agent in agents:
        cstr = []
        x = cas.MX.sym(f'A{agent.id}_x', agent.state_dim, 1)
        n = agent.state_dim
        for nghbr in agent.neighbours:
            x_nghbr = cas.MX.sym(f'A{nghbr.id}_x', nghbr.state_dim, 1)
            n_nghbr = nghbr.state_dim
            if n != n_nghbr:
                raise ValueError(f'The state dimensions of agent {agent.id} and agent {nghbr.id} do not match!')
            # Coupling constraints should be defined pointwise-in-time.        
            func = cas.Function(f'collision_avoidance_{agent.id}_{nghbr.id}', 
                                [x, x_nghbr], 
                                [distance**2 - cas.dot(x[0:2] - x_nghbr[0:2], x[0:2] - x_nghbr[0:2])], 
                                [x.name(), x_nghbr.name()], 
                                ['g'])
            cstr.append(func)
        agent.coupling_constraints = cstr
        
    # Define the cooperation ingredients.
    for idx, agent in enumerate(agents):
        q = agent.input_dim
        n = agent.state_dim
        p = agent.output_dim
        # Initialize the objective function.
        cooperation_objective = cas.MX(0)
        
        yT = agent.named_cooperation_dec_vars[f'A{agent.id}_yT']
        xT = agent.named_cooperation_dec_vars[f'A{agent.id}_xT']
        uT = agent.named_cooperation_dec_vars[f'A{agent.id}_uT']
        
        # Define the cooperation output constraint.
        cooperation_constraint_map = []
        for tau in range(T):
            cooperation_constraint_map.append(agent.cooperation_constraints['Ay'] @ yT[tau*p : (tau+1)*p, 0] - agent.cooperation_constraints['by'])
            # Constraints for the cooperation state:
            cooperation_constraint_map.append(agent.cooperation_constraints['Ax'] @ xT[tau*n : (tau+1)*n, 0] - agent.cooperation_constraints['bx'])
            # Constraints for the cooperation input:
            cooperation_constraint_map.append(agent.cooperation_constraints['Au'] @ uT[tau*q : (tau+1)*q, 0] - agent.cooperation_constraints['bu'])
            
        # Build the cooperation cost.
        # Add the cost.
        diff = yT - agent.target_pos
        #-------------------------------
        # ^4 vs ^2:
        # if idx == 0:
        #     cooperation_objective += (1/100)*cas.dot(diff, diff)**2
        # else:
        #     #cooperation_objective += cas.dot(diff, diff)
        #     cooperation_objective += (1/2)*cas.dot(diff, diff)
        #-------------------------------
        # pure linear
        # if idx == 0:
        #     cooperation_objective += 2*10*cas.norm_2(diff)
        # else:
        #     cooperation_objective += 10*cas.norm_2(diff)
        #-------------------------------
        # linear vs quadratic
        # if idx == 0:
        #     #cooperation_objective += cas.norm_2(diff)
        #     cooperation_objective += (1/25)*cas.dot(diff, diff)
        # else:
        #     cooperation_objective += (1/25)*cas.dot(diff, diff)
        #-------------------------------
        # low fractional
        # if idx == 0:
        #     cooperation_objective += 2*10*cas.norm_2(diff)**1.01
        # else:
        #     cooperation_objective += 10*cas.norm_2(diff)**1.01
        #--------------------------------
        # Huber loss function
        # Example: a*d**2*(np.sqrt(1 + (x/d)**2) - 1), a = 100, d = 0.01
        a = 10*100
        d = 0.01
        if idx == 0:
            cooperation_objective += 2*a*d**2*(cas.sqrt(1 + cas.dot(diff, diff)/(d**2)) - 1)
        else:
            cooperation_objective += a*d**2*(cas.sqrt(1 + cas.dot(diff, diff)/(d**2)) - 1)
        
        # Define ellipsoids that are non-negative in the feasible set to describe the narrow path.
        v = cas.MX.sym('v', 2)
        lower_blockage = cas.Function('lower_blockage', [v], [(v[0] / horizontal_length)**exponent + ((v[1] + vertical_length_tightened) / 1)**exponent - 1], ['v'], ['value'])
        upper_blockage = cas.Function('upper_blockage', [v], [(v[0] / horizontal_length)**exponent + ((v[1] - vertical_length_tightened) / 1)**exponent - 1], ['v'], ['value'])
        cooperation_constraint_map.append(-lower_blockage(yT[0:2]))
        cooperation_constraint_map.append(-upper_blockage(yT[0:2]))
        
        # Add coupling constraints on the cooperation output, which is the position of the agent.
        enlarged_distance = distance + 0.01
        for nghbr in agent.neighbours:
            yT_nghbr = nghbr.named_cooperation_dec_vars[f'A{nghbr.id}_yT']
            for tau in range(T):   
                cooperation_constraint_map.append(enlarged_distance**2 - cas.dot(yT[tau*p : (tau+1)*p, 0] - yT_nghbr[tau*p : (tau+1)*p, 0], yT[tau*p : (tau+1)*p, 0] - yT_nghbr[tau*p : (tau+1)*p, 0]))  
            agent.named_cooperation_dec_vars[yT_nghbr.name()] = yT_nghbr
        
        # Add the constraint to the agent.
        # Create the function, but name inputs and outputs uniquely based on the agent.
        agent.cooperation_constraints['function'] = cas.Function(f'A{agent.id}_cooperation_constraint', agent.named_cooperation_dec_vars.values(), [cas.vertcat(*cooperation_constraint_map)], agent.named_cooperation_dec_vars.keys(), [f'A{agent.id}_value'])
        
        # Create the objective function and assign it to the agent.
        agent.cooperation_objective_function = cas.Function(f'A{agent.id}_cooperation_objective_function', agent.named_cooperation_dec_vars.values(), [weight*cooperation_objective], agent.named_cooperation_dec_vars.keys(), [f'A{agent.id}_value'])
        
        
def set_cooperative_task_to_circle_double_integrator(t, agents, T, phase_shift, radius_lb, radius_ub, radius_softplus_weight, radius_softplus_decay, constrain_radius):
    """Define ingredients for the cooperative task, which are saved to the agents as attributes.
    We need a cooperation objective function for each agent as well as the set of admissible cooperation outputs. 
    
    Deprecated version used for the double integrator example.

    We want to provide a custom implementation of the constraints and not restrict ourselves to box constraints.
    Hence, we need to define a function that takes the cooperation output and returns the constraint.

    Periodicity of the cooperation output is handled in the MPC method.
    """
    A = np.array(
        [[1., 0.],
         [-1., 0.],
         [0., 1.],
         [0., -1.]])
    b = np.array([[2.], [2.], [2.], [2.]])
    Ax = np.array(
        [[1., 0., 0., 0.],
         [-1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., -1., 0., 0.],
         [0., 0., 1., 0.],  # Constraints for 3rd and 4th state from here.
         [0., 0., -1., 0.],
         [0., 0., 0., 1.],
         [0., 0., 0., -1.]
        ]) 
    bx = np.array([[2.], [2.], [2.], [2.], [2.], [2.], [2.], [2.]])
    Au = np.array(
        [[1., 0.],
         [-1., 0.],
         [0., 1.],
         [0., -1.]])
    bu = np.array([[2.], [2.], [2.], [2.]])

    # Initialize the variables that each agent uses for cooperation.
    for i, agent in enumerate(agents):
        p = agent.output_dim
        # Define the decision variables.
        yT = cas.MX.sym(f'A{agent.id}_yT', p*T)  # Define the T steps of the trajectory as decision variables.
        radius = cas.MX.sym(f'A{agent.id}_radius', 1)  # Define the radius as a decision variable.
        yT_centre = cas.MX.sym(f'A{agent.id}_yT_centre', p)  # Define the translation of the circle's centre as a decision 
                    
        # Assign the information about the decision variables to the agent.
        agent.named_cooperation_dec_vars = {yT.name(): yT, radius.name(): radius, yT_centre.name(): yT_centre}
        
    # Define the cooperation ingredients.
    for i, agent in enumerate(agents):
        # Initialize the objective function.
        cooperation_objective = cas.MX(0)
        # Initialize a list of decision variables.
        
        yT = agent.named_cooperation_dec_vars[f'A{agent.id}_yT']
        radius = agent.named_cooperation_dec_vars[f'A{agent.id}_radius']
        yT_centre = agent.named_cooperation_dec_vars[f'A{agent.id}_yT_centre']
        
        # Define the cooperation output constraint.
        cooperation_constraint_map = []
        cooperation_constraint_lb = []
        cooperation_constraint_ub = []
        for k in range(T):
            cooperation_constraint_map.append(A@yT[k*p : (k+1)*p, 0])
            cooperation_constraint_ub.append(b)
            cooperation_constraint_lb.append(-np.inf*np.ones((b.shape[0],1)))
            
        # Either add a constraint on the radius or implement a soft constraint using the softplus function.
        # In both cases, the radius should be non-negative and bounded from above by a reasonable value.
        if constrain_radius:
            cooperation_constraint_map.append(radius)
            cooperation_constraint_lb.append(np.array([[radius_lb]]))
            cooperation_constraint_ub.append(np.array([[radius_ub]]))
        else:
            cooperation_objective += radius_softplus_weight*cas.log(1 + cas.exp(radius_softplus_decay*(radius_lb - radius)))
            cooperation_constraint_map.append(radius)
            cooperation_constraint_lb.append(np.array([[0.0]]))
            cooperation_constraint_ub.append(np.array([[radius_ub]]))
            
        # Add constraints on the centre of the circle.
        cooperation_constraint_map.append(A@yT_centre)
        cooperation_constraint_ub.append(b)
        cooperation_constraint_lb.append(-np.inf*np.ones((b.shape[0],1)))
            
        # Add a cost on the difference between the steps of the trajectory and what they should be.
        for k in range(T):
            tau = (t+k)%T
            cooperation_objective += (yT[p*k] - radius*np.cos(2*np.pi*tau/T + (i-1)*phase_shift) - yT_centre[0])**2
            cooperation_objective += (yT[p*k + 1] - radius*np.sin(2*np.pi*tau/T + (i-1)*phase_shift) - yT_centre[1])**2

        # Add a consensus cost on the centre and the radius.
        for neighbour in agent.neighbours:
            radius_nghb = neighbour.named_cooperation_dec_vars[f'A{neighbour.id}_radius']
            cooperation_objective += (radius - radius_nghb)**2
            
            yT_centre_nghb = neighbour.named_cooperation_dec_vars[f'A{neighbour.id}_yT_centre']
            cooperation_objective += cas.dot(yT_centre - yT_centre_nghb, yT_centre - yT_centre_nghb)
            
            # Add the decision variable of the neighbour to the agent's dictionary.
            agent.named_cooperation_dec_vars[radius_nghb.name()] = radius_nghb
            agent.named_cooperation_dec_vars[yT_centre_nghb.name()] = yT_centre_nghb
            
        # Add the constraint to the agent.
        agent.cooperation_constraints = {}  # Create the attribute for the agent.
        # Create the function, but name inputs and outputs uniquely based on the agent.
        agent.cooperation_constraints['function'] = cas.Function(f'A{agent.id}_cooperation_constraint', [yT, yT_centre, radius], [cas.vertcat(*cooperation_constraint_map)], [yT.name(), yT_centre.name(), radius.name()], [f'A{agent.id}_value'])
        agent.cooperation_constraints['upper_bound'] = cooperation_constraint_ub
        agent.cooperation_constraints['lower_bound'] = cooperation_constraint_lb
        agent.cooperation_constraints['Ax'] = Ax
        agent.cooperation_constraints['bx'] = bx
        agent.cooperation_constraints['Au'] = Au
        agent.cooperation_constraints['bu'] = bu
        
        # Create the objective function and assign it to the agent.
        agent.cooperation_objective_function = cas.Function(f'A{agent.id}_cooperation_objective_function', agent.named_cooperation_dec_vars.values(), [cooperation_objective], agent.named_cooperation_dec_vars.keys(), [f'A{agent.id}_value'])
    
    return T


def set_cooperative_task_to_constellation(t:int, agents:list, weight:float, N:int, T:int, theta_des:float, sf=1.0) -> int:
    """Design a cooperative task for satellites flying in a constellation.
                            
    Arguments:
        - t (int): Current time step.
        - agents (list): List of active agents (.Agent).
        - weight (float): Multiplicative weight of the cooperation objective function.
        - N (int): Prediction horizon.
        - T (int): Periodicity of the task.
        - theta_des (float): Desired angular position of the constellation in degrees.
        - sf (float): Scaling factor used in the dynamics. (default is 1.0)
    """
        
    # If called at the initial time step, initialize variables and constraints. These are time-invariant.
    for i, agent in enumerate(agents):
        agent.named_cooperation_dec_vars = {}
        
        q = agent.input_dim
        n = agent.state_dim
        p = agent.output_dim
        # Define the decision variables.
        yT = cas.MX.sym(f'A{agent.id}_yT', p*T)  # Define the T steps of the trajectory as decision variables.
        uT = cas.MX.sym(f'A{agent.id}_uT', q*T, 1)  # input sequence of cooperation reference
        xT = cas.MX.sym(f'A{agent.id}_xT', n*T, 1)  # state sequence of cooperation reference
        # Add variabels to a dictionary.
        agent.named_cooperation_dec_vars = {f'A{agent.id}_yT': yT}
        agent.named_cooperation_dec_vars[f'A{agent.id}_uT'] = uT
        agent.named_cooperation_dec_vars[f'A{agent.id}_xT'] = xT
        
        # Define the cooperation output constraint.
        cooperation_constraint_map = []
        for tau in range(T):
            if 'Ay' in agent.cooperation_constraints and agent.cooperation_constraints['Ay'] is not None:
                cooperation_constraint_map.append(agent.cooperation_constraints['Ay']@yT[tau*p : (tau+1)*p, 0] - agent.cooperation_constraints['by'])
            # Constraints for the cooperation state:
            if 'Ax' in agent.cooperation_constraints and agent.cooperation_constraints['Ax'] is not None:
                cooperation_constraint_map.append(agent.cooperation_constraints['Ax'] @ xT[tau*n : (tau+1)*n, 0] - agent.cooperation_constraints['bx'])
            # Constraints for the cooperation input:
            if 'Au' in agent.cooperation_constraints and agent.cooperation_constraints['Au'] is not None:
                cooperation_constraint_map.append(agent.cooperation_constraints['Au'] @ uT[tau*q : (tau+1)*q, 0] - agent.cooperation_constraints['bu'])
            
        # Add the constraint to the agent.        
        vars = [yT, xT, uT]
        var_names = [var.name() for var in vars]
        # Create the function, but name inputs and outputs uniquely based on the agent.
        agent.cooperation_constraints['function'] = cas.Function(f'A{agent.id}_cooperation_constraint', vars, [cas.vertcat(*cooperation_constraint_map)], var_names, [f'A{agent.id}_value'])
    
    # Define the cooperation objective function.
    # This cost is evaluated at once using the variables contained in named_cooperation_dec_vars of each agent. These should have shape (T, 1).
    for agent in agents:
        cooperation_objective = cas.MX(0)
        yT = agent.named_cooperation_dec_vars[f'A{agent.id}_yT']
        
        if yT.shape[0] != T:
            raise ValueError(f'The decision variable yT of agent {agent.id} has the wrong shape!')
        
        for neighbour in agent.neighbours:
            yT_nghb = neighbour.named_cooperation_dec_vars[f'A{neighbour.id}_yT']
            if yT_nghb.shape[0] != T:
                raise ValueError(f'The decision variable yT of agent {neighbour.id} has the wrong shape!')
            
            a = 100.*len(agent.neighbours)
            d = 0.01
            for k in range(T):
                # Add cost for a desired angular distance between agents.
                # cooperation_objective += ((yT[1] - yT_nghb[1])**2 - np.radians(theta_des)**2)**2
                if agent.id < neighbour.id:
                    # Agent is behind neighbour.
                    target = yT_nghb[k : k+1] - yT[k : k+1] - np.radians(theta_des)
                else:
                    # Agent is in front of neighbour.
                    target = yT[k : k+1] - yT_nghb[k : k+1] - np.radians(theta_des)

                cooperation_objective += len(agent.neighbours)*target**2
                #cooperation_objective += a*(d**2)*(cas.sqrt(1 + target**2/(d**2)) - 1)
            
            # Add the decision variable of the neighbour to the agent's dictionary.
            agent.named_cooperation_dec_vars[yT_nghb.name()] = yT_nghb
    
        # Create the objective function and assign it to the agent.
        agent.cooperation_objective_function = cas.Function(f'A{agent.id}_cooperation_objective_function', agent.named_cooperation_dec_vars.values(), [weight*cooperation_objective], agent.named_cooperation_dec_vars.keys(), [f'A{agent.id}_value'])

    return None


def set_cooperative_task_to_circle(t:int, agents:list[Agent], T:int, switching_time:int, phase_shift:float, radius_lb:float, radius_ub:float, radius_softplus_weight:float, radius_softplus_decay:float, constrain_radius:bool, get_leader_reference, distance:float) -> int:
    
    # Define collision avoidance constraints.
    for agent in agents:
        cstr = []
        x = cas.MX.sym(f'A{agent.id}_x', agent.state_dim, 1)
        n = agent.state_dim
        for nghbr in agent.neighbours:
            x_nghbr = cas.MX.sym(f'A{nghbr.id}_x', nghbr.state_dim, 1)
            n_nghbr = nghbr.state_dim
            if n != n_nghbr:
                raise ValueError(f'The state dimensions of agent {agent.id} and agent {nghbr.id} do not match!')
            # Coupling constraints should be defined pointwise-in-time.        
            func = cas.Function(f'collision_avoidance_{agent.id}_{nghbr.id}', 
                                [x, x_nghbr], 
                                [distance**2 - cas.dot(x[0:3] - x_nghbr[0:3], x[0:3] - x_nghbr[0:3])], 
                                [x.name(), x_nghbr.name()], 
                                ['g'])
            cstr.append(func)
        agent.coupling_constraints = cstr
    
    if t < switching_time:
        # Initialize the variables that each agent uses for cooperation.
        for i, agent in enumerate(agents):
            q = agent.input_dim
            n = agent.state_dim
            p = agent.output_dim
            # Define the decision variables.
            yT = cas.MX.sym(f'A{agent.id}_yT', p*T)  # Define the T steps of the trajectory as decision variables.
            radius = cas.MX.sym(f'A{agent.id}_radius', 1)  # Define the radius as a decision variable.
            yT_centre = cas.MX.sym(f'A{agent.id}_yT_centre', p)  # Define the translation of the circle's centre as a decision 
            
            uT = cas.MX.sym(f'A{agent.id}_uT', q*T, 1)  # input sequence of cooperation reference
            xT = cas.MX.sym(f'A{agent.id}_xT', n*T, 1)  # state sequence of cooperation reference
                        
            # Assign the information about the decision variables to the agent.
            agent.named_cooperation_dec_vars = {yT.name(): yT, radius.name(): radius, yT_centre.name(): yT_centre, xT.name(): xT, uT.name(): uT}
            
        # Define the cooperation ingredients.
        for i, agent in enumerate(agents):
            q = agent.input_dim
            n = agent.state_dim
            p = agent.output_dim
            # Initialize the objective function.
            cooperation_objective = cas.MX(0)
            # Initialize a list of decision variables.
            
            yT = agent.named_cooperation_dec_vars[f'A{agent.id}_yT']
            radius = agent.named_cooperation_dec_vars[f'A{agent.id}_radius']
            yT_centre = agent.named_cooperation_dec_vars[f'A{agent.id}_yT_centre']
            xT = agent.named_cooperation_dec_vars[f'A{agent.id}_xT']
            uT = agent.named_cooperation_dec_vars[f'A{agent.id}_uT']
            
            # Define the cooperation output constraint.
            cooperation_constraint_map = []
            Ay = agent.cooperation_constraints['Ay']
            by = agent.cooperation_constraints['by']
            for tau in range(T):
                cooperation_constraint_map.append(Ay@yT[tau*p : (tau+1)*p, 0] - by)
                cooperation_constraint_map.append(agent.cooperation_constraints['Ax'] @ xT[tau*n : (tau+1)*n, 0] - agent.cooperation_constraints['bx'])
                cooperation_constraint_map.append(agent.cooperation_constraints['Au'] @ uT[tau*q : (tau+1)*q, 0] - agent.cooperation_constraints['bu'])
                
            # Either add a constraint on the radius or implement a soft constraint using the softplus function.
            # In both cases, the radius should be non-negative and bounded from above by a reasonable value.
            if constrain_radius:
                cooperation_constraint_map.append(radius - np.array([[radius_ub]]))
                cooperation_constraint_map.append(np.array([[radius_lb]]) - radius)
            else:
                cooperation_objective += radius_softplus_weight*cas.log(1 + cas.exp(radius_softplus_decay*(radius_lb - radius)))
                cooperation_constraint_map.append(radius - np.array([[radius_ub]]))
                
            # Add constraints on the centre of the circle.
            cooperation_constraint_map.append(Ay@yT_centre - by)
            
            # Add a cost pushing for large radii.
            cooperation_objective += 1.2 * ( radius_ub - radius )
                
            # Add a cost on the difference between the steps of the trajectory and what they should be.
            for k in range(T):
                tau = (t+k)%T
                cooperation_objective += (1/T)*(yT[p*k] - radius*np.cos(2*np.pi*tau/T + (i-1)*phase_shift) - yT_centre[0])**2
                cooperation_objective += (1/T)*(yT[p*k + 1] - radius*np.sin(2*np.pi*tau/T + (i-1)*phase_shift) - yT_centre[1])**2

            # Add a consensus cost on the centre, the radius, and the altitude.
            for neighbour in agent.cooperation_neighbours:
                radius_nghb = neighbour.named_cooperation_dec_vars[f'A{neighbour.id}_radius']
                cooperation_objective += (radius - radius_nghb)**2
                
                yT_centre_nghb = neighbour.named_cooperation_dec_vars[f'A{neighbour.id}_yT_centre']
                cooperation_objective += cas.dot(yT_centre - yT_centre_nghb, yT_centre - yT_centre_nghb)
                
                yT_nghb = neighbour.named_cooperation_dec_vars[f'A{neighbour.id}_yT']
                for k in range(T):
                    cooperation_objective += (1/T)*(1/10)*(yT[p*k + 2] - yT_nghb[p*k + 2])**2
                    
                # Add the decision variable of the neighbour to the agent's dictionary.
                agent.named_cooperation_dec_vars[radius_nghb.name()] = radius_nghb
                agent.named_cooperation_dec_vars[yT_centre_nghb.name()] = yT_centre_nghb
                agent.named_cooperation_dec_vars[yT_nghb.name()] = yT_nghb
                
            # Add coupling constraints on the cooperation output, which is the position of the agent.
            enlarged_distance = distance + 0.01
            for nghbr in agent.neighbours:
                yT_nghbr = nghbr.named_cooperation_dec_vars[f'A{nghbr.id}_yT']
                for tau in range(T):   
                    cooperation_constraint_map.append(enlarged_distance**2 - cas.dot(yT[tau*p : (tau+1)*p, 0] - yT_nghbr[tau*p : (tau+1)*p, 0], yT[tau*p : (tau+1)*p, 0] - yT_nghbr[tau*p : (tau+1)*p, 0]))  
                agent.named_cooperation_dec_vars[yT_nghbr.name()] = yT_nghbr
                
            # Add the constraint to the agent.
            # Create the function, but name inputs and outputs uniquely based on the agent.
            agent.cooperation_constraints['function'] = cas.Function(f'A{agent.id}_cooperation_constraint',
                                                                        agent.named_cooperation_dec_vars.values(),
                                                                        [cas.vertcat(*cooperation_constraint_map)],
                                                                        agent.named_cooperation_dec_vars.keys(),
                                                                        [f'A{agent.id}_value'])
            
            # Create the objective function and assign it to the agent.
            agent.cooperation_objective_function = cas.Function(f'A{agent.id}_cooperation_objective_function', agent.named_cooperation_dec_vars.values(), [cooperation_objective], agent.named_cooperation_dec_vars.keys(), [f'A{agent.id}_value'])
    
    else:  ## Switch to the flocking task.
        
        if t == switching_time:
            # The first agent should only follow an external reference signal.
            agents[0].cooperation_neighbours = []
            # Delete the gurobi model to reinitialise it at the switching time.
            for agent in agents:
                if hasattr(agent, 'gurobi_model'):
                    delattr(agent, 'gurobi_model')
        
        # Define decision variables.
        for i, agent in enumerate(agents):
            q = agent.input_dim
            n = agent.state_dim
            p = agent.output_dim
            # Define the decision variables.
            yT = cas.MX.sym(f'A{agent.id}_yT', p*T)
            uT = cas.MX.sym(f'A{agent.id}_uT', q*T, 1)  
            xT = cas.MX.sym(f'A{agent.id}_xT', n*T, 1) 
                        
            # Assign the information about the decision variables to the agent.
            agent.named_cooperation_dec_vars = {yT.name(): yT, xT.name(): xT, uT.name(): uT}
            
        for i, agent in enumerate(agents):
            q = agent.input_dim
            n = agent.state_dim
            p = agent.output_dim
            # Initialize the objective function.
            cooperation_objective = cas.MX(0)
            
            yT = agent.named_cooperation_dec_vars[f'A{agent.id}_yT']
            xT = agent.named_cooperation_dec_vars[f'A{agent.id}_xT']
            uT = agent.named_cooperation_dec_vars[f'A{agent.id}_uT']
            
            # Define the cooperation output constraint.
            cooperation_constraint_map = []
            cooperation_constraint_lb = []
            cooperation_constraint_ub = []
            Ay = agent.cooperation_constraints['Ay']
            by = agent.cooperation_constraints['by']
            for tau in range(T):
                cooperation_constraint_map.append(Ay@yT[tau*p : (tau+1)*p, 0] - by)
                cooperation_constraint_map.append(agent.cooperation_constraints['Ax'] @ xT[tau*n : (tau+1)*n, 0] - agent.cooperation_constraints['bx'])
                cooperation_constraint_map.append(agent.cooperation_constraints['Au'] @ uT[tau*q : (tau+1)*q, 0] - agent.cooperation_constraints['bu'])

            # Add a tracking cost for the leader.
            if i == 0:
                # Get the reference.
                ref = get_leader_reference(t, switching_time)
                for tau in range(T):
                    cooperation_objective = cooperation_objective + (1/T)*cas.dot(yT[tau*p : (tau+1)*p, 0] - ref, yT[tau*p : (tau+1)*p, 0] - ref)
            else:
                # Add a consensus cost. Choose it to be quartic such that the tracking cost of the leader beats the consensus cost if all quadrotors are close.
                for neighbour in agent.cooperation_neighbours:
                    
                    yT_nghb = neighbour.named_cooperation_dec_vars[f'A{neighbour.id}_yT']
                    
                    for tau in range(T):
                        # Weight the altitude less than the horizontal position.
                        cooperation_objective = cooperation_objective + (1/T)*cas.dot(yT[tau*p : tau*p + 1, 0] - yT_nghb[tau*p : tau*p + 1, 0], yT[tau*p : tau*p + 1, 0] - yT_nghb[tau*p : tau*p + 1, 0])**2
                        cooperation_objective = cooperation_objective + (1/T)*cas.dot(yT[tau*p + 1 : tau*p + 2, 0] - yT_nghb[tau*p + 1 : tau*p + 2, 0], yT[tau*p + 1 : tau*p + 2, 0] - yT_nghb[tau*p + 1 : tau*p + 2, 0])**2
                        cooperation_objective = cooperation_objective + (1/T)*(1/10)*cas.dot(yT[tau*p + 2 : tau*p + 3, 0] - yT_nghb[tau*p + 2 : tau*p + 3, 0], yT[tau*p + 2 : tau*p + 3, 0] - yT_nghb[tau*p + 2 : tau*p + 3, 0])**2
                    
                    # Add the decision variable of the neighbour to the agent's dictionary.
                    agent.named_cooperation_dec_vars[yT_nghb.name()] = yT_nghb
                    
            # Add coupling constraints on the cooperation output, which is the position of the agent.
            enlarged_distance = distance + 0.01
            for nghbr in agent.neighbours:
                yT_nghbr = nghbr.named_cooperation_dec_vars[f'A{nghbr.id}_yT']
                for tau in range(T):   
                    cooperation_constraint_map.append(enlarged_distance**2 - cas.dot(yT[tau*p : (tau+1)*p, 0] - yT_nghbr[tau*p : (tau+1)*p, 0], yT[tau*p : (tau+1)*p, 0] - yT_nghbr[tau*p : (tau+1)*p, 0]))  
                agent.named_cooperation_dec_vars[yT_nghbr.name()] = yT_nghbr
                
            # Add the constraint to the agent.
            # Create the function, but name inputs and outputs uniquely based on the agent.
            agent.cooperation_constraints['function'] = cas.Function(f'A{agent.id}_cooperation_constraint',
                                                                        agent.named_cooperation_dec_vars.values(),
                                                                        [cas.vertcat(*cooperation_constraint_map)],
                                                                        agent.named_cooperation_dec_vars.keys(),
                                                                        [f'A{agent.id}_value'])
            agent.cooperation_constraints['upper_bound'] = cooperation_constraint_ub
            agent.cooperation_constraints['lower_bound'] = cooperation_constraint_lb
    
            # Create the objective function and assign it to the agent.
            agent.cooperation_objective_function = cas.Function(f'A{agent.id}_cooperation_objective_function', agent.named_cooperation_dec_vars.values(), [cooperation_objective], agent.named_cooperation_dec_vars.keys(), [f'A{agent.id}_value'])


def get_quadrotor10_MAS(data) -> list:
    """Generate the 10-state quadrotor multi-agent system. Returns the agents in a list. Updates the data dictionary.
    
    Arguments:
     - data (dict):
        - cooperative_task['type'] (str): The cooperative task that the agents should solve.
        - positions (list): List of initial and eventual positions of the agents if the cooperative task is 'position exchange'.
        - num_agents (int): Number of agents.
    """
    coop_task = data['cooperative_task']['type']
    num_agents = data['MAS_parameters']['num_agents']
    h = data['MAS_parameters']['h']
    
    # Save information about agents in a list.
    data['agents'] = {}

    agents = []  # Initialize a list to collect the agents in.
    for i in range(num_agents):
        # Initialize the quadrotor agent.
        agents.append(Quadrotor(h))
        
    for agent in agents:
        data['agents'][f'A{agent.id}'] = {}

    """Set constraints for each agent."""
    for i, agent in enumerate(agents):
        #--------------------------------------------------------
        # Set state constraints.
        #--------------------------------------------------------        
        Ax = np.array([
            [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0], # z1
            [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0], # z1
            [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0], # z2
            [ 0, -1,  0,  0,  0,  0,  0,  0,  0,  0], # z2
            [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0], # z3
            [ 0,  0, -1,  0,  0,  0,  0,  0,  0,  0], # z3
            [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0], # theta; pitch angle 
            [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0], # theta; pitch angle
            [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0], # phi; roll angle
            [ 0,  0,  0,  0, -1,  0,  0,  0,  0,  0], # phi; roll angle
            [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0], # v1
            [ 0,  0,  0,  0,  0, -1,  0,  0,  0,  0], # v1
            [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0], # v2
            [ 0,  0,  0,  0,  0,  0, -1,  0,  0,  0], # v2
            [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0], # v3
            [ 0,  0,  0,  0,  0,  0,  0, -1,  0,  0], # v3
            [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0], # omega_theta
            [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  0], # omega_theta
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1], # omega_phi
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -1]  # omega_phi
            ])
        # Set the upper and lower bounds for the state constraints.
        # Upper bound first, then negative lower bound.
        bx = np.array([
            [21.0], [21.0], # z1
            [21.0], [21.0], # z2
            [21.0], [21.0], # z3
            [np.pi/4], [np.pi/4], # theta; pitch angle
            [np.pi/4], [np.pi/4], # phi; roll angle
            [2.0], [2.0], # v1
            [2.0], [2.0], # v2
            [2.0], [2.0], # v3
            [3.0], [3.0], # omega_theta
            [3.0], [3.0]  # omega_phi
            ])
        agent.state_constraints['A'] = Ax
        agent.state_constraints['b'] = bx
        agent.cooperation_constraints['Ax'] = Ax 
        agent.cooperation_constraints['bx'] = np.array([
            [20.95], [20.95], # z1
            [20.95], [20.95], # z2
            [20.95], [20.95], # z3
            [0.75], [0.75], # theta; pitch angle
            [0.75], [0.75], # phi; roll angle
            [1.95], [1.95], # v1
            [1.95], [1.95], # v2
            [1.95], [1.95], # v3
            [2.9], [2.9], # omega_theta
            [2.9], [2.9]  # omega_phi
            ]) # Tighten the constraints on the cooperation state.

        #--------------------------------------------------------
        # Set input constraints.
        #--------------------------------------------------------
        agent.input_constraints['A'] = np.array([
            [ 1,  0,  0], # u1
            [-1,  0,  0], # u1
            [ 0,  1,  0], # u2
            [ 0, -1,  0], # u2
            [ 0,  0,  1], # u3
            [ 0,  0, -1]  # u3
            ])
        agent.input_constraints['b'] = np.array([
            [np.pi/9],  [np.pi/9],      # u1
            [np.pi/9],  [np.pi/9],      # u2
            [2*agent.g],   [0]])     # u3 (thrust)
        agent.cooperation_constraints['Au'] = agent.input_constraints['A']
        agent.cooperation_constraints['bu'] = np.array([
            [0.3], [0.3],  
            [0.3], [0.3],
            [19.5], [0.05]
        ])
        
        #--------------------------------------------------------
        # Set cooperation output constraints.
        #--------------------------------------------------------
        Ay = np.array([
            [ 1,  0,  0], # y1
            [-1,  0,  0], # y1
            [ 0,  1,  0], # y2
            [ 0, -1,  0], # y2
            [ 0,  0,  1], # y3
            [ 0,  0, -1], # y3
            ])
        by = np.array([
            [20.95], [20.95], # y1
            [20.95], [20.95], # y2
            [20.95], [20.95], # y3
            ])
        agent.cooperation_constraints['Ay'] = Ay
        agent.cooperation_constraints['by'] = by
        
    """Define the stage costs."""
    for agent in agents:
        # Define the artificial equilibrium. 
        x = cas.MX.sym('x', agent.state_dim)
        u = cas.MX.sym('u', agent.input_dim)
        xT = cas.MX.sym('x_c', agent.state_dim)
        uT = cas.MX.sym('u_c', agent.input_dim)
        
        # Set the weight for the distance of the state to the equilibrium.
        Q = (1/4)*np.diag([
            2., 2., 2.,  # z1, z2, z3
            1.5, 1.5,          # theta, phi
            1., 1., 1.,      # v1, v2, v3
            0.5, 0.5         # omega_theta, omega_phi
        ])       
        # Set the weight for the distance of the input to the equilibrium.
        R = (1/4)*np.diag([1, 1, 0.02])
            
        # Add stage cost to agents.
        agent.stage_cost = cas.Function(
            'stage_cost', 
            [x, u, xT, uT], 
            [ (x - xT).T@Q@(x - xT) + (u - uT).T@R@(u - uT) ],
            ['x', 'u', 'xT', 'uT'], 
            ['l'])
        agent.stage_cost_weights = {'Q': Q, 'R': R}
        data['agents'][f'A{agent.id}']['stage_cost'] = {'Q': Q, 'R': R}
        
    #----------------------------------------
    # Define the communication topology.
    #----------------------------------------
    if coop_task in ['position exchange', 'narrow_position_exchange']:
        # Define an all-to-all topology.
        agents[0].neighbours = [agents[1]]
        for i, agent in enumerate(agents):
            agent.neighbours = []
            for j in range(len(agents)):
                if j != i:
                    agent.neighbours.append(agents[j])
    elif coop_task == 'circle' or coop_task == 'square':
        # Define a one-after-the-other communication structure.
        agents[0].neighbours = [agents[1]]
        for i, agent in enumerate(agents[1:-1]):
            i += 1
            agent.neighbours = [agents[i-1], agents[i+1]]
        agents[-1].neighbours = [agents[-2]]
    else:
        raise ValueError('The cooperative task is not recognized.')

    return agents


def get_satellites_MAS(data, scaling, method):
    """Generate the satellite multi-agent system. Returns the agents in a list. Updates the data dictionary.
    
    Arguments:
     - data (dict):
        - cooperative_task['type'] (str): The cooperative task that the agents should solve.
        - positions (list): List of initial and eventual positions of the agents if the cooperative task is 'position exchange'.
        - num_agents (int): Number of agents.
        - T (int): Period length of the cooperative task.
    - scaling (str): Whether to use SI units ('m') or scale the dynamics to kilometre ('km') or megametre ('Mm'). Default is 'm'.
    - method (str): Discretisation method.
    """
    # Extract the parameters from the data dictionary.
    T = data['sim_pars']['T']
    terminal_ingredients_type = data['sim_pars']['terminal_ingredients_type']
    # Compute the scaling factor.
    sf = 1e-6 if scaling == 'Mm' else 1e-3 if scaling == 'km' else 1
    #--------------------------------------------------
    # Set parameters.
    sat = Satellite(data['MAS_parameters']['h'], scaling=scaling, r0=0.0, method=method)
    r0 = compute_orbital_radius(sat, T, sf)
    del sat
    Agent._counter = 0  # Reset the counter for the agents.
    # All parameters should be set in metre.
    r_max =  0.1e6  # Maximum orbital radius deviation from r0; 7.4e6 m total is a reasonable upper bound.
    r_min = -0.1e6  # Minimum orbital radius deviation from r0; 6.5e6 m total is a reasonable upper bound.
    r_tight = abs(r_max - r_min)/8  # Tightening of orbital radius for constraints on cooperation reference.
    vr_max =  50   # Maximum radial velocity.
    vr_min = -50   # Minimum radial velocity.
    vr_tight = abs(vr_max - vr_min)*1e-5  # Tightening of radial velocity for constraints on cooperation reference.
    omega_max = 0.02  # Maximum angular velocity.
    omega_min = 0.0 #-0.01#-0.01  # Minimum angular velocity.
    omega_tight = abs(omega_max - omega_min)*1e-5  # Tightening of angular velocity for constraints on cooperation reference.
    
    # Thrust is limited to 237 mN; based on the NASA Evolutionary Xenon Thruster (NEXT).
    F_r_max =  0.237  # Maximum radial thrust.
    F_r_min = -0.237  # Minimum radial thrust.
    F_r_tight = abs(F_r_max - F_r_min)*1e-6  # Tightening of radial thrust for constraints on cooperation reference.
    F_t_max =  0.237  # Maximum tangential thrust.
    F_t_min = -0.237  # Minimum tangential thrust.
    F_t_tight = abs(F_t_max - F_t_min)*1e-6  # Tightening of tangential thrust for constraints on cooperation reference.
    #--------------------------------------------------

    # Save some of the parameters in the data dictionary (mainly for plotting).
    if True:
        data['MAS_parameters']['r_max'] = r_max
        data['MAS_parameters']['r_min'] = r_min
        data['MAS_parameters']['r0'] = r0
        data['MAS_parameters']['r_tight'] = r_tight
        data['MAS_parameters']['vr_max'] = vr_max
        data['MAS_parameters']['vr_min'] = vr_min
        data['MAS_parameters']['vr_tight'] = vr_tight
        data['MAS_parameters']['omega_max'] = omega_max
        data['MAS_parameters']['omega_min'] = omega_min
        data['MAS_parameters']['omega_tight'] = omega_tight
        data['MAS_parameters']['F_r_max'] = F_r_max
        data['MAS_parameters']['F_r_min'] = F_r_min
        data['MAS_parameters']['F_r_tight'] = F_r_tight
        data['MAS_parameters']['F_t_max'] = F_t_max
        data['MAS_parameters']['F_t_min'] = F_t_min
        data['MAS_parameters']['F_t_tight'] = F_t_tight
        data['MAS_parameters']['scaling_factor'] = sf
        data['MAS_parameters']['scaling'] = scaling
    
    coop_task = data['cooperative_task']['type']
    num_agents = data['MAS_parameters']['num_agents']
    h = data['MAS_parameters']['h']
    
    # Save information about agents in a list.
    data['agents'] = {}

    agents = []  # Initialize a list to collect the agents in.
    for i in range(num_agents):
        # Initialize the satellite agent.
        agents.append(Satellite(h, method=method, scaling=scaling, r0=r0))
        
    for agent in agents:
        data['agents'][f'A{agent.id}'] = {}

    """Set constraints for each agent."""
    for i, agent in enumerate(agents):
        #--------------------------------------------------------
        # Set state constraints.
        #--------------------------------------------------------
        Ax = np.array([
            [ 1,  0,  0,  0], # r (orbital radius)
            [-1,  0,  0,  0], # r (orbital radius)
            [ 0,  0,  1,  0], # v_r (radial velocity)
            [ 0,  0, -1,  0], # v_r (radial velocity)
            [ 0,  0,  0,  1], # omega (angular velocity)
            [ 0,  0,  0, -1], # omega (angular velocity)
            ])
        bx = np.array([
            [r_max*sf], [-r_min*sf],    # r (orbital radius)
            [vr_max*sf], [-vr_min*sf],  # v_r (radial velocity)
            [omega_max], [-omega_min]       # omega (angular velocity)
            ])
        agent.state_constraints['A'] = Ax
        agent.state_constraints['b'] = bx
        agent.cooperation_constraints['Ax'] = Ax 
        # Tighten the constraints on the cooperation state.
        # Tighten the constraints on the angular velocity such that only non-negative velocities are allowed.
        agent.cooperation_constraints['bx'] = bx - np.vstack([r_tight*sf, r_tight*sf, 
                                                              vr_tight*sf, vr_tight*sf, 
                                                              omega_tight, omega_tight])  
        # Allow only cooperation trajectories with no radial velocity.
        agent.cooperation_constraints['bx'][2] = 0.
        agent.cooperation_constraints['bx'][3] = 0.
        
        # If terminal constraints are used, then the orbital radius is constrained to the reference orbital radius, i.e. x[0] = 0.
        # Moreover, the angular velocity is constrained to the coasting velocity.
        if terminal_ingredients_type == 'set' or terminal_ingredients_type == 'equality':
            agent.cooperation_constraints['bx'][0] = 0.
            agent.cooperation_constraints['bx'][1] = 0.
            agent.cooperation_constraints['bx'][4] = np.sqrt(agent.mu/(agent.r0)**3)
            agent.cooperation_constraints['bx'][5] = np.sqrt(agent.mu/(agent.r0)**3)

        #--------------------------------------------------------
        # Set input constraints.
        #--------------------------------------------------------
        agent.input_constraints['A'] = np.array([
            [ 1,  0],  # thrust in radial direction
            [-1,  0],  # "-"
            [ 0,  1],  # thrust in tangential direction
            [ 0, -1]   # "-"
            ])
        agent.input_constraints['b'] = np.array([
            [F_r_max*sf],  [-F_r_min*sf],  # thrust in radial direction
            [F_t_max*sf],  [-F_t_min*sf]   # thrust in tangential direction
            ])  
        agent.cooperation_constraints['Au'] = agent.input_constraints['A'] 
        agent.cooperation_constraints['bu'] = agent.input_constraints['b'] - np.vstack([F_r_tight*sf, F_r_tight*sf,
                                                                                        F_t_tight*sf, F_t_tight*sf])  # Tighten the constraints on the cooperation input.
        # Allow only cooperation trajectories with no tangential thrust.
        agent.cooperation_constraints['bu'][2] = 0.
        agent.cooperation_constraints['bu'][3] = 0.
        
        # If terminal constraints are used, then the thrust is constrained to zero.
        if terminal_ingredients_type == 'set' or terminal_ingredients_type == 'equality':
            agent.cooperation_constraints['bu'][0] = 0.
            agent.cooperation_constraints['bu'][1] = 0.

        #--------------------------------------------------------        
        # Redefine the output map to only use the angular position.
        #--------------------------------------------------------
        x = cas.MX.sym('x', 4)  # Define a symbolic state [r, theta, v_r, omega]
        u = cas.MX.sym('u', 2)  # Define a symbolic input [F_r, F_theta]
        agent.output_map = cas.Function('output', [x, u], [cas.vertcat(x[1])], ['x', 'u'], ['y'])
        
    #--------------------------------------------------------
    # Set the stage costs.
    #--------------------------------------------------------
    for agent in agents:
        # Define the artificial equilibrium. 
        x = cas.MX.sym('x', agent.state_dim)
        u = cas.MX.sym('u', agent.input_dim)
        xT = cas.MX.sym('x_c', agent.state_dim)
        uT = cas.MX.sym('u_c', agent.input_dim)
        
        # Set the weight for the distance of the state to the cooperation state
        Q = np.diag([1e-7/sf, 0.1, 0.001, 0.01])/data['MAS_parameters']['num_agents']
        # Set the weight for the distance of the input to the cooperation input.
        R = 1e-7/sf*np.eye(agent.input_dim)
            
        # Add stage cost to agents.
        agent.stage_cost = cas.Function(
            'stage_cost', 
            [x, u, xT, uT], 
            [ (x - xT).T@Q@(x - xT) + (u - uT).T@R@(u - uT) ],
            ['x', 'u', 'xT', 'uT'], 
            ['l'])
        agent.stage_cost_weights = {'Q': Q, 'R': R}
        data['agents'][f'A{agent.id}']['stage_cost'] = {'Q': Q, 'R': R}
        
    #----------------------------------------
    # Define the communication topology.
    #----------------------------------------
    if coop_task in ['constellation']:
        # Define a one-after-the-other communication structure.
        agents[0].neighbours = [agents[1]]
        for i, agent in enumerate(agents[1:-1]):
            i += 1
            agent.neighbours = [agents[i-1], agents[i+1]]
        agents[-1].neighbours = [agents[-2]]
    else:
        raise ValueError(f'The cooperative task {coop_task} is not recognized for the satellite multi-agent system.')

    return agents


def get_double_integrator_MAS(data) -> list:
    """Generate the double integrator multi-agent system. Returns the agents in a list. Updates the data dictionary."""
    coop_task = data['cooperative_task']['type']
    num_agents = data['MAS_parameters']['num_agents']

    # Save information about agents in a list.
    data['agents'] = {}

    dynamics_list = []  # Collect the functions defining the dynamics in a list.
    output_maps = []  # Collect the functions defining the output in a list.

    agents = []
    for i in range(num_agents):
        A = np.array([[1., 0., 1., 0.],
                      [0., 1., 0., 1.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])
        B = np.array([[0., 0.],
                      [0., 0.],
                      [1., 0.],
                      [0., 1.]])
        state_dim = A.shape[0]
        input_dim = B.shape[1]
        x = cas.MX.sym('x', state_dim)  # Create a symbolic for the state.
        u = cas.MX.sym('u', input_dim)  # Create a symbolic for the input.
        # Create the state dynamics as a casadi function.
        dynamics_list.append(cas.Function('f', [x,u], [A@x + B@u], ['x', 'u'], ['x+']))
        # Create the output map as a casadi function.
        C = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])
        output_maps.append(cas.Function('h', [x,u], [C@x], ['x', 'u'], ['y']))
        
        # Initialise the agent.
        agents.append(Agent(dynamics = dynamics_list[i], initial_time=0, output_map=output_maps[i]))

    # Save the data.
    for agent in agents:
        data['agents'][f'A{agent.id}'] = {}
        data['agents'][f'A{agent.id}']['state_dim'] = agent.state_dim
        data['agents'][f'A{agent.id}']['input_dim'] = agent.input_dim
        data['agents'][f'A{agent.id}']['output_dim'] = agent.output_dim
        
    # Set box constraints for each agent.
    # Use simple constraints.
    for agent in agents:
        agent.set_box_input_constraints(box_input_constraints=[-0.25, 0.25])
        agent.cooperation_constraints['Au'] = agent.input_constraints['A'] 
        agent.cooperation_constraints['bu'] = agent.input_constraints['b'] - 0.05  # Tighten the constraints on the cooperation input.
        agent.state_constraints['A'] = np.array([[1., 0., 0., 0.],
                                                 [-1., 0., 0., 0.],
                                                 [0., 1., 0., 0.],
                                                 [0., -1., 0., 0.],
                                                 [0., 0., 1., 0.],  # Constraints for 3rd and 4th state from here.
                                                 [0., 0., -1., 0.],
                                                 [0., 0., 0., 1.],
                                                 [0., 0., 0., -1.]
                                                ]) 
        agent.state_constraints['b'] = np.array([[25.1], [25.1], [25.1], [25.1], [1.0], [1.0], [1.0], [1.0]])
        agent.cooperation_constraints['Ax'] = agent.state_constraints["A"] 
        agent.cooperation_constraints['bx'] = agent.state_constraints["b"] - 0.5e-1  # Tighten the constraints on the cooperation state.
        agent.cooperation_constraints['Ay'] = np.array([[1., 0.],
                                                        [-1., 0.],
                                                        [0., 1.],
                                                        [0., -1.]])  
        agent.cooperation_constraints['by'] = np.array([[25.1], [25.1], [25.1], [25.1]]) - 0.1  # Tighten the constraints on the cooperation output.
        
    # Define the stage cost of each agent.
    for agent in agents:
        x = cas.MX.sym('x', agent.state_dim)
        u = cas.MX.sym('u', agent.input_dim)
        xT = cas.MX.sym('xT', agent.state_dim)
        uT = cas.MX.sym('uT', agent.input_dim)
        
        # Set the weight for the distance of the state to the equilibrium.
        Q = np.diag([1., 1., 0.1, 0.1])
        # Set the weight for the distance of the input to the equilibrium.
        R = 0.1*np.eye(agent.input_dim)
        data['agents'][f'A{agent.id}']['stage_cost'] = {'Q': Q, 'R': R}
        
        stage_cost = cas.Function('stage_cost', [x, u, xT, uT], [ (x - xT).T@Q@(x - xT) + (u - uT).T@R@(u - uT) ],
                                ['x', 'u', 'xT', 'uT'], ['l'])
        
        # Add stage cost to agents.
        agent.stage_cost = stage_cost
                
        #------------------------------------------------
        # Define the communication topology.
        #------------------------------------------------
        if coop_task in ['position exchange', 'narrow_position_exchange', 'harbour']:
            # Define an all-to-all topology.
            for i, agent in enumerate(agents):
                agent.neighbours = []
                for j in range(len(agents)):
                    if j != i:
                        agent.neighbours.append(agents[j])
        elif coop_task == 'circle' or coop_task == 'square':
            # Define a one-after-the-other communication structure.
            agents[0].neighbours = [agents[1]]
            for i, agent in enumerate(agents[1:-1]):
                i += 1
                agent.neighbours = [agents[i-1], agents[i+1]]
            agents[-1].neighbours = [agents[-2]]
        else:
            raise ValueError('The cooperative task is not recognized.')

    return agents


def compute_terminal_ingredients_for_quadrotor(agent:Agent, data:dict, grid_resolution:int, num_decrease_samples:int, alpha:float, alpha_tol:float = 1e-8, references_are_equilibria:bool=False, compute_size_for_decrease:bool=True, compute_size_for_constraints:bool=True, epsilon:float=1.0, verbose:int=1, solver:str='MOSEK'):
    """Design terminal ingredients for the 10-state quadrotor."""

    if not hasattr(agent, 'terminal_ingredients'):
        agent.terminal_ingredients = {}  # Initialize the dictionary.
        
    if references_are_equilibria:
        # Since the linearization is independent of the reference, we can compute the terminal matrices by standard design.
        # If terminal ingredients are designed for equilibria, then the LPV parametrisation consists only of a static matrix, respectively.
        A, B = agent.compute_jacobians()
        A_LPV = {'static': A}
        B_LPV = {'static': B}
        get_lpv_par = None  # No need to compute the LPV parameters.
        get_next_points = None  # No need to compute the next points.
        
        # Create a 'grid' with one equilibrium in order to check the decrease condition, if desired.
        if compute_size_for_decrease:
            grid = {'xT': [np.vstack([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])], 'uT': [np.vstack([0.0, 0.0, agent.g/agent.k_thrust])]}
        if compute_size_for_constraints:
            if grid_resolution < 2:
                grid_resolution = 2
            # Compute a grid at the boundary of the constraints to compute the size for constraint satisfaction.
            x_lbs, x_ubs = get_bounds_of_affine_constraint(agent.cooperation_constraints['Ax'], agent.cooperation_constraints['bx'])
            x_samples = [np.linspace(x_lbs[i], x_ubs[i], grid_resolution) for i in range(3)]  # Only the position can vary for an equilibrium.
            x_mesh = np.meshgrid(*x_samples, indexing='ij')  # 'ij' ensures Cartesian indexing.
            # Flatten each dimension.
            x_flat = [axis.ravel() for axis in x_mesh]
            # Combine the state dimensions into one list.
            xT_list = np.column_stack(x_flat).tolist()
            [xT.extend([0.0]*7) for xT in xT_list]  # Add zeros for the velocities and angles.
            if compute_size_for_decrease:
                grid['xT'] += [np.vstack(xT) for xT in xT_list] 
                grid['uT'] += [np.vstack([0.0, 0.0, agent.g/agent.k_thrust]) for xT in xT_list]
            else:
                grid = {'xT': [np.vstack(xT) for xT in xT_list] , 'uT': [np.vstack([0.0, 0.0, agent.g/agent.k_thrust]) for xT in xT_list]}
            agent.terminal_ingredients['grid_resolution'] = grid_resolution
            
    else:        
        if agent.cooperation_constraints['Au'] is None or agent.cooperation_constraints['bu'] is None:
            raise ValueError('Affine constraints for the cooperation input are not defined in agent.cooperation_constraints!')
        if agent.cooperation_constraints['Ax'] is None or agent.cooperation_constraints['bx'] is None:
            raise ValueError('Affine constraints for the cooperation input are not defined in agent.cooperation_constraints!')
                
        n = agent.state_dim
        q = agent.input_dim
        
        # The LPV parameterisation is explicitly designed for a discrete-time model achieved by Euler discretisation.
        # Raise an error if this is not the chosen discretisation method.
        if agent.method != 'Euler':
            raise NotImplementedError('Terminal ingredients for trajectories can only be computed for Euler discretization!')

        xTsym = cas.MX.sym('xT', n, 1)
        uTsym = cas.MX.sym('uT', q, 1)
        
        get_lpv_par = cas.Function(
            f'A{agent.id}_lpv_parameter',
            [xTsym, uTsym],
            [
                1 / ( cas.cos(xTsym[3])**2 ),
                1 / ( cas.cos(xTsym[4])**2 )
            ],
            [xTsym.name(), uTsym.name()],
            ['par1', 'par2']
        )
                
        # Compute the matrices that define the LPV system.
        h = agent.h
        d_phi = agent.d_phi
        d_omega = agent.d_omega
        k_motor = agent.k_motor
        k_thrust = agent.k_thrust
        g = agent.g
        A0 = np.array([
            [1., 0., 0., 0., 0.,  h, 0., 0., 0., 0.], # z1
            [0., 1., 0., 0., 0., 0.,  h, 0., 0., 0.], # z2
            [0., 0., 1., 0., 0., 0., 0.,  h, 0., 0.], # z3
            [0., 0., 0., 1 - h*d_phi, 0., 0., 0., 0.,  h, 0.], # phi1
            [0., 0., 0., 0., 1 - h*d_phi, 0., 0., 0., 0.,  h], # phi2
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], # v1
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], # v2
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], # v3
            [0., 0., 0., -h*d_omega, 0., 0., 0., 0., 1., 0.], # omega1
            [0., 0., 0., 0., -h*d_omega, 0., 0., 0., 0., 1.]  # omega2       
        ])
        B0 = np.array([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., h*k_thrust],
            [h*k_motor, 0., 0.],
            [0., h*k_motor, 0.]   
        ])
        A1 = np.zeros((10, 10))
        A1[5, 3] = h*g
        B1 = np.zeros((10, 3))
        A2 = np.zeros((10, 10))
        A2[6, 4] = h*g
        B2 = np.zeros((10, 3))	
        A_LPV = {'static': A0, 'par1': A1, 'par2': A2}
        B_LPV = {'static': B0, 'par1': B1, 'par2': B2} 

        # The LPV parameterisation depends only on the fourth and fifth states (pitch angle and roll angle)
        # and the constraints are assumed to be polytopic.
        # Hence, we can convexify the design. 
        
        # Compute the polytopic bounds on the constraints.
        x_lbs, x_ubs = get_bounds_of_affine_constraint(agent.cooperation_constraints['Ax'], agent.cooperation_constraints['bx'])
        
        # Compute the bounds on the LPV parameters.
        # The smallest absolute bound results in the largest bound on the LPV parameters.
        par1_max = np.max([1 / ( np.cos(x_lbs[3])**2 ), 1 / ( np.cos(x_ubs[3])**2 )])
        par1_min = 1.0  # The minimum of 1/cos^2(x) is 1.
        par2_max = np.max([1 / ( np.cos(x_lbs[4])**2 ), 1 / ( np.cos(x_ubs[4])**2 )])
        par2_min = 1.0  # The minimum of 1/cos^2(x) is 1.
        
        # Create the vertices.
        vertices = []
        vertices.append({'par1': par1_min, 'par2': par2_min})
        vertices.append({'par1': par1_min, 'par2': par2_max})
        vertices.append({'par1': par1_max, 'par2': par2_min})
        vertices.append({'par1': par1_max, 'par2': par2_max})
        grid = {'vertices': vertices}
        
        # Sample the reference space.
        # We use Latin Hypercube sampling since the ten dimensional state space is too large to sample uniformly.
        # The number of samples is given by the grid resolution in an abuse of naming.
        grid['reference_points'] = generate_lhs_reference_grid(agent, num_samples = int(grid_resolution), seed=42)

        agent.terminal_ingredients['grid_resolution'] = None
        
        def get_next_points(agent: Agent, grid_point: dict) -> list:
            """
            Compute the next reference point (xT+, uT+) from the current (xT, uT)
            for use in terminal cost decrease validation.
            
            Ensures that xT+ satisfies cooperation state constraints.
            Reuses uT as a valid future reference input.
            """
            xT = grid_point['xT']
            uT = grid_point['uT']

            # Propagate one step using the agent's dynamics.
            xT_next = agent.dynamics(x=xT, u=uT)[agent.dynamics.name_out()[0]]
            xT_next = np.array(xT_next)

            # Check cooperation state constraints Ax x <= b.
            Ax = agent.cooperation_constraints['Ax']
            bx = agent.cooperation_constraints['bx']
            if Ax is not None and bx is not None:
                if not np.all(Ax @ xT_next <= bx + 1e-8):
                    return []  # Skip infeasible transitions.

            return [{'xT': xT_next, 'uT': uT}]
       
    agent.terminal_ingredients.update({'get_lpv_par': get_lpv_par, 
                                       'A_LPV': A_LPV, 
                                       'B_LPV': B_LPV, 
                                       'Q':agent.stage_cost_weights['Q'], 
                                       'R':agent.stage_cost_weights['R'], 
                                       'get_next_points': get_next_points})

    prob = compute_generic_terminal_ingredients(
        agent, 
        grid, 
        num_decrease_samples = num_decrease_samples, 
        alpha = alpha, 
        compute_size_for_decrease = compute_size_for_decrease,
        alpha_tol = alpha_tol,
        compute_size_for_constraints = compute_size_for_constraints, 
        epsilon = epsilon, 
        verbose = verbose, 
        solver = solver)


def compute_terminal_ingredients_for_satellite(agent:Satellite, data:dict, num_decrease_samples:int, alpha:float, alpha_tol:float = 1e-8, compute_size_for_decrease:bool=True, compute_size_for_constraints:bool=True, epsilon:float=1.0, verbose:int=1, solver:str='CLARABEL'):
    """Design the terminal ingredients for a satellite.
    Assuming that for cooperation trajectories, the radial velocity is zero, and no tangential thrust is allowed.
    This couples the radial thust, angular velocity, and the radial position.
    
    Arguments:
    - agent: Satellite -- The satellite agent for which the terminal ingredients are designed.
    - data: dict -- The data dictionary.
    - num_decrease_samples: int -- The number of samples used to check the decrease condition.
    - alpha: float -- The initial guess for the decrease condition.
    - alpha_tol: float -- The tolerance for the decrease condition.
    - compute_size_for_decrease: bool -- Whether to compute the terminal set size for the decrease condition.
    - compute_size_for_constraints: bool -- Whether to compute the terminal set size for constraint satisfaction.
    - epsilon: float -- Tightening for the Lyapunov decrease condition.
    - verbose (int): 0: No printing; 1: Printing of solution stats; 2: Solver set to verbose (default is 1)
    - solver (str): Solver that is used to solve the problem, e.g. 'CLARABEL', 'MOSEK', 'OSQP', 'SCS' (default is 'MOSEK')
    """
    
    if not hasattr(agent, 'terminal_ingredients'):
        agent.terminal_ingredients = {}  # Initialize the dictionary.
    
    # We allow only for one point, the orbital radius at the reference orbit and the coasting angular velocity.
    # All inputs are set to zero.
    xT = [0.0, 0.0, 0.0, np.sqrt(agent.mu/(agent.r0)**3)]
    uT = [0.0, 0.0]
    grid = {'xT': [np.vstack(xT)], 'uT': [np.vstack(uT)]}
    
    def get_next_points(agent:Agent, grid_point:dict)->list:
        # Extract the grid point.
        xT = grid_point['xT']
        uT = grid_point['uT']
        
        # # Compute the next state.
        xTnext = np.vstack([agent.dynamics(xT, uT)])
        
        # The next input is also zero
        uTnext = np.vstack([0.0, 0.0])
                    
        return [{'xT': xTnext, 'uT': uTnext}] 
    
    # This turns the problem into a static one, i.e. no parameters need to be chosen.
    get_lpv_par = None

    next_point = get_next_points(agent, {'xT': np.vstack(xT), 'uT': np.vstack(uT)})[0]
    if np.linalg.norm(next_point['xT'][0] - xT[0]) > 1e-12 or np.linalg.norm(next_point['xT'][2] - xT[2]) > 1e-12 or np.linalg.norm(next_point['xT'][3] - xT[3]) > 1e-12:
        raise RuntimeError("Something went wrong in the satellite's dynamics.")
    # The matrices are then simply the Jacobians of the dynamics evaluated at the point, since they do not depend on the angular position, which is the only state that changes.
    A0, B0 = agent.compute_jacobians(xval=xT, uval=uT)
    A_LPV = {'static': np.array(A0)}
    B_LPV = {'static': np.array(B0)} 
           
    agent.terminal_ingredients.update({'get_lpv_par': get_lpv_par, 
                                       'A_LPV': A_LPV, 
                                       'B_LPV': B_LPV, 
                                       'Q':agent.stage_cost_weights['Q'], 
                                       'R':agent.stage_cost_weights['R'], 
                                       'get_next_points': get_next_points,
                                       'type': 'set'})

    prob = compute_generic_terminal_ingredients(
        agent, 
        grid, 
        num_decrease_samples=num_decrease_samples, 
        alpha=alpha, 
        alpha_tol=alpha_tol, 
        compute_size_for_decrease=compute_size_for_decrease, 
        compute_size_for_constraints=compute_size_for_constraints, 
        epsilon=epsilon, 
        parameter_threshold=0.0,
        verbose=verbose, 
        solver=solver)
    
    return prob


def compute_generic_terminal_ingredients(agent:Agent, grid:dict, num_decrease_samples:int, alpha:float, alpha_tol:float = 1e-8, compute_size_for_decrease:bool=True, compute_size_for_constraints:bool=True, epsilon:float=1.0, parameter_threshold:float=1e-10, verbose:int=1, solver:str='MOSEK') -> cvxpy.Problem:
    """Compute generic terminal ingredients following the scheme proposed in [1] for quadratic stage costs.
    
    The quadratic stage cost has the form: (x - xT).T @ Q @ (x - xT) + (u - uT).T @ R @ (u - uT)
    Multiple LMIs are set up and solved using CVXPY.
    The computed terminal ingredients are saved to the attribute 'terminal_ingredients' of the agent. 
    This attribute is overwritten if present.
    Currenlty, supports only polytopic constraints for the agent's state and input and cooperation reference. 
    
    Requires packages 'cvxpy', 'scipy', and 'casadi'.
    
    [1] 2020 - J. Koehler et al. - A Nonlinear Model Predictive Control Framework Using Reference Generic Terminal Ingredients - IEEE TAC. doi: 10.1109/TAC.2019.2949350
    
    Arguments:
        - agent (Agent): Agent for which the terminal ingredients are designed. 
            Must have the attribute terminal_ingredients (dict) with entries:
            - 'get_lpv_par' (casadi.Function): A function that takes a point on the cooperative trajectory (cas.MX.sym) called 'xT' and 'uT' and returns the parameters (cas.MX.sym) used in the quasi-LPV description, cf. [(11), (12); 1]. For example, 'thetas = get_lpv_par(xT=xT[2], uT=uT[2])', where thetas (dict) contains as keys the variables' names and as values the numerical value. Note that the individual thetas are scalars. If 'get_lpv_par' is set to None, then it is assumed that
            the LPV description is static, and the terminal matrix design is standard.
            - 'A_LPV' (dict): A dictionary containing as keys the names of the variables 'get_lpv_par' returns, and as respective values the matrix that is multiplied with that specific parameter to get the LPV dynamic matrix A, cf. [(11), 1]. Must also contain 'static' which is the static component (A0).
            - 'B_LPV' (dict): A dictionary containing as keys the names of the variables 'get_lpv_par' returns, and as respective values the matrix that is multiplied with that specific parameter to get the LPV dynamic matrix B, cf. [(11), 1]. Must also contain 'static' which is the static component (B0).
            - Q (np.ndarray): Weighting matrix for the state in the quadratic stage cost.
            - R (np.ndarray): Weighting matrix for the input in the quadratic stage cost.
            - 'get_next_points' (function): A function that returns the next points from a point on the grid. The function will be called using the agent as the first positional argument, and then keyword arguments; using the names of the variables in the grid and a value. If 'get_next_points' is not defined, then it is assumed that terminal ingredients for equilibria should be computed.
            For example, for a given index 'idx' to select the point on the grid, 'function({key: grid[key][idx] for key in grid})' is performed. Must return the next grid points, which is a list containing dictionaries with the same keys as the grid and the value of the points as values.
            If compute_size_for_constraints is True, then agent must have the attribute cooperation_constraints (dict) with entries:
            - 'Ax' (np.ndarray): Defining the left-hand side of pointwise-in-time polytopic constraints on the reference state: Ax <= b
            - 'bx' (np.ndarray): Defining the right-hand side of pointwise-in-time polytopic constraints on the reference state: Ax <= b
            - 'Au' (np.ndarray): Defining the left-hand side of pointwise-in-time polytopic constraints on the reference input: Au <= b
            - 'bu' (np.ndarray): Defining the right-hand side of pointwise-in-time polytopic constraints on the reference input: Au <= b
            If compute_size_for_constraints is True, then agent must have the attribute state_constraints (dict) and input_constraints (dict) with entries:
            - 'A' (np.ndarray): Defining the left-hand side of pointwise-in-time polytopic constraints on the state or input: Az <= b
            - 'b' (np.ndarray): Defining the right-hand side of pointwise-in-time polytopic constraints on the state or input: Az <= b
        - grid (dict): Containing a grid for the variables of the reference. Each key must correspond to the variable name in 'get_lpv_par' ('xT' and 'uT'), which will be explicitly called using these names. The values should be lists containing the respective part of the grid point. If 'get_lpv_par' is None, i.e. the LPV parametrization is static, the grid is ignored for the design of the terminal matrices, but it is required for the computation of the terminal set size, if 'compute_size_for_decrease' is set to True.
        - num_decrease_samples (int): Number of samples that are taken to check the decrease condition in the terminal set in order to determine the terminal set size.
        - alpha (float): A first guess and upper bound for the terminal set size.
        - alpha_tol (float): Tolerance of the terminal set size. If no terminal set size larger than or equal to this value can be found, the method fails. (default is 1e-8)
        - compute_size_for_decrease (bool): Whether to compute the terminal set size such that the decrease condition is satisfied. (default is True)
        - compute_size_for_constraints (bool): Whether to compute the terminal set size such that state and input constraints are satisfied. Since this is only supported for polytopic constraints on the state, input and cooperation reference, setting this to False skips that step and allows for manual adjustment of the terminal set size (the decrease condition is always ensured on the samples). (default is True)
        - epsilon (float): Tightening of the Lyapunov decrease equation the terminal ingredients need to satisfy, cf. [(10), 1]. (defaults to 1.0)
        - parameter_threshold (float): Threshold for the parameters of the LPV description. If the parameters are below this threshold, it is set to 0.0 to improve numerical stability. 
            Note that this corresponds to solving the LMIs with the decision matrix corresponding to this parameter set to zero. Hence, a feasible solution to the problem is admissible. (default is 1e-10).
        - verbose (int): 0: No printing; 1: Printing of solution stats; 2: Solver set to verbose (default is 1)
        - solver (str): Solver that is used to solve the problem, e.g. 'CLARABEL', 'MOSEK', 'OSQP', 'SCS' (default is 'MOSEK')
            
    Returns:
        - (cvxpy.Problem) Solution object returned by solving the optimization problem.
            The following is added to the agent's attribute 'terminal_ingredients':
                - 'X': A list of matrices that are multiplied with the parameters of the quasi-LPV description to obtain the terminal cost matrix, cf. [Prop. 1; 1].
                - 'Y': A list of matrices that are multiplied with the parameters of the quasi-LPV description used to obtain the terminal controller matrix, cf. [Prop. 1; 1].
                - 'size': A scalar determining the terminal set size, cf. [Sec. III.C; 1].
    """
    import cvxpy
    
    # Extract the functions.
    get_lpv_par = agent.terminal_ingredients['get_lpv_par']
    if 'get_next_points' in agent.terminal_ingredients:
        get_next_points = agent.terminal_ingredients['get_next_points']
    else:
        get_next_points = None

    # Extract the state dimension of the agent.
    n = agent.state_dim
    q = agent.input_dim
    # Extract the matrices of the stage cost.
    Q = agent.terminal_ingredients['Q']
    R = agent.terminal_ingredients['R']
    # Extract the dictionaries defining the LPV matrices.
    A_LPV = agent.terminal_ingredients['A_LPV']
    B_LPV = agent.terminal_ingredients['B_LPV']
    
    Qepsilonroot = scipy.linalg.sqrtm(Q + epsilon*np.eye(n))
    Rroot = scipy.linalg.sqrtm(R)
    # Ensure these matrices are symmetric.
    Qepsilonroot = cvxpy.Constant((Qepsilonroot + Qepsilonroot.T) / 2)
    Rroot = cvxpy.Constant((Rroot + Rroot.T) / 2)
    
    if 'vertices' in grid:
        # Vertices are supplied in the grid, so a convexification approach is assumed.

        # Create decision variables.
        X_min = cvxpy.Variable((n, n), name='Xmin', PSD=True)
        lambdas = {par_name: cvxpy.Variable((2*n, 2*n), name=par_name, PSD=True) for par_name in get_lpv_par.name_out()} 
        X_conc = {par_name : cvxpy.Variable((n, n), name=f'X_{par_name}') for par_name in get_lpv_par.name_out()}
        Y_conc = {par_name : cvxpy.Variable((q, n), name=f'Y_{par_name}') for par_name in get_lpv_par.name_out()}  
            
        X_dict = {'static': cvxpy.Variable((n, n), name='X0')}
        Y_dict = {'static': cvxpy.Variable((q, n), name='Y0')}
        if get_lpv_par:
            # Create a decision variable per parameter supplied by 'get_lpv_par'.
            for name_out in get_lpv_par.name_out():
                X_dict[name_out] = cvxpy.Variable((n, n), name=f'X_{name_out}')
                Y_dict[name_out] = cvxpy.Variable((q, n), name=f'Y_{name_out}')

        constraints = []
        # Add LMIs per vertex.
        for vertex in grid['vertices']:
            # Compute the dynamic matrices at the current vertex.
            A = A_LPV['static']
            B = B_LPV['static']
            # Also construct the decision variables.
            X = X_dict['static']
            Y = Y_dict['static']
            for par_name, par in vertex.items():
                A = A + float(par) * A_LPV[par_name]
                B = B + float(par) * B_LPV[par_name]
                X = X + float(par) * X_dict[par_name]
                Y = Y + float(par) * Y_dict[par_name]              

            # Add constraints.
            constraints.append(X >> 0)
            constraints.append(X >> X_min)
            constraints.append(X == X.T)
                  
            # Compute the RHS matrix of the LMI.
            lambda_sum = cvxpy.Constant(np.zeros((2*n, 2*n)))
            for name_out in get_lpv_par.name_out():
                lambda_sum = lambda_sum + vertex[name_out]**2 * lambdas[name_out]
                
            RHS = cvxpy.bmat([
                [lambda_sum,                                cvxpy.Constant(np.zeros((2*n, n + q)))],
                [cvxpy.Constant(np.zeros((n + q, 2*n))),    cvxpy.Constant(np.zeros((n + q, n + q)))]
            ])
                    
            # The next point can be all vertices.
            for next_vertex in grid['vertices']:
                X_next = X_dict['static']
                for par_name, par in next_vertex.items():
                    X_next = X_next + float(par) * X_dict[par_name]
                            
                LMI = cvxpy.bmat([
                    [X,                     X@A.T + Y.T@B.T,                    Qepsilonroot@X,                     (Rroot@Y).T],
                    [(X@A.T + Y.T@B.T).T,   X_next,                             cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.zeros((n,q)))],
                    [(Qepsilonroot@X).T,    cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.eye(n)),          cvxpy.Constant(np.zeros((n,q)))],
                    [Rroot@Y,               cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.eye(q))]
                    ])
            
                constraints.append(LMI >> RHS)
                constraints.append(X_next >> 0)
                constraints.append(X_next >> X_min)
                constraints.append(X_next == X_next.T)
            
        # Add the multi-concavity constraints.
        for par_name, par in vertex.items():
            LMI_conc = cvxpy.bmat([
                [ cvxpy.Constant(np.zeros((n, n))),                                     (A_LPV[par_name]@X_conc[par_name] + B_LPV[par_name]@Y_conc[par_name]).T ],
                [ A_LPV[par_name]@X_conc[par_name] + B_LPV[par_name]@Y_conc[par_name],  cvxpy.Constant(np.zeros((n, n)))]
            ])
            constraints.append(lambdas[par_name] >> LMI_conc)
    else:
        # Grid points are supplied in the grid.

        # Create decision variables.
        X_min = cvxpy.Variable((n, n), name='Xmin', PSD=True)
        
        X_dict = {'static': cvxpy.Variable((n, n), name='X0')}
        Y_dict = {'static': cvxpy.Variable((q, n), name='Y0')}
        if get_lpv_par:
            # Create a decision variable per parameter supplied by 'get_lpv_par'.
            for name_out in get_lpv_par.name_out():
                X_dict[name_out] = cvxpy.Variable((n, n), name=f'X_{name_out}')
                Y_dict[name_out] = cvxpy.Variable((q, n), name=f'Y_{name_out}')
                
        # Initialise all decision variables with zero, except for X0 and Y0
        for name, var in X_dict.items():
            if name != 'static':
                X_dict[name].value = np.zeros((n, n))  # Set to zero except for 'static'

        for name, var in Y_dict.items():
            if name != 'static':
                Y_dict[name].value = np.zeros((q, n))  # Set to zero except for 'static'

        constraints = []
        
        # Compute the length of the grid.
        if grid:
            grid_length = len(grid[next(iter(grid))])
            
        if grid and get_lpv_par:
            
            for idx in range(grid_length):
                grid_point = {key: grid[key][idx] for key in grid}  # Extract a grid point.
                thetas = get_lpv_par.call(grid_point)  # Get the LPV parameters for the grid points.
                thetas = {key: cvxpy.Constant(float(thetas[key])) if abs(thetas[key]) > parameter_threshold else 0.0 for key in thetas}  # Transform the values into scalars.
                
                # Compute the dynamic matrices at the current grid point.
                A = agent.terminal_ingredients['A_LPV']['static']
                B = agent.terminal_ingredients['B_LPV']['static']
                for key in thetas:
                    A = A + thetas[key]*agent.terminal_ingredients['A_LPV'][key]
                    B = B + thetas[key]*agent.terminal_ingredients['B_LPV'][key]

                # Construct the parametrisation from the decision variables.
                X = X_dict['static']
                Y = Y_dict['static']
                for theta in thetas:
                    X = X + thetas[theta]*X_dict[theta]
                    Y = Y + thetas[theta]*Y_dict[theta]

                # Add constraints.
                constraints.append(X >> 0)
                constraints.append(X >> X_min)
                constraints.append(X == X.T)
                
                # Compute the next points from this grid point.
                if get_next_points is not None:
                    next_grid_points = get_next_points(agent, grid_point)
                
                    for next_point in next_grid_points:          
                        next_thetas = get_lpv_par.call(next_point)  # Get the LPV parameters for the next grid points.
                        next_thetas = {key: cvxpy.Constant(float(next_thetas[key])) if abs(next_thetas[key]) > parameter_threshold else 0.0 for key in next_thetas}  # Transform the values into scalars.
                        X_next = X_dict['static']
                        for next_theta in next_thetas:
                            X_next = X_next + next_thetas[next_theta]*X_dict[next_theta]
                        # constraints.append(X_next >> 0)
                        constraints.append(X_next >> X_min)
                        constraints.append(X_next == X_next.T)
                            
                        LMI = cvxpy.bmat([
                            [X,                     X@A.T + Y.T@B.T,                    Qepsilonroot@X,                     (Rroot@Y).T],
                            [(X@A.T + Y.T@B.T).T,   X_next,                             cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.zeros((n,q)))],
                            [(Qepsilonroot@X).T,    cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.eye(n)),          cvxpy.Constant(np.zeros((n,q)))],
                            [Rroot@Y,               cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.eye(q))]
                            ])
                    
                        constraints.append(LMI >> 0)
                else:
                    # If no next reference points are provided, then the terminal ingredients are computed for equilibria.
                    LMI = cvxpy.bmat([
                            [X,                     X@A.T + Y.T@B.T,                    Qepsilonroot@X,                     (Rroot@Y).T],
                            [(X@A.T + Y.T@B.T).T,   X,                                  cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.zeros((n,q)))],
                            [(Qepsilonroot@X).T,    cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.eye(n)),          cvxpy.Constant(np.zeros((n,q)))],
                            [Rroot@Y,               cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.eye(q))]
                            ])
                    
                    constraints.append(LMI >> 0)
        else:
            # The problem is static, i.e. A and B are the same for all parameters defined by the grid points.
            
            # Get the LPV matrices, which are static.
            A = agent.terminal_ingredients['A_LPV']['static']
            B = agent.terminal_ingredients['B_LPV']['static']
            
            # Xmin can be directly used as the decision variable.
            X_dict['static'] = X_min
            X = X_dict['static']
            Y = Y_dict['static']
            
            # Set up the LMI.
            LMI = cvxpy.bmat([
                    [X,                     X@A.T + Y.T@B.T,                    Qepsilonroot@X,                     (Rroot@Y).T],
                    [(X@A.T + Y.T@B.T).T,   X,                                  cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.zeros((n,q)))],
                    [(Qepsilonroot@X).T,    cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.eye(n)),          cvxpy.Constant(np.zeros((n,q)))],
                    [Rroot@Y,               cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.eye(q))]
                    ])
            # Add the LMI to the constraints.
            constraints.append(LMI >> 0)

    # Form objective.
    obj = cvxpy.Minimize(-cvxpy.log_det(X_min))

    # Form and solve the problem.
    prob = cvxpy.Problem(obj, constraints)
    # Try the solver first, then try MOSEK.
    if solver != 'MOSEK':
        try:
            prob.solve(solver=solver, verbose=(verbose > 1))  # Get the optimal value.
        except:
            if verbose > 0:
                print(f"Solver {solver} failed. Trying MOSEK.")
            solver = 'MOSEK'
            prob.solve(solver=solver, verbose=(verbose > 1), warm_start=True)  # Get the optimal value.
    else:
        prob.solve(solver=solver, verbose=(verbose > 1), warm_start=True)

    if verbose > 0:
        print(f"Solving for Agent {agent.id} ------------------------------------")
        print(f"Problem with {len(constraints)} LMI constraints.")
        print("status:", prob.status)
        print("solver:", prob.solver_stats.solver_name)
        print("optimal value", prob.value)
        print("X_min \n", X_min.value)
        # Check if the solution is positive definite and compute its condition number.
        try:
            np.linalg.cholesky(X_min.value)
            pos_def = 'positive definite'
        except:
            pos_def = 'not positive definite'
        print(f"Solution is {pos_def} with minimum eigenvalue {min(np.linalg.eigvalsh(np.linalg.inv(X_min.value)))}")
        
    # Validate the solution.
    _validate_generic_terminal_ingredients_solution(agent, grid, X_dict, Y_dict, X_min, epsilon)

    agent.terminal_ingredients['X'] = {theta_name: X_dict[theta_name].value for theta_name in X_dict}
    agent.terminal_ingredients['Y'] = {theta_name: Y_dict[theta_name].value for theta_name in Y_dict}
    
    if compute_size_for_decrease:
        alpha1 = compute_terminal_set_size_cost_decrease(agent, grid, alpha, num_decrease_samples, alpha_tol=alpha_tol, verbose=verbose)
    else:
        alpha1 = alpha
        warnings.warn("Terminal set size does not ensure cost decrease. Make adjustments as needed.", UserWarning)

    if compute_size_for_constraints:   
        terminal_set_size = compute_terminal_set_size_constraint_satisfaction(agent, grid, alpha_tol, verbose, solver)
        terminal_set_size = min([alpha1, terminal_set_size])
        agent.terminal_ingredients['size'] = terminal_set_size  # Transfer the terminal set size to the agent.
    else:
        terminal_set_size = alpha1
        warnings.warn("Terminal set size does not ensure constraint satisfaction. Make adjustments as needed.", UserWarning)
        # Write the terminal set size on the agent.
        agent.terminal_ingredients['size'] = terminal_set_size

    return prob


def _validate_generic_terminal_ingredients_solution(agent, grid, X_dict, Y_dict, X_min, epsilon):
    """
    Validates the solution of compute_generic_terminal_ingredients by checking whether the LMI constraints are satisfied.
    The parameter_threshold in compute_generic_terminal_ingredients is ignored, i.e. also parameters below this threshold are considered.

    Parameters:
        agent: The agent object containing problem parameters.
        grid: The dictionary of grid points.
        X_dict: Dictionary of solved X decision variable values.
        Y_dict: Dictionary of solved Y decision variable values.
        X_min: The minimum X matrix.
        epsilon: Small positive number added to Q.

    Returns:
        validity: Boolean indicating if the solution is valid.
        failed_points: List of grid points where constraints were violated.
    """
    if 'vertices' in grid:
        # This method is not yet implemented for the convexification approach.
        # Warn the user and return that the solution is valid.
        warnings.warn("Validation for convexification approach is not implemented yet. Skipping validation.", UserWarning)
        return True, []
    
    get_lpv_par = agent.terminal_ingredients['get_lpv_par']
    get_next_points = agent.terminal_ingredients.get('get_next_points', None)
    
    n = agent.state_dim
    q = agent.input_dim
    Q = agent.terminal_ingredients['Q']
    R = agent.terminal_ingredients['R']
    
    X_min = X_min.value
    X_min = (X_min + X_min.T)/2
    
    try:
        scipy.linalg.cholesky(X_min)
    except np.linalg.LinAlgError:
        raise ValueError("X_min is not positive definite!")
    
    lambda_min = np.min(np.linalg.eigvalsh(X_min))
    # Check X > X_min against a shifted X_min which is positive definite.
    X_min_shifted = X_min - (0.5 * lambda_min) * np.eye(X_min.shape[0])
    try:
        scipy.linalg.cholesky(X_min_shifted)
    except np.linalg.LinAlgError:
        raise ValueError("X_min_shifted is not positive definite!")

    # Compute the length of the grid
    if grid:
        grid_length = len(grid[next(iter(grid))])

    # Iterate over grid points
    if grid and get_lpv_par:
        for idx in range(grid_length):
            grid_point = {key: grid[key][idx] for key in grid}  # Extract a grid point
            thetas = get_lpv_par.call(grid_point)  # Get LPV parameters
            thetas = {key: float(thetas[key]) for key in thetas}

            # Compute dynamic matrices
            A = agent.terminal_ingredients['A_LPV']['static']
            B = agent.terminal_ingredients['B_LPV']['static']
            for key in thetas:
                A = A + thetas[key] * agent.terminal_ingredients['A_LPV'][key] 
                B = B + thetas[key] * agent.terminal_ingredients['B_LPV'][key]

            # Compute Q and R square roots
            Qepsilonroot = scipy.linalg.sqrtm(Q + epsilon * np.eye(n))
            Rroot = scipy.linalg.sqrtm(R)
            Qepsilonroot = (Qepsilonroot + Qepsilonroot.T) / 2
            Rroot = (Rroot + Rroot.T) / 2

            # Construct decision variables at this grid point
            X = X_dict['static'].value
            Y = Y_dict['static'].value
            for theta in thetas:
                X = X + thetas[theta] * X_dict[theta].value
                Y = Y + thetas[theta] * Y_dict[theta].value
                
            X = (X + X.T)/2
                
            # Check if X > X_min (positive definite difference)
            try:
                scipy.linalg.cholesky(X - X_min_shifted)
            except np.linalg.LinAlgError:
                raise ValueError(f"Constraint X > X_min failed at grid point {grid_point}")

            # Compute next points if provided
            if get_next_points is not None:
                next_grid_points = get_next_points(agent, grid_point)
                for next_point in next_grid_points:
                    next_thetas = get_lpv_par.call(next_point)
                    next_thetas = {key: float(next_thetas[key]) for key in next_thetas}

                    X_next = X_dict['static'].value
                    for next_theta in next_thetas:
                        X_next = X_next + next_thetas[next_theta] * X_dict[next_theta].value
                    X_next = (X_next + X_next.T)/2
                        
                    try:
                        scipy.linalg.cholesky(X_next - X_min_shifted)
                    except np.linalg.LinAlgError:
                        raise ValueError(f"Constraint X_next > X_min failed at grid point {grid_point}")

                    # Form LMI
                    LMI = np.block([
                        [X, X @ A.T + Y.T @ B.T, Qepsilonroot @ X, (Rroot @ Y).T],
                        [(X @ A.T + Y.T @ B.T).T, X_next, np.zeros((n, n)), np.zeros((n, q))],
                        [(Qepsilonroot @ X).T, np.zeros((n, n)), np.eye(n), np.zeros((n, q))],
                        [Rroot @ Y, np.zeros((q, n)), np.zeros((q, n)), np.eye(q)]
                    ])

                    # Check if LMI is positive definite up to a tolerance using the Cholesky decomposition.
                    try:
                        scipy.linalg.cholesky(LMI + 1e-8 * np.eye(LMI.shape[0])) 
                    except:
                        raise ValueError(f"Constraint LMI failed at grid point {grid_point} with minimal eigenvalue {np.min(np.linalg.eigvalsh(LMI))}")

            else:
                # Check equilibrium case
                LMI = np.block([
                    [X, X @ A.T + Y.T @ B.T, Qepsilonroot @ X, (Rroot @ Y).T],
                    [(X @ A.T + Y.T @ B.T).T, X, np.zeros((n, n)), np.zeros((n, q))],
                    [(Qepsilonroot @ X).T, np.zeros((n, n)), np.eye(n), np.zeros((n, q))],
                    [Rroot @ Y, np.zeros((q, n)), np.zeros((q, n)), np.eye(q)]
                ])

                # Check if LMI is positive definite up to a tolerance using the Cholesky decomposition.
                try:
                    scipy.linalg.cholesky(LMI + 1e-8 * np.eye(LMI.shape[0])) 
                except:
                    raise ValueError(f"Constraint LMI failed at grid point {grid_point} with minimal eigenvalue {np.min(np.linalg.eigvalsh(LMI))}")

    else:
        # Single equilibrium case
        A = agent.terminal_ingredients['A_LPV']['static']
        B = agent.terminal_ingredients['B_LPV']['static']

        Qepsilonroot = scipy.linalg.sqrtm(Q + epsilon * np.eye(n))
        Rroot = scipy.linalg.sqrtm(R)

        X = X_dict['static'].value
        Y = Y_dict['static'].value
        
        # Check if X > X_min
        try:
                scipy.linalg.cholesky(X - X_min_shifted)
        except np.linalg.LinAlgError:
            raise ValueError("Constraint X > X_min failed in equilibrium case")
        
        LMI = np.block([
            [X, X @ A.T + Y.T @ B.T, Qepsilonroot @ X, (Rroot @ Y).T],
            [(X @ A.T + Y.T @ B.T).T, X, np.zeros((n, n)), np.zeros((n, q))],
            [(Qepsilonroot @ X).T, np.zeros((n, n)), np.eye(n), np.zeros((n, q))],
            [Rroot @ Y, np.zeros((q, n)), np.zeros((q, n)), np.eye(q)]
        ])

        # Check if LMI is positive definite up to a tolerance using the Cholesky decomposition.
        # Try again with a shifted LMI since it does not have to hold strictly.
        try:
            scipy.linalg.cholesky(LMI + 1e-8 * np.eye(LMI.shape[0]))
        except:
            raise ValueError(f"LMI not feasible with {np.min(np.linalg.eigvalsh(LMI))}.")



def compute_terminal_set_size_cost_decrease(agent:Agent, grid:dict, alpha:float, num_decrease_samples:int, alpha_tol:float = 1e-8, change_in_cost_tol:float = 1e-8, verbose:int=1) -> float:
    """Compute the terminal set size such that the terminal cost decreases, cf [Sec. III.C; 1].

    This is based on gridding the references and evaluating the cost decrease condition on the grid.
    If the cost does not decrease, the terminal set size is reduced until alpha_tol, at which point the method fails.
        
    [1] 2020 - J. Koehler et al. - A Nonlinear Model Predictive Control Framework Using Reference Generic Terminal Ingredients - IEEE TAC. doi: 10.1109/TAC.2019.2949350
    
    Arguments:
        - agent (Agent): Agent for which the terminal ingredients are designed. 
        Must have the attribute terminal_ingredients (dict) with entries:
            - 'get_lpv_par' (casadi.Function or None): A function that takes a point on the cooperative trajectory (cas.MX.sym) called 'xT' and 'uT' and returns the parameters (cas.MX.sym) used in the quasi-LPV description, cf. [(11), (12); 1]. For example, 'thetas = get_lpv_par(xT=xT[2], uT=uT[2])', where thetas (dict) contains as keys the variables' names and as values the numerical value. Note that the individual thetas are scalars. If set to None, then it is assumed that the LPV description is static.
            - 'X': A list of matrices that are multiplied with the parameters of the quasi-LPV description to obtain the terminal cost matrix, cf. [Prop. 1; 1].
            - 'Y': A list of matrices that are multiplied with the parameters of the quasi-LPV description used to obtain the terminal controller matrix, cf. [Prop. 1; 1].
            For example, for a given index 'idx' to select the point on the grid, 'function({key: grid[key][idx] for key in grid})' is performed. Must return the next grid points, which is a list containing dictionaries with the same keys as the grid and the value of the points as values.
        - alpha (float): A first guess and upper bound for the terminal set size.
        - num_decrease_samples (int): Number of samples that are taken to check the decrease condition in the terminal set in order to determine the terminal set size.
        - grid (dict): Containing a grid for the variables of the reference. Each key must correspond to the variable name in 'get_lpv_par' ('xT' and 'uT'), which will be explicitly called using these names. The values should be lists containing the respective part of the grid point.
        - alpha_tol (float): Tolerance of the terminal set size. If no terminal set size larger than or equal to this value can be found, the method fails. (default is 1e-8)
        - change_in_cost_tol (float): Tolerance for the change in cost. If the change in cost is larger than this value, the terminal set size is reduced. (default is 1e-8)
        - verbose (int): 0: No printing; 1: Printing of solution stats; 2: Solver set to verbose (default is 1)
            
    Returns:
        - (float) The computed terminal set size
    """
    if 'vertices' in grid:
        grid = grid['reference_points']  # Extract the reference points from the grid.
    
    grid_length = len(grid[next(iter(grid))])  # Compute the length of the grid.
    
    if type(num_decrease_samples) is float:
        num_decrease_samples = int(num_decrease_samples)
    elif type(num_decrease_samples) is not int:
        raise TypeError(f"num_decrease_samples must be int, but is {type(num_decrease_samples)}!")
        
    # Extract functions.
    get_lpv_par = agent.terminal_ingredients['get_lpv_par']
    if 'get_next_points' in agent.terminal_ingredients:
        get_next_points = agent.terminal_ingredients['get_next_points']
    else:
        get_next_points = None
    
    ## Compute the terminal set size using [Alg. 1, 1]:
    # Initialize binary search bounds.
    alpha_low = alpha_tol/2  # Lower bound on feasible alpha; allow it to fall below the tolerance at which point the method fails.
    alpha_high = alpha     # Initial upper bound.
    tolerance = alpha_tol        # Desired precision on alpha.

    # Iterate over all grid points.
    for idx in range(grid_length):
        grid_point = {key: grid[key][idx] for key in grid}  # Extract a grid point.

        # Convert list entries to numpy arrays if needed.
        for key in grid_point:
            if isinstance(grid_point[key], list):
                grid_point[key] = np.vstack(grid_point[key])

        # Compute the terminal cost and control matrices.
        X = agent.terminal_ingredients['X']
        Y = agent.terminal_ingredients['Y']
        P = X['static'].copy()
        K = Y['static'].copy()
        if get_lpv_par is not None:
            thetas = get_lpv_par.call(grid_point)
            thetas = {key: np.array(thetas[key]).item() for key in thetas}
            for theta_name in thetas:
                P += thetas[theta_name] * X[theta_name]
                K += thetas[theta_name] * Y[theta_name]
        P = (P + P.T) / 2  # Ensure symmetry.

        # Validate that P is positive definite.
        try:
            np.linalg.cholesky(P)
        except:
            raise ValueError(f"Ill-defined terminal ingredients: terminal cost matrix is not positive definite for {grid_point['xT']} and {grid_point['uT']}.")

        # Invert P to get the ellipsoid definition.
        P_inv = np.linalg.inv(P)
        L = np.linalg.cholesky(P_inv)  # Cholesky factor for sampling.
        n = P_inv.shape[0]

        # Compute the terminal control gain K * P^{-1}.
        K = K @ P_inv
        L_inv = np.linalg.inv(L)

        # Determine next reference points if applicable.
        if get_next_points is not None:
            next_grid_points = get_next_points(agent, grid_point)
            if not next_grid_points:
                continue  # Skip if no next points are provided.
        else:
            next_grid_points = None

        # Binary search loop for terminal set size.
        while alpha_high - alpha_low > tolerance:
            alpha_candidate = 0.5 * (alpha_low + alpha_high)
            all_samples_ok = True  # Assume all samples pass until one fails.

            for _ in range(num_decrease_samples):
                if not all_samples_ok:
                    break
                # Sample from inside the ellipsoid of size alpha_candidate.
                v = np.random.randn(n)
                v /= np.linalg.norm(v)
                scale = np.random.uniform(0, 1) ** (1.0 / n)
                xdelta = np.sqrt(alpha_candidate) * (L_inv.T @ (scale * v))
                xdelta = xdelta.reshape((-1, 1))
                xT = grid_point['xT']
                x = xT + xdelta
                uT = grid_point['uT']
                kf = uT + K @ (x - xT)

                # Simulate one step forward.
                xnext = agent.dynamics(x=x, u=kf)[agent.dynamics.name_out()[0]]

                if next_grid_points is not None:
                    for next_grid_point in next_grid_points:
                        for key in next_grid_point:
                            if isinstance(next_grid_point[key], list):
                                next_grid_point[key] = np.vstack(next_grid_point[key])
                        xTnext = next_grid_point['xT']
                        Pn = X['static'].copy()
                        if get_lpv_par is not None:
                            thetas = get_lpv_par.call(next_grid_point)
                            thetas = {key: np.array(thetas[key]).item() for key in thetas}
                            for theta_name in thetas:
                                Pn += thetas[theta_name] * X[theta_name]
                        Pn = (Pn + Pn.T) / 2
                        try:
                            np.linalg.cholesky(Pn)
                        except:
                            raise ValueError(f"Ill-defined terminal ingredients: terminal cost matrix is not positive definite for {next_grid_point['xT']} and {next_grid_point['uT']}.")
                        Pn_inv = np.linalg.inv(Pn)
                        delta_cost = (xnext - xTnext).T @ Pn_inv @ (xnext - xTnext) - (x - xT).T @ P_inv @ (x - xT)
                        stage_cost_val = agent.stage_cost(x=x, u=kf, xT=xT, uT=uT)[agent.stage_cost.name_out()[0]]
                        change_in_cost = delta_cost + stage_cost_val
                        if change_in_cost > change_in_cost_tol:
                            all_samples_ok = False
                            break
                else:
                    delta_cost = (xnext - xT).T @ P_inv @ (xnext - xT) - (x - xT).T @ P_inv @ (x - xT)
                    stage_cost_val = agent.stage_cost(x=x, u=kf, xT=xT, uT=uT)[agent.stage_cost.name_out()[0]]
                    change_in_cost = delta_cost + stage_cost_val
                    if change_in_cost > change_in_cost_tol:
                        all_samples_ok = False
                        break  # Stop sampling if one violation is found.

            if all_samples_ok:
                alpha_low = alpha_candidate  # Try larger size.
            else:
                alpha_high = alpha_candidate  # Too large, shrink.

        alpha1 = alpha_low  # Best feasible alpha after binary search.
        
        if verbose >= 2:
            print(f'Point {idx} of {grid_length}: current terminal set size = {alpha1}')

    if alpha1 < alpha_tol:
        raise RuntimeError(f"Could not compute terminal set size that ensures decrease ≥ {alpha_tol}. Best candidate: {alpha1}.")
    if verbose > 0:
        print(f"Computed terminal set size {alpha1} with {num_decrease_samples} samples.")
    return alpha1


def compute_terminal_set_size_constraint_satisfaction(agent:Agent, grid:dict, alpha_tol:float = 1e-8, verbose:int=1, solver:str='CLARABEL'):
    """Compute the terminal set size that satisfies the constraints, cf [Sec. III.C; 1].
    
    This is only supported for polytopic constraints on the state, input and cooperation reference.
        
    [1] 2020 - J. Koehler et al. - A Nonlinear Model Predictive Control Framework Using Reference Generic Terminal Ingredients - IEEE TAC. doi: 10.1109/TAC.2019.2949350
    
    Arguments:
        - agent (Agent): Agent for which the terminal ingredients are designed. 
        Must have the attribute terminal_ingredients (dict) with entries:
            - 'get_lpv_par' (casadi.Function): A function that takes a point on the cooperative trajectory (cas.MX.sym) called 'xT' and 'uT' and returns the parameters (cas.MX.sym) used in the quasi-LPV description, cf. [(11), (12); 1]. For example, 'thetas = get_lpv_par(xT=xT[2], uT=uT[2])', where thetas (dict) contains as keys the variables' names and as values the numerical value. Note that the individual thetas are scalars. If 'get_lpv_par' is set to None, then it is assumed that the LPV description is static.
            - 'X': A list of matrices that are multiplied with the parameters of the quasi-LPV description to obtain the terminal cost matrix, cf. [Prop. 1; 1].
            - 'Y': A list of matrices that are multiplied with the parameters of the quasi-LPV description used to obtain the terminal controller matrix, cf. [Prop. 1; 1].
            For example, for a given index 'idx' to select the point on the grid, 'function({key: grid[key][idx] for key in grid})' is performed. Must return the next grid points, which is a list containing dictionaries with the same keys as the grid and the value of the points as values.
        Furthermore the agent must have the attribute cooperation_constraints (dict) with entries:
            - 'Ax' (np.ndarray): Defining the left-hand side of pointwise-in-time polytopic constraints on the reference state: Ax <= b
            - 'bx' (np.ndarray): Defining the right-hand side of pointwise-in-time polytopic constraints on the reference state: Ax <= b
            - 'Au' (np.ndarray): Defining the left-hand side of pointwise-in-time polytopic constraints on the reference input: Au <= b
            - 'bu' (np.ndarray): Defining the right-hand side of pointwise-in-time polytopic constraints on the reference input: Au <= b
        In addition, the agent must have the attribute state_constraints (dict) and input_constraints (dict) with entries:
            - 'A' (np.ndarray): Defining the left-hand side of pointwise-in-time polytopic constraints on the state or input: Az <= b
            - 'b' (np.ndarray): Defining the right-hand side of pointwise-in-time polytopic constraints on the state or input: Az <= b
        - grid (dict): Containing a grid for the variables of the reference. Each key must correspond to the variable name in 'get_lpv_par' ('xT' and 'uT'), which will be explicitly called using these names. The values should be lists containing the respective part of the grid point.
        - alpha_tol (float): Tolerance of the terminal set size. If no terminal set size larger than or equal to this value can be found, the method fails. (default is 1e-8)
        - verbose (int): 0: No printing; 1: Printing of solution stats; 2: Solver set to verbose (default is 1)
        - solver (str): Solver that is used to solve the problem, e.g. 'CLARABEL', 'MOSEK', 'OSQP', 'SCS' (default is 'CLARABEL')
            
    Returns:
        - (float) The computed terminal set size.
    """
    import cvxpy
    
    if 'vertices' in grid:
        grid = grid['reference_points']  # Extract the reference points from the grid.
        
    # Extract the functions.
    get_lpv_par = agent.terminal_ingredients['get_lpv_par']
    # Compute the length of the grid.
    grid_length = len(grid[next(iter(grid))])
    
    ## Compute the largest terminal set size such that the constraints are satisfied, cf. [(18), 1]:    
    L_r_1 = np.hstack((agent.state_constraints["A"], np.zeros((agent.state_constraints["b"].shape[0], agent.input_dim))))
    L_r_2 = np.hstack([np.zeros((agent.input_constraints["b"].shape[0], agent.state_dim)), agent.input_constraints["A"]])
    L_r = np.vstack([L_r_1, L_r_2])

    l_r = np.vstack([agent.state_constraints["b"], agent.input_constraints["b"]])

    alpha2 = cvxpy.Variable(1)
    alpha2.value = np.array([1e-6])
    obj_tss = cvxpy.Maximize(alpha2)
    constraints_tss = [alpha2 >= alpha_tol]
    
    for idx in range(grid_length):
        grid_point = {key: grid[key][idx] for key in grid}  # Extract a grid point.
        
        r = np.vstack([np.vstack(grid_point['xT']), np.vstack(grid_point['uT'])])
        
        X = agent.terminal_ingredients['X']
        Y = agent.terminal_ingredients['Y']
        # Compute the terminal cost matrix and the terminal control matrix.
        P = X['static'].copy()
        K = Y['static'].copy()
        
        if get_lpv_par is not None:
            thetas = get_lpv_par.call(grid_point)  # Get the LPV parameters for the grid points.
            thetas = {key: np.array(thetas[key]).item() for key in thetas}  # Transform the values into scalars.
            for theta_name in thetas:
                P = P + thetas[theta_name]*X[theta_name]
                K = K + thetas[theta_name]*Y[theta_name]
        P = 0.5*(P + P.T)

        # Compute the terminal control matrix.
        Pf = np.linalg.inv(P)
        Pf = 0.5*(Pf + Pf.T)
        K = K@Pf
        
        mult = scipy.linalg.sqrtm(P)@np.hstack([np.eye(agent.state_dim), K.T])

        for j in range(l_r.shape[0]):
            right_hand_side = (l_r[j] - L_r[j,:]@r)**2
            # Note that P is not inverted since the square root of the inverse of the terminal weight matrix is needed.
            left_hand_side = cvxpy.power(cvxpy.norm( mult @L_r[j,:].T), 2)
            constraints_tss.append(left_hand_side*alpha2 <= right_hand_side)

    # Form and solve the problem.
    prob_tss = cvxpy.Problem(obj_tss, constraints_tss)
    # Try the solver. If it fails, try MOSEK.
    if solver != 'MOSEK':
        try:
            prob_tss.solve(solver=solver)
        except:
            if verbose > 0:
                print(f'Solver {solver} failed. Trying MOSEK.')
            prob_tss.solve(solver='MOSEK', warm_start=True)
    else:
        prob_tss.solve(solver='MOSEK', warm_start=True)
    if verbose > 0:
        print(f"Computed terminal set size {prob_tss.value} for constraint satisfaction.")
        print("status:", prob_tss.status)
        print("solver:", prob_tss.solver_stats.solver_name)
        print("optimal value", prob_tss.value)

    return alpha2.value


def double_integrator_warm_start_at_t0(agents, N, T):
    warm_start = {}
    for agent in agents:
        p = agent.output_dim
        n = agent.state_dim
        q = agent.input_dim
        
        x = np.copy(agent.current_state)
        u = np.zeros((q, 1))
        
        x_ws = np.zeros((n*N, 1))
        u_ws = np.zeros((q*N, 1))
        yT_ws = np.zeros((p*T, 1))
        xT_ws = np.zeros((n*T, 1))
        uT_ws = np.zeros((q*T, 1))
        
        for i in range(N):
            x = agent.dynamics(x, u)
            x_ws[i*n:(i+1)*n] = x
        
        x = np.copy(agent.current_state)
        for i in range(T):
            xT_ws[i*n:(i+1)*n] = x
            x = agent.dynamics(x, u)
        
        for i in range(T):
            uT_ws[i*q:(i+1)*q] = u
        
        for i in range(N):
            u_ws[i*q:(i+1)*q] = u
            
        for i in range(T):
            yT_ws[i*p:(i+1)*p] = agent.output_map(xT_ws[i*n:(i+1)*n], uT_ws[i*q:(i+1)*q])
            
        # Reassign the new trajectories.
        warm_start[f'A{agent.id}_x'] = x_ws
        warm_start[f'A{agent.id}_u'] = u_ws
        warm_start[f'A{agent.id}_yT'] = yT_ws
        warm_start[f'A{agent.id}_xT'] = xT_ws
        warm_start[f'A{agent.id}_uT'] = uT_ws
        
    return warm_start


def satellite_warm_start_at_t0(agents, N, T):
    warm_start = {}
    for agent in agents:
        p = agent.output_dim
        n = agent.state_dim
        q = agent.input_dim
        
        x = agent.current_state
        if T == 1:
            u = np.vstack([agent.mu/((agent.current_state[0] + agent.r0)**2)*agent.m, 0.])
        else:
            u = np.zeros((q, 1))
        
        x_ws = np.zeros((n*N, 1))
        u_ws = np.zeros((q*N, 1))
        yT_ws = np.zeros((p*T, 1))
        xT_ws = np.zeros((n*T, 1))
        uT_ws = np.zeros((q*T, 1))
        
        for i in range(N):
            x = agent.dynamics(x, u)
            x_ws[i*n:(i+1)*n] = x
        
        x = agent.current_state
        for i in range(T):
            xT_ws[i*n:(i+1)*n] = x
            x = agent.dynamics(x, u)
        
        for i in range(T):
            uT_ws[i*q:(i+1)*q] = u
        
        for i in range(N):
            u_ws[i*q:(i+1)*q] = u
            
        for i in range(T):
            yT_ws[i*p:(i+1)*p] = agent.output_map(xT_ws[i*n:(i+1)*n], uT_ws[i*q:(i+1)*q])
            
        # Reassign the new trajectories.
        warm_start[f'A{agent.id}_x'] = x_ws
        warm_start[f'A{agent.id}_u'] = u_ws
        warm_start[f'A{agent.id}_yT'] = yT_ws
        warm_start[f'A{agent.id}_xT'] = xT_ws
        warm_start[f'A{agent.id}_uT'] = uT_ws
        
    return warm_start


def quadrotor_warm_start_at_t0(agents, N, T):
    warm_start = {}
    for agent in agents:
        p = agent.output_dim
        n = agent.state_dim
        q = agent.input_dim
        
        x = np.copy(agent.current_state)
        u = np.vstack([0.0, 0.0, 9.81/0.91])
        
        x_ws = np.zeros((n*N, 1))
        u_ws = np.zeros((q*N, 1))
        yT_ws = np.zeros((p*T, 1))
        xT_ws = np.zeros((n*T, 1))
        uT_ws = np.zeros((q*T, 1))
        
        for i in range(N):
            x = agent.dynamics(x, u)
            x_ws[i*n:(i+1)*n] = x
        
        x = np.copy(agent.current_state)
        for i in range(T):
            xT_ws[i*n:(i+1)*n] = x
            x = agent.dynamics(x, u)
        
        for i in range(T):
            uT_ws[i*q:(i+1)*q] = u
        
        for i in range(N):
            u_ws[i*q:(i+1)*q] = u
            
        for i in range(T):
            yT_ws[i*p:(i+1)*p] = agent.output_map(xT_ws[i*n:(i+1)*n], uT_ws[i*q:(i+1)*q])
            
        # Reassign the new trajectories.
        warm_start[f'A{agent.id}_x'] = x_ws
        warm_start[f'A{agent.id}_u'] = u_ws
        warm_start[f'A{agent.id}_yT'] = yT_ws
        warm_start[f'A{agent.id}_xT'] = xT_ws
        warm_start[f'A{agent.id}_uT'] = uT_ws
        
        warm_start[f'A{agent.id}_radius'] = 2.0
        warm_start[f'A{agent.id}_yT_centre'] = x[0:3]

    return warm_start


def quadrotor_warm_start_at_switching_time(agents, N, T, N_old, T_old, terminal_ingredients_type):
    """Provide a warm start for the quadrotor agents at the switching time.
    This warm start will probably not be feasible with respect to terminal constraints and constraints on the cooperation reference.
    The prediction horizon N and the periodicity of the cooperative task T change at the switching time.
    Hence, we append parts of the previously optimal cooperation reference to the quadrotor's trajectory to achieve a trajectory with N steps,
    and we prune the cooperation reference to T steps.
    """
    # Only consider the case where T == 1.
    if T != 1:
        warnings.warn("The warm start at the switching time is only implemented for T == 1.")
        # Try the unchanged previously optimal solution.
    else:
        for agent in agents:
            p = agent.output_dim
            n = agent.state_dim
            q = agent.input_dim
            x_ws = agent.MPC_sol[f'A{agent.id}_x']
            u_ws = agent.MPC_sol[f'A{agent.id}_u']
            yT_ws = agent.MPC_sol[f'A{agent.id}_yT']
            xT_ws = agent.MPC_sol[f'A{agent.id}_xT']
            uT_ws = agent.MPC_sol[f'A{agent.id}_uT']
            
            # Remove all other entries in MPC_sol which are not needed anymore.
            for name in list(agent.MPC_sol):
                if name.endswith('radius') or name.endswith('centre'):
                    del agent.MPC_sol[name]
            
            # Construct the standard state and input candidate sequence.
            if terminal_ingredients_type == 'set':
                tau = N_old%T_old
                get_lpv_par = agent.terminal_ingredients['get_lpv_par']
                if get_lpv_par is None:
                    thetas = []
                else:
                    thetas = get_lpv_par(xT=xT_ws[tau*n : (tau+1)*n, 0], uT=uT_ws[tau*q : (tau+1)*q, 0])  # Get the LPV parameters for the grid points.
                    thetas = {key: np.array(thetas[key]).item() for key in thetas}  # Transform the values into scalars.
                X = agent.terminal_ingredients['X']
                Y = agent.terminal_ingredients['Y']
                # Compute the terminal cost matrix and the terminal control matrix.
                P = X['static'].copy()
                K = Y['static'].copy()
                for theta_name in thetas:
                    if np.linalg.norm(X[theta_name]) < 1e-10:
                        # Ignore matrices close to zero.
                        pass
                    else:
                        P = P + thetas[theta_name]*X[theta_name]
                    if np.linalg.norm(Y[theta_name]) < 1e-10:
                        # Ignore matrices close to zero.
                        pass
                    else:
                        K = K + thetas[theta_name]*Y[theta_name]
                if thetas:
                    # Use CasADi to compute the inverse of P if it is parameter dependent.
                    P = cas.inv(P)  # Invert P to get the terminal cost matrix.
                else:
                    P = np.linalg.inv(P)
                # Ensure that P is symmetric.
                P = 0.5*(P + P.T)
                K = K@P  # Compute the terminal control matrix.
                
                kf = uT_ws[tau*q : (tau+1)*q, :] + K@(x_ws[-n:] - xT_ws[tau*n:(tau+1)*n])  # Compute the terminal control input.
                xnext = agent.dynamics(x=x_ws[-n:], u=kf)[agent.dynamics.name_out()[0]]  # Compute the next point if the terminal control input is applied.
                # Shift the input and append the terminal control input.
                u_ws = np.vstack([u_ws[q:], kf])
                # Shift the state trajectory and append the next state based on the terminal control input.
                x_ws = np.vstack([x_ws[n:], xnext])
                
                # Do this until we have filled the input sequence with N steps.
                while u_ws.shape[0] < N*q:
                    tau += 1
                    
                    if get_lpv_par is None:
                        thetas = []
                    else:
                        thetas = get_lpv_par(xT=xT_ws[tau*n : (tau+1)*n, 0], uT=uT_ws[tau*q : (tau+1)*q, 0])  # Get the LPV parameters for the grid points.
                        thetas = {key: np.array(thetas[key]).item() for key in thetas}  # Transform the values into scalars.
                    # Compute the terminal cost matrix and the terminal control matrix.
                    P = X['static'].copy()
                    K = Y['static'].copy()
                    for theta_name in thetas:
                        if np.linalg.norm(X[theta_name]) < 1e-10:
                            # Ignore matrices close to zero.
                            pass
                        else:
                            P = P + thetas[theta_name]*X[theta_name]
                        if np.linalg.norm(Y[theta_name]) < 1e-10:
                            # Ignore matrices close to zero.
                            pass
                        else:
                            K = K + thetas[theta_name]*Y[theta_name]
                    if thetas:
                        # Use CasADi to compute the inverse of P if it is parameter dependent.
                        P = cas.inv(P)  # Invert P to get the terminal cost matrix.
                    else:
                        P = np.linalg.inv(P)
                    # Ensure that P is symmetric.
                    P = 0.5*(P + P.T)
                    K = K@P  # Compute the terminal control matrix.
                    
                    kf = uT_ws[tau*q : (tau+1)*q, :] + K@(x_ws[-n:] - xT_ws[tau*n:(tau+1)*n])  # Compute the terminal control input.
                    xnext = agent.dynamics(x=x_ws[-n:], u=kf)[agent.dynamics.name_out()[0]]  # Compute the next point if the terminal control input is applied.
                    # Append the new input and state.
                    u_ws = np.vstack([u_ws, kf])
                    # Shift the state trajectory and append the next state based on the terminal control input.
                    x_ws = np.vstack([x_ws, xnext])
            else:
                raise NotImplementedError("The warm start at the switching time is only implemented for terminal ingredients of type 'set'. "
                                          + "As a workaround, keep N and T constant at the switching time and do not use this method.")
        
            # For the cooperation reference, use the new terminal part.
            tau = N_old%T_old
            yT_ws = yT_ws[tau*p : (tau+1)*p, :]
            xT_ws = xT_ws[tau*n : (tau+1)*n, :]
            uT_ws = uT_ws[tau*q : (tau+1)*q, :]
            
            # Reassign the new trajectories.
            agent.MPC_sol[f'A{agent.id}_x'] = x_ws
            agent.MPC_sol[f'A{agent.id}_u'] = u_ws
            agent.MPC_sol[f'A{agent.id}_yT'] = yT_ws
            agent.MPC_sol[f'A{agent.id}_xT'] = xT_ws
            agent.MPC_sol[f'A{agent.id}_uT'] = uT_ws

    # Also update copies neighbours hold. Note that this would require communication.
    # Each agent sends an update of their trajectory to neighbours in order to synchronise the warm start.
    # If one considers a decentralised optimisation algorithm based on ADMM which features an averaging step
    # requiring at least two communication phases per iteration, then this additional communication should be negligible.
    # If one would like to avoid this, then the previously optimal solution could be used to warm start, i.e. no change
    # in 'agent.MPC_sol' should be made.
    
    for agent in agents:
        for nghbr in agent.neighbours:
            for name in [f'A{agent.id}_x', f'A{agent.id}_u', f'A{agent.id}_yT', f'A{agent.id}_xT', f'A{agent.id}_uT']:
                if name in nghbr.MPC_sol or name in nghbr.named_cooperation_dec_vars:
                    nghbr.MPC_sol[name] = agent.MPC_sol[name]

    return 'previous'


def compute_decentralized_following_warm_start_dynamic_cooperative_DMPC(agents:list[Agent], T:int, N:int, terminal_ingredients_type:str = 'equality')->dict:
    """
    Compute a warm start based on following the previously optimal periodic trajectory used in dynamic cooperative DMPC.
    
    Arguments:
    - agents (list): List of agent objects participating in the DMPC.
    - r (dict): Dictionary containing the previously optimal solution.
    - T (int): Period of the periodic trajectory.
    - N (int): Prediction horizon.
    - terminal_ingredients_type (str, optional): Type of terminal ingredients, either 'equality' or 'set'. Default is 'equality'.
    
    Returns:
    - dict: A dictionary containing the warm start trajectories for each agent.
        The function generates a warm start for the dynamic cooperative DMPC by either shifting the previously optimal 
        trajectories or computing new terminal control inputs based on the terminal ingredients type. The warm start 
        trajectories are updated for each agent and returned in a dictionary.
    """

    for agent in agents:
        p = agent.output_dim
        n = agent.state_dim
        q = agent.input_dim
        x_ws = agent.MPC_sol[f'A{agent.id}_x']
        u_ws = agent.MPC_sol[f'A{agent.id}_u']
        yT_ws = agent.MPC_sol[f'A{agent.id}_yT']
        xT_ws = agent.MPC_sol[f'A{agent.id}_xT']
        uT_ws = agent.MPC_sol[f'A{agent.id}_uT']            
        
        if terminal_ingredients_type == 'set':
            tau = N%T
            get_lpv_par = agent.terminal_ingredients['get_lpv_par']
            if get_lpv_par is None:
                thetas = []
            else:
                thetas = get_lpv_par(xT=xT_ws[tau*n : (tau+1)*n, 0], uT=uT_ws[tau*q : (tau+1)*q, 0])  # Get the LPV parameters for the grid points.
                thetas = {key: np.array(thetas[key]).item() for key in thetas}  # Transform the values into scalars.
            X = agent.terminal_ingredients['X']
            Y = agent.terminal_ingredients['Y']
            # Compute the terminal cost matrix and the terminal control matrix.
            P = X['static'].copy()
            K = Y['static'].copy()
            for theta_name in thetas:
                if np.linalg.norm(X[theta_name]) < 1e-10:
                    # Ignore matrices close to zero.
                    pass
                else:
                    P = P + thetas[theta_name]*X[theta_name]
                if np.linalg.norm(Y[theta_name]) < 1e-10:
                    # Ignore matrices close to zero.
                    pass
                else:
                    K = K + thetas[theta_name]*Y[theta_name]
            if thetas:
                # Use CasADi to compute the inverse of P if it is parameter dependent.
                P = cas.inv(P)  # Invert P to get the terminal cost matrix.
            else:
                P = np.linalg.inv(P)
            # Ensure that P is symmetric.
            P = 0.5*(P + P.T)
            K = K@P  # Compute the terminal control matrix.
            
            wrap = N // T
            if isinstance(agent, Satellite) and agent.state_dim == 4:
                xT_ws_corrected = xT_ws[tau*n:(tau+1)*n] + np.vstack([0., 2*np.pi*wrap, 0., 0.])
            else:
                xT_ws_corrected = xT_ws[tau*n:(tau+1)*n]
            kf = uT_ws[tau*q : (tau+1)*q, :] + K@(x_ws[-n:] - xT_ws_corrected)  # Compute the terminal control input.
            xnext = agent.dynamics(x=x_ws[-n:], u=kf)[agent.dynamics.name_out()[0]]  # Compute the next point if the terminal control input is applied.
            # Shift the input and append the terminal control input.
            u_ws = np.vstack([u_ws[q:], kf])
            # Shift the state trajectory and append the next state based on the terminal control input.
            x_ws = np.vstack([x_ws[n:], xnext])
        elif terminal_ingredients_type == 'equality':
            # Use this warm start if the terminal ingredients are of equality type or of an unknown type.
            # Create the new input sequence by shifting the previously optimal one and appending the first part of the cooperation input.
            tau = N%T  # Find the time step after the terminal point on the cooperation trajectry.
            u_ws = np.vstack([u_ws[q:], uT_ws[tau*q:(tau+1)*q]])
            # Create the new state sequence by shifting the previously optimal one and appending the first part of the cooperation state.
            tau = (N+1)%T
            if isinstance(agent, Satellite):
                wrap = N // T
                x_ws = np.vstack([x_ws[n:], xT_ws[tau*n:(tau+1)*n] + np.vstack([0., 2*np.pi*wrap, 0., 0.])])
            else:
                x_ws = np.vstack([x_ws[n:], xT_ws[tau*n:(tau+1)*n]])
        else:
            # In the case of an unconstrained scheme, compute the next state using the dynamics and append the last input again.
            xnext = agent.dynamics(x_ws[-n:], u_ws[-q:])
            u_ws = np.vstack([u_ws[q:], u_ws[-q:]])
            x_ws = np.vstack([x_ws[n:], xnext])
        
        # Shift the cooperation trajectories.
        uT_ws = np.vstack([uT_ws[q:], uT_ws[0:q]])
        xT_ws = np.vstack([xT_ws[n:], xT_ws[0:n]])
        yT_ws = np.vstack([yT_ws[p:], yT_ws[0:p]])
        
        # For satellite agents, the second state (theta; angular position) wraps around 2pi and needs to be adjusted.
        # Here, shifting is not enough, since the agent's state increments theta and does not consider the modulo behaviour.
        # Hence, theta needs to be increased by 2pi when shifted.
        if isinstance(agent, Satellite) and agent.state_dim == 4:
            if p != 1:
                raise ValueError("Satellite agent must have output dimension 1. The output is only the angular position!")
            xT_ws[-3] = xT_ws[-3] + 2*np.pi
            yT_ws[-1] = yT_ws[-1] + 2*np.pi
        
        # Reassign the new trajectories.
        agent.MPC_sol[f'A{agent.id}_x'] = x_ws
        agent.MPC_sol[f'A{agent.id}_u'] = u_ws
        agent.MPC_sol[f'A{agent.id}_yT'] = yT_ws
        agent.MPC_sol[f'A{agent.id}_xT'] = xT_ws
        agent.MPC_sol[f'A{agent.id}_uT'] = uT_ws
        
    # Also update copies neighbours hold. Note that this would require communication.
    # Each agent sends an update of their trajectory to neighbours in order to synchronise the warm start.
    # If one considers a decentralised optimisation algorithm based on ADMM which features an averaging step
    # requiring at least two communication phases per iteration, then this additional communication should be negligible.
    # If one would like to avoid this, then the previously optimal solution could be used to warm start, i.e. no change
    # in 'agent.MPC_sol' should be made.
    
    for agent in agents:
        for nghbr in agent.neighbours:
            for name in [f'A{agent.id}_x', f'A{agent.id}_u', f'A{agent.id}_yT', f'A{agent.id}_xT', f'A{agent.id}_uT']:
                if name in nghbr.MPC_sol or name in nghbr.named_cooperation_dec_vars:
                    nghbr.MPC_sol[name] = agent.MPC_sol[name]

    return 'previous'


def solve_MPC_for_dynamic_cooperation_decentrally(sqp_max_iter, admm_max_iter, admm_penalty, t, agents, N=1, T=None, solver='qpoases', terminal_ingredients_type = 'equality', feas_tol = 1e-8, warm_start=None, coop_task_builder=None, coop_kwargs=None, print_level=5, max_iter=None, verbose=1, parallel=False):
    """Set up the optimization problem used in MPC for dynamic cooperation, and solve it decentrally. The solution is stored in the dictionary attribute agent.MPC_sol for each agent.
        
    The explicit relation of the cooperative output trajectory and the cooperative state and input trajectory is not used. Instead, it is defined implicitely by adding a suitable constraint to the optimisation problem.
    
    Terminal ingredients of the type 'set' assume the structure developed in
    [1] 2020 - J. Koehler et al. - A Nonlinear Model Predictive Control Framework Using Reference Generic Terminal Ingredients - IEEE TAC. doi: 10.1109/TAC.2019.2949350
    
    The implementation is based on
    [2] 2025 - G. Stomberg et al - Decentralized real-time iterations for distributed nonlinear model predictive control - arXiv:2401.14898. doi: 10.48550/arXiv.2401.14898
    
    Arguments:
        - sqp_max_iter (int): Maximum number of outer SQP iterations.
        - admm_max_iter (int): Maximum number of inner ADMM iterations.
        - admm_penalty (float): Penalty parameter of the inner ADMM iterations.
        - t (int): Current time step. Used for time-varying costs and constraints and printing.
        - agents (list): Contains all agents ( objects) in the multi-agent system for which the optimization problem is build. They must have the following attributes not provided by the class:
            - named_cooperation_dec_vars (dict): Containing the decision variables used in the cooperation objective function and the constraints of admissible cooperation outputs, including those of neighbours. The keys must be the unique name of the decision variable, which should match the name of the symbolic. The naming scheme is 'A{agent.id}_name', e.g. 'A3_yT'. The respective value is the symbolic. It must contain the cooperation output of the agent, i.e. 'A{agent.id}_yT'.
            -  cooperation_constraints (dict): Containing 
                - function (cas.Function): The function defining the constraint on the cooperation output and additional decision variables. Must also contain affine constraints if applicable.
            - yT_pre: The previously optimal cooperation output, already shifted. Can be set to None if there is none available.
            - penalty_weight: A weight for the penalty on the change in the cooperation output. For now, this is implemented as a quadratic penalty function.
            - 'nonlinear_constraints' (list[casadi.Function]): Defining nonlinear constraints local to the agent. 
                Currently, only pointwise-in-time constraints on the state are allowed.
                The input must be named 'x' and the output 'g'.
                The constraint should be non-positive if and only if the state is feasible.
        - N (int): Prediction horizon of the MPC optimisation problem.
        - T (int): Period length of the cooperative output trajectory. Periodicity of the cooperation output and the corresponding state and input trajectories is automatically constrained in this method and does not need to be supplied. If None is passed, and a coop_task_builder function is provided, the period length of this function is used. (default is None)
        - solver (string): Solver used for solving the QPs in the ADMM iteration. Options are all available solvers for QPs that CasADi can handle, gurobi and ipopt. If ipopt is chosen, a nlp formulation is used. If gurobi is chosen, a direct interface is used and parallelisation is possible. (default is 'qpoases')
        - terminal_ingredients_type (str): Specifying the type of terminal ingredients, possible are 'equality' (default), 'without', 'set'. 
            - 'equality': Equality terminal constraints are enforced.
            - 'without': Agents must have the attribute 'tracking_bound', whose value will be enforced as an upper bound on the tracking part of the cost, as required by the scheme. No additional terminal ingredients or cost are added.
            - 'set': A terminal set constraint with non-empty interior and a terminal cost are used. Since both depend on the reference, the agents must have an attribute 'terminal_ingredients', which is a dictionary containing
                'get_lpv_par': A function that takes a point on the cooperative trajectory and returns the parameters used in the quasi-LPV description, cf. [(11), (12); 1].
                'X': A list of matrices that are multiplied with the parameters of the quasi-LPV description to obtain the terminal cost matrix, cf. [(12), Prop. 1; 1].
                'size': A scalar determining the terminal set size, cf. [Sec. III.C; 1].
        - check_feasibility_flag (bool): If true, feasibility of the computed solution is checked with tolerance feas_tol. The computed residuals are written to r with keys 'lower_residual' and 'upper_residual'.
        - feas_tol (float): Constraint violation tolerance for ipopt, does not affect any other solver. (default is 1e-8)
        - warm_start: Warm start for the solver. Either a dictionary containing the named variables of the objective function as keys and the respective warm start as values, or a string 'previous'; then the values in agent.MPC_sol are taken. If None is passed, all values are initialised with zeros. (default is None)
        - coop_task_builder (function): Provides a function that sets up all necessary ingredients of the cooperative task. For example, in the cast that the cooperative task needs to be specified at runtime, e.g. because it is time-varying or parameter dependend. (default is None)
        - coop_kwargs (dict): Contains the kewyword arguments for 'coop_task_builder' to be unpacked and passed along, e.g. coop_task_builder(**coop_kwargs). (default is None)
        - print_level (int): The verbosity level of ipopt. Does not affect any other solver (set verbose for this). (default is 5) 
        - max_iter (int): The maximum number of iterations ipopt is allowed to perform. Does not affect any other solver. If None, the default of ipopt is used. (default is None)
        - verbose (int): At 0, print minimial information. At 1, also print SQP iterations. At 2, also print ADMM iterations. At 3, also print information about Hessian corrections. At 4, also print solver iterations (default is 1)
        - parallel (bool): If true, the local ADMM step is done in parallel (only supported for gurobi as the solver). (default is False)
    
    Returns:
        - res (dict): Containing information about the cost. The solution for each agent is stored in the attribute agent.MPC_sol.
    """   
    # Update the cooperative task.
    if coop_task_builder is not None:
        if coop_kwargs is None:
            raise AssertionError(f'{t}: Cooperative task builder function supplied, but arguments to be passed not specified in "coop_kwargs"!')
        else:
            coop_task_builder(**coop_kwargs)  # Set the ingredients for the cooperative task.

    # -----------------------------------------------------------------------------------------
    # Set up the parts entirely local to an agent.
    # All local decision variables are saved in 'agent.named_dec_vars'.
    # -----------------------------------------------------------------------------------------
    for agent in agents:
        # Shorthands.
        n = agent.state_dim
        q = agent.input_dim
        p = agent.output_dim
        yT_pre = agent.yT_pre
        
        # Introduce a local objective functions.
        agent.local_objective = cas.MX(0)  
        agent.local_tracking_objective = cas.MX(0)
        agent.local_cooperation_objective = cas.MX(0)
        agent.local_change_penalty_objective = cas.MX(0)
        
        agent.local_equality_constraints = []
        agent.local_inequality_constraints = []
        # agent.local_inequality_constraints_lb = []
        # agent.local_inequality_constraints_ub = []
        agent.local_equality_constraints_names = []
        agent.local_inequality_constraints_names = []
            
        # Create decision variables.
        # Time steps are stacked beneath each other.
        u = cas.MX.sym(f'A{agent.id}_u', q*N, 1)  # input sequence
        x = cas.MX.sym(f'A{agent.id}_x', n*N, 1)  # predicted state sequence; the initial state is not predicted
        uT = cas.MX.sym(f'A{agent.id}_uT', q*T, 1)  # input sequence of cooperation reference
        xT = cas.MX.sym(f'A{agent.id}_xT', n*T, 1)  # state sequence of cooperation reference
        # Create a dictionary of used decision variables for this agent.
        agent.named_dec_vars = {}
        # Add these decision variables to the agent's dictionary.
        agent.named_dec_vars[u.name()] = u
        agent.named_dec_vars[x.name()] = x
        
        # Extract the decision variables of the cooperative part.
        if f'A{agent.id}_yT' not in agent.named_cooperation_dec_vars:
            raise KeyError(f"Key 'A{agent.id}_yt' not found in the dictionary containing the decision variables of the cooperative part.")
        yT = agent.named_cooperation_dec_vars[f'A{agent.id}_yT']
        if f'A{agent.id}_uT' in agent.named_cooperation_dec_vars:
            uT = agent.named_cooperation_dec_vars[f'A{agent.id}_uT']
        if f'A{agent.id}_xT' in agent.named_cooperation_dec_vars:
            xT = agent.named_cooperation_dec_vars[f'A{agent.id}_xT']
            
        agent.named_dec_vars[uT.name()] = uT
        agent.named_dec_vars[xT.name()] = xT
        agent.named_dec_vars[yT.name()] = yT
        
        # Set the standard constraints, i.e. adherence to dynamics, state and input constraints (also nonlinear) and fixing the initial state.
        (ineq_constraints, _, _, eq_constraints, _, _, ineq_constraints_names, eq_constraints_names) = get_standard_MPC_constraints(agent, u, x, N, t)
        # Add these constraints.
        agent.local_inequality_constraints.extend(ineq_constraints)
        agent.local_inequality_constraints_names.extend(ineq_constraints_names)
        # Convert equality constraints to inequality constraints.
        agent.local_equality_constraints.extend(eq_constraints)
        agent.local_equality_constraints_names.extend(eq_constraints_names)

        # Add constraints that link the cooperation state and input sequence to the cooperation output sequence.
        for k in range(T):
            # This constraint is an equality constraint.
            agent.local_equality_constraints.append(agent.output_map(xT[k*n:(k+1)*n, 0], uT[k*q:(k+1)*q, 0]) - yT[k*p:(k+1)*p, 0])
            agent.local_equality_constraints_names.extend([f'A{agent.id}_yT_{k} output map']*agent.output_dim)
            
        # Add a constraint that enforces the cooperation state and input sequence to be a trajectory.
        for k in range(T-1):
            agent.local_equality_constraints.append(agent.dynamics(xT[k*n : (k+1)*n, 0], uT[k*q : (k+1)*q, 0]) - xT[(k+1)*n : (k+2)*n, 0])
            agent.local_equality_constraints_names.extend([f'A{agent.id}_xT_{k} dynamics']*n)
        
        # Add a constraint that enforces the cooperation state and input trajectory to be periodic.
        if isinstance(agent, Satellite) and agent.state_dim == 4:
            # For a satellite agent in polar coordinates, the angular position wraps around, which needs to be considered in the constraints.            
            # Add a constraint that enforces the cooperation state and input trajectory to be periodic.
            agent.local_equality_constraints.append(agent.dynamics(xT[(T-1)*n : T*n, 0], uT[(T-1)*q : T*q, 0]) - xT[0 : n, 0] - np.vstack([0.0, 2*np.pi, 0.0, 0.0]))
            agent.local_equality_constraints_names.extend([f'A{agent.id}_xT periodic']*n)
        elif isinstance(agent, Vessel) and T > 1:
            # For a vessel agent, the heading is not constrained on the circle, which needs to be considered in the periodicity constraint,
            # if the periodicity is larger than 1.
            xTc = agent.dynamics(xT[(T-1)*n : T*n, 0], uT[(T-1)*q : T*q, 0])
            agent.local_equality_constraints.append(xTc[0 : 2, 0] - xT[0 : 2, 0])
            # Transform the heading to a point on the circle.
            zT1 = cas.cos(xTc[2, 0])
            zT2 = cas.sin(xTc[2, 0])
            z01 = cas.cos(xT[2, 0])
            z02 = cas.sin(xT[2, 0]) 
            agent.local_equality_constraints.append(cas.vertcat(zT1, zT2) - cas.vertcat(z01, z02))
            agent.local_equality_constraints_names.extend([f'A{agent.id}_xT periodic']*2)
            # Constrain the rest of the state and input trajectory.
            agent.local_equality_constraints.append(xTc[3 : n, 0] - xT[3 : n, 0])
            agent.local_equality_constraints_names.extend([f'A{agent.id}_xT periodic']*n)
        else:            
            agent.local_equality_constraints.append(agent.dynamics(xT[(T-1)*n : T*n, 0], uT[(T-1)*q : T*q, 0]) - xT[0 : n, 0])
            agent.local_equality_constraints_names.extend([f'A{agent.id}_xT periodic']*n)
        
        # Set the cooperation constraints (only those acting on the agent's own decision variables, not the copies of neighbours' decision variables.)
        # Extract the variables that are needed for the function call.
        dv_cooperation_constraint = {key: agent.named_cooperation_dec_vars[key] for key in agent.cooperation_constraints['function'].name_in()}
        agent.named_dec_vars.update(dv_cooperation_constraint)  # Track these variables.
        evaluated_constraint = agent.cooperation_constraints['function'].call(dv_cooperation_constraint)[agent.cooperation_constraints['function'].name_out()[0]]
        agent.local_inequality_constraints.append(evaluated_constraint)
        # agent.local_inequality_constraints_lb.extend(agent.cooperation_constraints['lower_bound'])
        # agent.local_inequality_constraints_ub.extend(agent.cooperation_constraints['upper_bound'])
        agent.local_inequality_constraints_names.extend([f'A{agent.id} {agent.cooperation_constraints['function'].name()}']*evaluated_constraint.shape[0])
        
        # Create the objective function:
        # Sum up the stage cost.
        stage_cost = agent.stage_cost
        # Note that the decision variables begin with x(1) (which is x[0:n]) and u(0) (which is u[0:q]).
        # First, add the stage cost for x(0) and u(0).
        tracking_objective_agent = cas.MX(0)
        tracking_objective_agent += stage_cost(x=agent.current_state, u=u[0 : q, 0], xT=xT[0 : n, 0], uT=uT[0 : q, 0])['l']
        # Second, add the stage cost for the remaining horizon.
        if isinstance(agent, Satellite) and agent.state_dim == 4:
            for k in range(1, N):
                tau = k%T  # Calculate the corresponding step in the T-periodic trajectory.
                wrap = k // T  # Calculate the number of wraps around the T-periodic trajectory.
                tracking_objective_agent += stage_cost(x[(k-1)*n : k*n, 0], u[k*q : (k+1)*q, 0], xT[tau*n : (tau+1)*n, 0] + np.vstack([0., 2*np.pi*wrap, 0., 0.]), uT[tau*q : (tau+1)*q, 0])
        else:
            for k in range(1, N):
                tau = k%T  # Calculate the corresponding step in the T-periodic trajectory.
                wrap = k // T  # Calculate the number of wraps around the T-periodic trajectory.
                tracking_objective_agent += stage_cost(x[(k-1)*n : k*n, 0], u[k*q : (k+1)*q, 0], xT[tau*n : (tau+1)*n, 0], uT[tau*q : (tau+1)*q, 0])
        
        # Add a terminal constraint and cost in the case if equality or set constraints, or add a bound on the tracking cost.
        if terminal_ingredients_type == 'equality':
            if isinstance(agent, Satellite) and agent.state_dim == 4:
                # As before, theta does not wrap around.
                wrap = N // T  # Calculate the number of wraps around the T-periodic trajectory.
                tau = N%T  # Calculate the step in the T-periodic trajectory at the end of the prediction horizon.
                agent.local_equality_constraints.append(x[(N-1)*n : N*n, 0] - xT[tau*n : (tau+1)*n, 0] - np.vstack([0., 2*np.pi*wrap, 0., 0.]))
            else:
                # Set a terminal equality constraint.
                tau = N%T  # Calculate the step in the T-perioid trajectory at the end of the prediction horizon.
                agent.local_equality_constraints.append(x[(N-1)*n : N*n, 0] - xT[tau*n : (tau+1)*n, 0])
            agent.local_equality_constraints_names.extend([f'A{agent.id}_xT TEC']*n)
        elif terminal_ingredients_type == 'set':
            tau = N%T
            get_lpv_par = agent.terminal_ingredients['get_lpv_par']
            if get_lpv_par is None:
                thetas = []
            else:
                thetas = get_lpv_par(xT=xT[tau*n : (tau+1)*n, 0], uT=uT[tau*q : (tau+1)*q, 0])  # Get the LPV parameters for the grid points.
                thetas = {key: np.array(thetas[key]).item() for key in thetas}  # Transform the values into scalars.
            X = agent.terminal_ingredients['X']
            #Y = agent.terminal_ingredients['Y']
            # Compute the terminal cost matrix and the terminal control matrix.
            P = X['static'].copy()
            #K = Y['static'].copy()
            for theta_name in thetas:
                if np.linalg.norm(X[theta_name]) < 1e-10:
                    # Ignore matrices close to zero.
                    continue
                P = P + thetas[theta_name]*X[theta_name]
                #K = K + thetas[theta_name]*Y[theta_name]
            if thetas:
                # Use CasADi to compute the inverse of P if it is parameter dependent.
                P = cas.inv(P)  # Invert P to get the terminal cost matrix.
            else:
                P = np.linalg.inv(P)
            # Ensure that P is symmetric.
            P = 0.5*(P + P.T)
            
            if isinstance(agent, Satellite) and agent.state_dim == 4:
                # The angular position does not wrap in the prediction, hence if the horizon is longer than the period, the ensuing offset must be considered.
                wrap = N // T  # Calculate the number of wraps around the T-periodic trajectory.
                
                # Compute the terminal cost.
                diff = x[(N-1)*n : N*n, 0] - xT[tau*n : (tau+1)*n, 0] - np.vstack([0., 2*np.pi*wrap, 0., 0.])
                terminal_cost = (diff).T @ P @ (diff)
            else:
                # Compute the terminal cost.
                terminal_cost = (x[(N-1)*n : N*n, 0] - xT[tau*n : (tau+1)*n, 0]).T @ P @ (x[(N-1)*n : N*n, 0] - xT[tau*n : (tau+1)*n, 0])
                
            tracking_objective_agent += terminal_cost  # Add the terminal cost to the tracking objective.
            # Add the terminal constraint.
            agent.local_inequality_constraints.append(terminal_cost - agent.terminal_ingredients['size'])
            # agent.local_inequality_constraints_lb.append(np.array([[-np.inf]]))
            # agent.local_inequality_constraints_ub.append(np.array([[]]))
            agent.local_inequality_constraints_names.extend([f'A{agent.id}_xT terminal constraint'])
        else:
            # Add a bound on the tracking cost.
            agent.local_inequality_constraints.append(tracking_objective_agent - agent.tracking_bound)
            # agent.local_inequality_constraints_lb.append([[-np.inf]])
            # agent.local_inequality_constraints_ub.append([[0.0]])
            agent.local_inequality_constraints_names.extend([f'A{agent.id}_xT tracking bound'])
        
        agent.local_tracking_objective += tracking_objective_agent
        agent.local_objective += tracking_objective_agent
        
        if yT_pre is not None:
            # Add the penalty on the change in the cooperation output.
            penalty_function = cas.MX(0)
            for k in range(T):
                penalty_function += cas.dot(yT[k*p:(k+1)*p, 0] - yT_pre[k*p:(k+1)*p, 0], yT[k*p:(k+1)*p, 0] - yT_pre[k*p:(k+1)*p, 0])
            penalty_function = agent.penalty_weight*penalty_function
            agent.local_change_penalty_objective += penalty_function
            agent.local_objective += penalty_function
    
    # -----------------------------------------------------------------------------------------
    # Create copies of the agents neighbours' decision variables and add them to the agent's 
    # dictionary.
    # -----------------------------------------------------------------------------------------
    for agent in agents:
        # Create copies of variables coupled by the cooperation objective function.
        for decision_variable_name in agent.named_cooperation_dec_vars:
            if decision_variable_name not in agent.named_dec_vars:
                # Create a copy of the neighbour's decision variable and add it to the agent's local decision variables.
                agent.named_dec_vars[decision_variable_name] = cas.MX.sym(decision_variable_name, *agent.named_cooperation_dec_vars[decision_variable_name].shape)
        # Create copies of the variables coupled by the coupling constraints.
        # Note that coupling constraints on the cooperation outputs, states or inputs are defined in the agent's cooperation constraints.
        if agent.coupling_constraints is not None:
            for cstr_fun in agent.coupling_constraints:
                for nghbr in agent.neighbours:
                     agent.named_dec_vars.update({dv_name: cas.MX.sym(dv_name, *nghbr.named_dec_vars[dv_name].shape) for dv_name in nghbr.named_dec_vars if dv_name in cstr_fun.name_in() and dv_name not in agent.named_dec_vars})

    # -----------------------------------------------------------------------------------------
    # Add the coupled parts 
    # The coupled parts are the cooperation objective function and the coupling constraints.
    # -----------------------------------------------------------------------------------------
    for agent in agents:
        # Add the cooperation objective function.
        cooperation_objective_agent = cas.MX(0)
        # Extract the decision variables that are needed for the function call, where we use the local decision variables of the agent that includes the copies.
        cooperation_decision_variables = {dv_name: agent.named_dec_vars[dv_name] for dv_name in agent.cooperation_objective_function.name_in() if dv_name in agent.named_dec_vars}
        cooperation_objective_agent += agent.cooperation_objective_function.call(cooperation_decision_variables)[agent.cooperation_objective_function.name_out()[0]]
        agent.local_cooperation_objective += cooperation_objective_agent
        agent.local_objective += cooperation_objective_agent 
        
        # Add the coupling constraints.
        # Note that coupling constraints on the cooperation outputs, states or inputs are defined in the agent's cooperation constraints.
        if agent.coupling_constraints is not None:
            for cstr_func in agent.coupling_constraints:
                # Extract the variables that are needed for the function call.
                cstr_func_dec_vars = {name: agent.named_dec_vars[name] for name in cstr_func.name_in() if name in agent.named_dec_vars}
                # Add the constraint.
                for k in range(N):
                    cstr_func_dec_vars_at_k = {name: cstr_func_dec_vars[name][k*n:(k+1)*n] for name in cstr_func_dec_vars}
                    coupling_cstr_eval_at_k = cstr_func.call(cstr_func_dec_vars_at_k)
                    for out_name in coupling_cstr_eval_at_k:
                        agent.local_inequality_constraints.append(coupling_cstr_eval_at_k[out_name])
                        # agent.local_inequality_constraints_lb.append(-np.inf*np.ones(coupling_cstr_eval_at_k[out_name].shape))
                        # agent.local_inequality_constraints_ub.append(np.zeros(coupling_cstr_eval_at_k[out_name].shape))
                        agent.local_inequality_constraints_names.extend([f'A{agent.id} {cstr_func.name()} on x']*coupling_cstr_eval_at_k[out_name].shape[0]) 
                        
    # -----------------------------------------------------------------------------------------
    # Create the consensus constraint coupling the local copies.
    # -----------------------------------------------------------------------------------------
    # Generate a list of all decision variables.
    merged_named_dec_vars = {k: v for a in agents for k, v in a.named_dec_vars.items()}
    # Count how many times each variable appears across all agents, i.e. how many copies exist.
    var_counts = {name: 0 for name in merged_named_dec_vars}
    for agent in agents:
        for name in agent.named_dec_vars:
            if name in var_counts:
                var_counts[name] += 1

    # The local variables are contained in 'agent.named_dec_vars'. 
    # The consensus variables are contained in 'agent.named_copied_dec_vars'.
    for agent in agents:
        agent.stacked_dec_vars = cas.vertcat(*[agent.named_dec_vars[name] for name in agent.named_dec_vars])
        
        # Create local copies.
        agent.named_copied_dec_vars = {name: cas.MX.sym(name, *agent.named_dec_vars[name].shape) for name in agent.named_dec_vars}
        # Create a multiplier for the copies.
        agent.consensus_multiplier = cas.MX.sym(f'A{agent.id}_consensus_mult', *cas.vertcat(*[agent.named_copied_dec_vars[name] for name in agent.named_copied_dec_vars]).shape)
    
    # -----------------------------------------------------------------------------------------
    # Initialize the SQP and start with the outer iteration. 
    # -----------------------------------------------------------------------------------------
        
    # Initialise the outer iterate from the warm start if applicable.
    if warm_start is not None and type(warm_start) != str:
        # Check if all variables are warm started.
        missing_vars = [key for key in merged_named_dec_vars if key not in warm_start]
        if len(missing_vars):
            # Raise an error since a variable has not been assigned in the warm start dictionary.
            raise ValueError(f'The following variables are missing in the warm start: {missing_vars}!')
        else:
            # Assign the warm start.
            for agent in agents:
                # Initialise the primal variables.
                if warm_start is not None:
                    agent.outer_iterate_values = {name: warm_start[name] for name in agent.named_dec_vars}
                else:
                    agent.outer_iterate_values = {name: np.zeros(*agent.named_dec_vars[name].shape) for name in agent.named_dec_vars}
                # Initialise the multipliers.
                if f'A{agent.id}_eq_mult' in warm_start:
                    agent.eq_multiplier = warm_start[f'A{agent.id}_eq_mult']
                else:
                    agent.eq_multiplier = np.zeros(cas.vertcat(*agent.local_equality_constraints).shape)
                if f'A{agent.id}_ineq_mult' in warm_start:
                    agent.ineq_multiplier = warm_start[f'A{agent.id}_ineq_mult']
                else:
                    agent.ineq_multiplier = np.zeros(cas.vertcat(*agent.local_inequality_constraints).shape)
                if f'A{agent.id}_consensus_mult' in warm_start:
                    agent.consensus_multiplier_at_l = warm_start[f'A{agent.id}_consensus_mult']
                else:
                    agent.consensus_multiplier_at_l = np.zeros(agent.consensus_multiplier.shape)
    elif type(warm_start) == str and warm_start == 'previous':
        # Assign the previous solution as a warm start.
        for agent in agents:
            if agent.MPC_sol is None:
                raise ValueError(f'{t}: No previous solution available for agent {agent.id}!')
            else:
                # Initialise the primal variables.
                agent.outer_iterate_values = {name: agent.MPC_sol[name] for name in agent.named_dec_vars}
                # Initialise the multipliers.
                if f'A{agent.id}_eq_mult' in agent.MPC_sol:
                    agent.eq_multiplier = agent.MPC_sol[f'A{agent.id}_eq_mult']
                else:
                    agent.eq_multiplier = np.zeros(cas.vertcat(*agent.local_equality_constraints).shape)
                if f'A{agent.id}_ineq_mult' in agent.MPC_sol:
                    agent.ineq_multiplier = agent.MPC_sol[f'A{agent.id}_ineq_mult']
                else:
                    agent.ineq_multiplier = np.zeros(cas.vertcat(*agent.local_inequality_constraints).shape)
                if f'A{agent.id}_consensus_mult' in agent.MPC_sol:
                    agent.consensus_multiplier_at_l = agent.MPC_sol[f'A{agent.id}_consensus_mult']
                else:
                    agent.consensus_multiplier_at_l = np.zeros(agent.consensus_multiplier.shape)
    else:
        for agent in agents:
            agent.eq_multiplier = np.zeros(cas.vertcat(*agent.local_equality_constraints).shape)
            agent.ineq_multiplier = np.zeros(cas.vertcat(*agent.local_inequality_constraints).shape)
            agent.consensus_multiplier_at_l = np.zeros(agent.consensus_multiplier.shape)
            # Initialise the primal variables.
            agent.outer_iterate_values = {name: np.zeros(agent.named_dec_vars[name].shape) for name in agent.named_dec_vars}
            
    # Initialise each agent.
    if parallel:
        with ThreadPoolExecutor(max_workers=min(len(agents), os.cpu_count())) as executor:
            agents = list(executor.map(lambda ag: _initialize_agent(ag, solver, verbose), agents))
    else:
        for agent in agents:
            _initialize_agent(agent, solver, verbose)
        
    # Initialise the initial standard deviation.
    max_deviation = 0.0
    worst_deviated_var = None
    
    for sqp_iter in range(sqp_max_iter):
        if verbose > 0:
            print(f'\n{t}: ====== Starting SQP iteration {sqp_iter+1} of {sqp_max_iter} ===============================\n')
        
        if parallel:
            with ThreadPoolExecutor(max_workers=min(len(agents), os.cpu_count())) as executor:
                agents = list(executor.map(
                    lambda ag: _prepare_agent_qp(ag, admm_penalty, sqp_iter, solver, print_level, max_iter, feas_tol, verbose),
                    agents
                ))
        else:
            for agent in agents:
                _prepare_agent_qp(agent, admm_penalty, sqp_iter, solver, print_level, max_iter, feas_tol, verbose)

        # -----------------------------------------------------------------------------------------
        # Start with the inner ADMM iteration.
        # recent_max_deviation_values = []
        for admm_iter in range(admm_max_iter):
            # ----------- Minimise the augmented Lagrangian with respect to the local decision variables.
            if solver == 'gurobi':
                    if parallel:
                        with ThreadPoolExecutor(max_workers=min(len(agents), os.cpu_count())) as executor:
                            agents = list(executor.map(lambda ag: _solve_agent_qp(ag, admm_penalty), agents))
                    else:
                        for agent in agents:
                            _solve_agent_qp(agent, admm_penalty)
            else:
                for agent in agents:
                    lower_bound = np.vstack([- agent.eq_cstr_at_k + agent.jac_eq_cstr_at_k @ agent.stacked_outer_iterate_values, -np.inf*np.ones(agent.inner_inequality_constraints.shape)])
                    upper_bound = np.vstack([- agent.eq_cstr_at_k + agent.jac_eq_cstr_at_k @ agent.stacked_outer_iterate_values, - agent.ineq_cstr_at_k + agent.jac_ineq_cstr_at_k @ agent.stacked_outer_iterate_values])
                    
                    inner_sol = agent.S_inner_program(  # Solve the optimisation problem.
                        # x0 = agent.stacked_copied_dec_vals_at_l, 
                        x0 = agent.stacked_dec_vals_at_l, 
                        p = np.concatenate([agent.stacked_copied_dec_vals_at_l, agent.consensus_multiplier_at_l]),
                        lbg=lower_bound, ubg=upper_bound, 
                        lam_g0=cas.vertcat(agent.eq_multiplier, agent.ineq_multiplier), 
                        lam_x0=agent.lam_x0)
                    
                    # Extract the solution.
                    agent.stacked_dec_vals_at_l = np.array(inner_sol['x']).flatten()
                    
                    agent.eq_multiplier = np.array(inner_sol['lam_g']).flatten()[0:agent.inner_equality_constraints.shape[0]]
                    agent.ineq_multiplier = np.array(inner_sol['lam_g']).flatten()[agent.inner_equality_constraints.shape[0]:]
                    agent.lam_x0 = np.array(inner_sol['lam_x']).flatten()
                    
                    ## Extract and assign the solution to the agent.
                    agent.dec_vals_at_l = {}
                    agent.copied_dec_vals_at_l = {}
                    agent.named_consensus_multiplier_at_l = {}
                    start = 0
                    for var_name, var_sym in agent.named_dec_vars.items():
                        rows, cols = var_sym.shape
                        step = rows*cols
                        agent.dec_vals_at_l[var_name] = agent.stacked_dec_vals_at_l[start:start+step].reshape(rows, cols)
                        agent.copied_dec_vals_at_l[var_name] = np.vstack([agent.stacked_copied_dec_vals_at_l[start:start+step]]).reshape(rows, cols)
                        agent.named_consensus_multiplier_at_l[var_name] = np.vstack([agent.consensus_multiplier_at_l[start:start+step]]).reshape(rows, cols)
                        start += step
                    
            # ----------- Perform the consensus update, i.e. averaging.
            if admm_iter == 0:
                for name in merged_named_dec_vars:
                    shapes = [agent.dec_vals_at_l[name].shape for agent in agents if name in agent.dec_vals_at_l]
                    if len(set(shapes)) > 1:
                        raise ValueError(f"Inconsistent shapes for variable '{name}': {shapes}")
                        
            # Initialise the average.
            average_consensus_values = {name: np.zeros(value.shape) for name, value in merged_named_dec_vars.items()}

            # Collect all local copies of each variable.
            all_copies = {name: [] for name in merged_named_dec_vars}
            for agent in agents:
                for name in average_consensus_values:
                    if name in agent.named_dec_vars:
                        val = np.array(agent.dec_vals_at_l[name]).copy()
                        average_consensus_values[name] += val
                        all_copies[name].append(val)

            # Compute the average.
            for name in average_consensus_values:
                average_consensus_values[name] /= var_counts[name]

            # Assign the new consensus values.
            for name in average_consensus_values:
                for agent in agents:
                    if name in agent.named_dec_vars:
                        agent.copied_dec_vals_at_l[name] = np.copy(average_consensus_values[name])

            # Update the stacked vector of consensus values.
            for agent in agents:
                agent.stacked_copied_dec_vals_at_l = np.vstack(
                    [agent.copied_dec_vals_at_l[name] for name in agent.named_dec_vars if name in agent.copied_dec_vals_at_l]
                )

            # Compute absolute deviation
            deviation_consensus_values = {}
            for name, copies in all_copies.items():
                if len(copies) > 1:
                    stacked = np.stack([np.array(c).flatten() for c in copies], axis=0)
                    mean_val = np.mean(stacked, axis=0)
                    diffs = stacked - mean_val  # deviations from average
                    norms = np.linalg.norm(diffs, axis=1)  # norm per copy
                    deviation_consensus_values[name] = float(np.max(norms)) 
                elif len(copies) == 1:
                    deviation_consensus_values[name] = 0.0

            # Get max deviation and worst variable.
            max_deviation = max(deviation_consensus_values.values())
            worst_deviated_var = max(deviation_consensus_values, key=deviation_consensus_values.get)
            
            # ----------- Update the consensus multiplier.
            for agent in agents:
                # Update the consensus multiplier.
                agent.consensus_multiplier_at_l = agent.consensus_multiplier_at_l + admm_penalty*(agent.stacked_dec_vals_at_l - agent.stacked_copied_dec_vals_at_l)
                
            if verbose > 1:
                print(f'\n{t}: ------ Completed ADMM iteration {admm_iter+1} of {admm_max_iter} ---- max dev: {max_deviation:.3g} (in {worst_deviated_var}) ---- SQP {sqp_iter+1} of {sqp_max_iter}\n')
            elif verbose == 1 and admm_iter == admm_max_iter-1:
                print(f'\n{t}: ------ Completed ADMM iteration {admm_iter+1} of {admm_max_iter} ---- max dev: {max_deviation:.3g} (in {worst_deviated_var}) ---- SQP {sqp_iter+1} of {sqp_max_iter}\n')
            
        for agent in agents:
            for var_name in agent.outer_iterate_values:
                agent.outer_iterate_values[var_name] = np.copy(agent.copied_dec_vals_at_l[var_name])
                
    # Save the MPC solution.
    for agent in agents:
        if agent.MPC_sol is None:
            agent.MPC_sol = {}
        agent.MPC_sol[f'A{agent.id}_eq_mult'] = agent.eq_multiplier.copy()
        agent.MPC_sol[f'A{agent.id}_ineq_mult'] = agent.ineq_multiplier.copy()
        agent.MPC_sol[f'A{agent.id}_consensus_mult'] = agent.consensus_multiplier_at_l.copy()
        for name in agent.named_dec_vars:
            #agent.MPC_sol[name] = agent.outer_iterate_values[name]
            agent.MPC_sol[name] = agent.dec_vals_at_l[name].copy()
            
    # Compute the costs.
    res = {}
    res['tracking_cost'] = 0.0
    res['cooperative_cost'] = 0.0
    res['change_cost'] = 0.0
    for agent in agents:
        tracking_cost_func = cas.Function('tracking_cost', [var for _, var in agent.named_dec_vars.items()], [agent.local_tracking_objective], [name for name in agent.named_dec_vars], ['Jtr'])
        cooperative_cost_func = cas.Function('cooperative_cost', [var for _, var in agent.named_dec_vars.items()], [agent.local_cooperation_objective], [name for name in agent.named_dec_vars], ['Wc'])
        change_cost_func = cas.Function('change_cost', [var for _, var in agent.named_dec_vars.items()], [agent.local_change_penalty_objective], [name for name in agent.named_dec_vars], ['VD'])
        # res['tracking_cost'] += tracking_cost_func.call(agent.outer_iterate_values)['Jtr']
        # res['cooperative_cost'] += cooperative_cost_func.call(agent.outer_iterate_values)['Wc']
        # res['change_cost'] += change_cost_func.call(agent.outer_iterate_values)['VD']
        res['tracking_cost'] += tracking_cost_func.call(agent.dec_vals_at_l)['Jtr']
        res['cooperative_cost'] += cooperative_cost_func.call(agent.dec_vals_at_l)['Wc']
        res['change_cost'] += change_cost_func.call(agent.dec_vals_at_l)['VD']
    res['J'] = res['tracking_cost'] + res['cooperative_cost'] + res['change_cost']
    res['deviation_consensus_values'] = deviation_consensus_values
    
    return res

def _initialize_agent(agent, solver, verbose):
    agent.stacked_dec_vals_at_l = cas.vertcat(*[agent.outer_iterate_values[name] for name in agent.named_dec_vars if name in agent.outer_iterate_values])
    agent.lam_x0 = np.zeros(agent.stacked_dec_vals_at_l.shape)

    agent.stacked_dec_vars_consensus = cas.vertcat(*[agent.named_copied_dec_vars[name] for name in agent.named_dec_vars if name in agent.named_copied_dec_vars])
    agent.param = cas.MX.sym(f"A{agent.id}_param", cas.vertcat(agent.stacked_dec_vars_consensus, agent.consensus_multiplier).shape)
    agent.eq_multiplier_var = cas.MX.sym('eq_mult', cas.vertcat(*agent.local_equality_constraints).shape)
    agent.ineq_multiplier_var = cas.MX.sym('ineq_mult', cas.vertcat(*agent.local_inequality_constraints).shape)

    agent.grad_objective = cas.gradient(agent.local_objective, agent.stacked_dec_vars)
    agent.grad_objective = cas.Function(f'grad_{agent.id}', [agent.stacked_dec_vars], [agent.grad_objective])

    local_Lag = agent.local_objective \
        + cas.dot(agent.eq_multiplier_var, cas.vertcat(*agent.local_equality_constraints)) \
        + cas.dot(agent.ineq_multiplier_var, cas.vertcat(*agent.local_inequality_constraints))
    agent.hess_Lag, _ = cas.hessian(local_Lag, cas.vertcat(*[agent.named_dec_vars[name] for name in agent.named_dec_vars]))
    agent.hess = cas.Function(f'hess_{agent.id}', [agent.stacked_dec_vars, agent.eq_multiplier_var, agent.ineq_multiplier_var], [agent.hess_Lag])

    agent.eq_cstr = cas.Function(f'g_{agent.id}', [agent.named_dec_vars[name] for name in agent.named_dec_vars], 
                                 [cas.vertcat(*agent.local_equality_constraints)], 
                                 [name for name in agent.named_dec_vars], [f'g'])
    agent.ineq_cstr = cas.Function(f'h_{agent.id}', [agent.named_dec_vars[name] for name in agent.named_dec_vars], 
                                   [cas.vertcat(*agent.local_inequality_constraints)], 
                                   [name for name in agent.named_dec_vars], [f'h'])

    if solver == 'gurobi':
        if not hasattr(agent, 'gurobi_model'):
            agent.gurobi_model = gp.Model()
            if verbose < 4:
                agent.gurobi_model.setParam('OutputFlag', 0)
            agent.gurobi_dec_vars = agent.gurobi_model.addMVar(agent.stacked_dec_vars.shape, lb=-np.inf, ub=np.inf, name='dec_vars')
            agent.gurobi_eq_cstr = None
            agent.gurobi_ineq_cstr = None
        else:
            agent.gurobi_model.remove(agent.gurobi_eq_cstr)
            agent.gurobi_model.remove(agent.gurobi_ineq_cstr)
            agent.gurobi_model.update()
    return agent

def _prepare_agent_qp(agent, admm_penalty, sqp_iter, solver, print_level, max_iter, feas_tol, verbose):
    # Stack the current outer iterate values.
    agent.stacked_outer_iterate_values = cas.vertcat(*[agent.outer_iterate_values[name] for name in agent.named_dec_vars if name in agent.outer_iterate_values])
    
    # Evaluate the equality constraints at the current outer iterate.
    agent.eq_cstr_at_k = agent.eq_cstr.call(agent.outer_iterate_values)[f'g']
    jac_eq_cstr = cas.Function(f'jac_g_{agent.id}', [agent.stacked_dec_vars], [cas.jacobian(cas.vertcat(*agent.local_equality_constraints), agent.stacked_dec_vars)])
    agent.jac_eq_cstr_at_k = jac_eq_cstr(agent.stacked_outer_iterate_values)
    
    # Evaluate the inequality constraints at the current outer iterate.
    agent.ineq_cstr_at_k = agent.ineq_cstr.call(agent.outer_iterate_values)[f'h']
    jac_ineq_cstr = cas.Function(f'jac_h_{agent.id}', [agent.stacked_dec_vars], [cas.jacobian(cas.vertcat(*agent.local_inequality_constraints), agent.stacked_dec_vars)])
    agent.jac_ineq_cstr_at_k = jac_ineq_cstr(agent.stacked_outer_iterate_values)
    
    # Evaluate the Hessian and ensure a positive definite approximation.
    agent.hess_objective_at_k = agent.hess(agent.stacked_outer_iterate_values, agent.eq_multiplier, agent.ineq_multiplier)
    min_eigval = min(np.linalg.eigvalsh(agent.hess_objective_at_k))
    if min_eigval < -1 and (isinstance(agent, Satellite) or isinstance(agent, Quadrotor)):
        # Evaluate the Hessian while setting the equality multipliers to 0.
        # This is especially necessary for the satellite agent, where the nonlinear dynamics together and terminal equality constraints can lead to heavily non-convex Hessians.
        old_min_eigval = min_eigval
        agent.hess_objective_at_k = agent.hess(agent.stacked_outer_iterate_values, np.zeros(agent.eq_multiplier.shape), agent.ineq_multiplier)
        min_eigval = min(np.linalg.eigvalsh(agent.hess_objective_at_k))
        if verbose > 2:
            print(f"[Satellite {agent.id}] Hessian corrected by zeroing equality multipliers. Old min eig: {old_min_eigval:.2e}; new min eig: {min_eigval:.2e}")
    if min_eigval < 1e-8:
        # Add a multiple of the identity matrix to the Hessian to ensure definiteness.
        agent.hess_objective_at_k += (1e-8 - min_eigval)*np.eye(agent.hess_objective_at_k.shape[0])
        
    # Evaluate the gradient.
    agent.grad_objective_at_k = agent.grad_objective(agent.stacked_outer_iterate_values)
    
    agent.stacked_copied_dec_vals_at_l = np.array(agent.stacked_outer_iterate_values).copy()

    if verbose > 1:
        if solver != 'ipopt':
            print(f'Setting up QP problem of agent {agent.id} using {solver}...')
        else:
            print(f'Setting up NLP problem of agent {agent.id} using {solver}...')
            
    if solver == 'gurobi':
        # Compute the quadratic term.
        agent.H_qp = np.array(agent.hess_objective_at_k) + admm_penalty*np.eye(agent.hess_objective_at_k.shape[0])
        
        # Cast evaluated constraints and their gradients into numpy arrays.
        agent.jac_eq_cstr_at_k = np.array(agent.jac_eq_cstr_at_k)
        agent.eq_cstr_at_k = np.array(agent.eq_cstr_at_k)
        agent.jac_ineq_cstr_at_k = np.array(agent.jac_ineq_cstr_at_k)
        agent.ineq_cstr_at_k = np.array(agent.ineq_cstr_at_k)
        agent.stacked_outer_iterate_values = np.array(agent.stacked_outer_iterate_values) 
        
        if sqp_iter == 0:
            # Set the equality constraints.
            cstr = agent.gurobi_model.addConstr( agent.jac_eq_cstr_at_k @ agent.gurobi_dec_vars == agent.jac_eq_cstr_at_k @ agent.stacked_outer_iterate_values - agent.eq_cstr_at_k )
            agent.gurobi_eq_cstr = cstr
            
            # Set the inequality constraints.
            cstr = agent.gurobi_model.addConstr( agent.jac_ineq_cstr_at_k @ agent.gurobi_dec_vars <= agent.jac_ineq_cstr_at_k @ agent.stacked_outer_iterate_values - agent.ineq_cstr_at_k )
            agent.gurobi_ineq_cstr = cstr
            
        else:
            # Remove old constraints from model
            agent.gurobi_model.remove(agent.gurobi_eq_cstr)
            agent.gurobi_model.remove(agent.gurobi_ineq_cstr)
            agent.gurobi_model.update()

            # Rebuild equality constraints.
            cstr = agent.gurobi_model.addConstr( agent.jac_eq_cstr_at_k @ agent.gurobi_dec_vars == agent.jac_eq_cstr_at_k @ agent.stacked_outer_iterate_values - agent.eq_cstr_at_k )
            agent.gurobi_eq_cstr = cstr

            # Rebuild inequality constraints.
            cstr = agent.gurobi_model.addConstr( agent.jac_ineq_cstr_at_k @ agent.gurobi_dec_vars <= agent.jac_ineq_cstr_at_k @ agent.stacked_outer_iterate_values - agent.ineq_cstr_at_k )
            agent.gurobi_ineq_cstr = cstr
            
            agent.gurobi_model.update()
        
    else:
        # Initialise the inner equality constraints.
        agent.inner_equality_constraints = agent.jac_eq_cstr_at_k @ agent.stacked_dec_vars
        # Initialise the inner inequality constraints.
        agent.inner_inequality_constraints = agent.jac_ineq_cstr_at_k @ agent.stacked_dec_vars
        # Initialise the objective function for ADMM.
        agent.aug_Lag_expr = (
                0.5*(agent.stacked_dec_vars - agent.stacked_outer_iterate_values).T @ agent.hess_objective_at_k @ (agent.stacked_dec_vars - agent.stacked_outer_iterate_values)
                + agent.grad_objective_at_k.T @ (agent.stacked_dec_vars - agent.stacked_outer_iterate_values)
                + agent.param[agent.stacked_dec_vars_consensus.shape[0]:].T @ (agent.stacked_dec_vars - agent.param[0:agent.stacked_dec_vars_consensus.shape[0]])
                + 0.5*admm_penalty*cas.dot(agent.stacked_dec_vars - agent.param[0:agent.stacked_dec_vars_consensus.shape[0]], agent.stacked_dec_vars - agent.param[0:agent.stacked_dec_vars_consensus.shape[0]])
                )
        agent.nlp_inner = {
                'x': agent.stacked_dec_vars, 
                'f': agent.aug_Lag_expr,
                'p': agent.param,
                'g': cas.vertcat(agent.inner_equality_constraints, agent.inner_inequality_constraints)
                }
            
        if solver == 'ipopt':
            # Set up the solver.
            nlp_inner_options = {'ipopt': 
                {'print_level': print_level,
                    'sb': 'yes',
                    'max_iter': max_iter,
                    'linear_solver': 'ma97',
                    'warm_start_init_point': 'yes',
                    'constr_viol_tol': feas_tol,
                    'tol': 1e-8,
                    'dual_inf_tol': 1e-8
                    }
                }
            agent.S_inner_program = cas.nlpsol('S_nlp_inner', 'ipopt', agent.nlp_inner, nlp_inner_options)
        else:
            if solver == 'qpoases':
                solver_opts = {
                    'printLevel': 'none' if verbose <= 2 else 'low'
                }
            elif solver == 'osqp':
                solver_opts = {
                    'verbose': verbose > 2
                }
            else:
                solver_opts = {}
            agent.S_inner_program = cas.qpsol('S_qp_inner', solver, agent.nlp_inner, solver_opts)
    return agent


def _solve_agent_qp(agent, admm_penalty):
    # Update the linear term of the objective function. (The quadratic term is constant in the inner iteration.)
    agent.g_qp = np.array(
        agent.grad_objective_at_k
        - agent.hess_objective_at_k @ agent.stacked_outer_iterate_values
        + agent.consensus_multiplier_at_l
        - admm_penalty * agent.stacked_copied_dec_vals_at_l
    )
    quad_expr = 0.5 * agent.gurobi_dec_vars.T @ agent.H_qp @ agent.gurobi_dec_vars + agent.g_qp.T @ agent.gurobi_dec_vars
    agent.gurobi_model.setObjective(quad_expr, gp.GRB.MINIMIZE)
    agent.gurobi_model.update()
    # Warm start using the current iterate (gurobi may ignore it).
    agent.gurobi_dec_vars.start = np.array(agent.stacked_dec_vals_at_l)

    # Solve the QP.
    agent.gurobi_model.optimize()
    if agent.gurobi_model.status == gp.GRB.INF_OR_UNBD:
        print(f"Gurobi returned status {agent.gurobi_model.status} for agent {agent.id}. Disabling dual reductions and reoptimizing.")
        agent.gurobi_model.setParam('DualReductions', 0)
        agent.gurobi_model.optimize()
        # Check the status again after reoptimization
        if agent.gurobi_model.status == gp.GRB.INFEASIBLE:
            raise RuntimeError(f"QP for agent {agent.id} is infeasible.")
        elif agent.gurobi_model.status == gp.GRB.UNBOUNDED:
            raise RuntimeError(f"QP for agent {agent.id} is unbounded.")
        else:
            raise RuntimeError(f"Reoptimization resulted in status {agent.gurobi_model.status} for agent {agent.id}.")
    elif agent.gurobi_model.status == gp.GRB.SUBOPTIMAL:
        print(f"Gurobi solved QP of agent {agent.id} only to suboptimality.") 
    elif agent.gurobi_model.status != gp.GRB.OPTIMAL:
        print(f"Gurobi failed at agent {agent.id} — status {agent.gurobi_model.status}")
    # Compute the constant term of the objective function for comparison.
    const = (
        0.5*agent.stacked_outer_iterate_values.T @ agent.hess_objective_at_k @ agent.stacked_outer_iterate_values 
        - agent.grad_objective_at_k.T @ agent.stacked_outer_iterate_values
        - agent.consensus_multiplier_at_l.T @ agent.stacked_copied_dec_vals_at_l
        + 0.5*admm_penalty*agent.stacked_copied_dec_vals_at_l.T @ agent.stacked_copied_dec_vals_at_l
    )
    # Extract primal and dual solutions.
    agent.stacked_dec_vals_at_l = np.array(agent.gurobi_dec_vars.X)
    agent.eq_multiplier = agent.gurobi_eq_cstr.Pi
    # Gurobi returns non-positive dual values for inequality constraints defined as Ax <= b.
    # However, we approximate h(x) <= 0 in the KKT sense, so expect positive dual values. Compare:
    # https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/constraintlinear.html#pi
    agent.ineq_multiplier = -agent.gurobi_ineq_cstr.Pi
    
    ## Extract and assign the solution to the agent.
    agent.dec_vals_at_l = {}
    agent.copied_dec_vals_at_l = {}
    agent.named_consensus_multiplier_at_l = {}
    start = 0
    for var_name, var_sym in agent.named_dec_vars.items():
        rows, cols = var_sym.shape
        step = rows*cols
        agent.dec_vals_at_l[var_name] = agent.stacked_dec_vals_at_l[start:start+step].reshape(rows, cols)
        agent.copied_dec_vals_at_l[var_name] = np.vstack([agent.stacked_copied_dec_vals_at_l[start:start+step]]).reshape(rows, cols)
        agent.named_consensus_multiplier_at_l[var_name] = np.vstack([agent.consensus_multiplier_at_l[start:start+step]]).reshape(rows, cols)
        start += step

    return agent  # Return the modified agent.


def get_ic_MPC_constraints(agent, u, x, t=None, tol=1e-8):
    """Return the MPC constraints on the initial condition, this is a equality constraint.
    
    Checks if the current state is feasible with respect to the agents affine and nonlinear state constraints.

    By convention, equality constraints are always assumed to be equal to zero.
    
    Parameters:
    - agent (Agent): Agent solving the optimization problem.
    - u (casadi MX): Decision variable, the input sequence vertically stacked (at least) from u(0) to u(horizon-1).
    - x (casadi MX): Decision variable, the state sequence vertically stacked (at least) from x(1) to x(horizon).
    
    Returns:
    In a tuple with the following order
    - constraints (list): Contains the MX equation that defines the constraints.
    - constraints_lb (list): Contains the lower bound, i.e. 0, to the MX equation at the same index in 'constraints'.
    - constraints_ub (list): Contains the upper bound, i.e. 0, to the MX equation at the same index in 'constraints'.
    - t (int): Optional time step at which the initial condition is evaluated. Used for warnings. (default is None)
    - tol (float): Tolerance for checking of initial state's feasibility.
    """
    # Define shorthands for state and input dimensions.
    n = agent.state_dim
    q = agent.input_dim
    
    # Collect the constraints and a corresponding upper and lower bound in a list.
    # Same indices should correspond to the same scalar constraint.
    constraints = []
    constraints_lb = []
    constraints_ub = []
    
    infeasible = False
    residual = 0.0   
    
    # Check if the initial state is feasible.
    if "A" in agent.state_constraints and "b" in agent.state_constraints:
        A = agent.state_constraints["A"]
        b = agent.state_constraints["b"]
        cstr = A@agent.current_state - b
        if np.max(cstr) > tol:
            residual = np.max([residual, np.max(cstr)])
            infeasible = True
            
    if agent.nonlinear_constraints is not None:
        for cstr in agent.nonlinear_constraints:
            if len(cstr.name_out()) != 1 and cstr.name_out()[0] != 'g':
                raise ValueError(f"Nonlinear constraints must have exactly one output named 'g'. Check Agent {agent.id}.")
            if len(cstr.name_in()) == 1 and cstr.name_in()[0] == 'x':
                cstr_value = cstr(x=agent.current_state)['g']
                if np.max(cstr_value) > tol:
                    infeasible = True
                    residual = np.max([residual, np.max(cstr_value)])
            else: 
                raise NotImplementedError(f"Nonlinear constraints depending on {cstr.name_in()} are not implemented.")
    
    # Set the initial condition as a constraints.
    # Note that the first entry of x contains the first predicted state.
    constraints.append(agent.dynamics(agent.current_state, u[0 : q, 0]) - x[0 : n, 0])
    # These constraints are equality constraints.
    constraints_lb.append(np.zeros((n,1)))
    constraints_ub.append(np.zeros((n,1)))
    
    if infeasible:
        if t is not None:
            warnings.warn(f"Initial state appears infeasible at t={t} with residual {residual:.3e}.", RuntimeWarning)
        else:
            warnings.warn("Initial state appears infeasible with residual {residual:.3e}", RuntimeWarning)
    return constraints, constraints_lb, constraints_ub


def get_system_MPC_constraints(agent, u, x, horizon):
    """Return the MPC constraints on dynamics and pointwise state and input constraints. 
    
    Inequality constraints are upper bounded by zero.
    Equality constraints are equal to zero.

    Parameters:
    - agent (Agent): Agent solving the optimisation problem.
    - u (casadi MX): Decision variable, the input sequence vertically stacked (at least) from u(0) to u(horizon-1).
    - x (casadi MX): Decision variable, the state sequence vertically stacked (at least) from x(1) to x(horizon).
    - horizon (int): Prediction horizon of the MPC optimisation problem.
    
    Returns:
    In a tuple with the following order
    - ineq_constraints (list): Contains the MX equation that defines the ineqquality constraints.
    - ineq_constraints_lb (list): Contains the lower bound to the MX equation at the same index in 'constraints'. For inequality constraints this is always -inf.
    - ineq_constraints_ub (list): Contains the upper bound to the MX equation at the same index in 'constraints'. If there is an upper bound, it is set to 0 and the value of the constraint shifted.
    - eq_constraints (list): Contains the MX equation that defines the equality constraints.
    - eq_constraints (list): Contains the lower bound, i.e. 0, to the MX equation at the same index in 'constraints'.
    - eq_constraints (list): Contains the upper bound, i.e. 0, to the MX equation at the same index in 'constraints'.
    - ineq_names (list): Contains the names of the inequality constraints.
    - eq_names (list): Contains the names of the equality constraints.
    """
    # Define shorthands for state and input dimensions.
    n = agent.state_dim
    q = agent.input_dim
    
    # Collect the constraints and a corresponding upper and lower bound in a list.
    # Same indices should correspond to the same scalar constraint.
    ineq_constraints = []
    ineq_constraints_lb = []
    ineq_constraints_ub = []
    ineq_names = []
    eq_constraints = []
    eq_constraints_lb = []
    eq_constraints_ub = []
    eq_names = []
    
    # Create constraints containing the dynamics.
    # The initial coupling between x(0), u(0) and x(1) is part of the initial condition, provided by a different method.
    for i in range(0, horizon-1):
        eq_constraints.append(agent.dynamics(x[i*n : (i+1)*n, 0], u[(i+1)*q : (i+2)*q, 0]) - x[(i+1)*n : (i+2)*n, 0])
        # These constraints are equality constraints.
        eq_constraints_lb.append(np.zeros((n,1)))
        eq_constraints_ub.append(np.zeros((n,1)))
        eq_names.extend([f'A{agent.id}_dynamics_{i}']*n)
    
    # Set the state constraints if there are any.
    # Do not add constraints on the fixed initial state.
    if "A" in agent.state_constraints and "b" in agent.state_constraints:
        A = agent.state_constraints["A"]
        b = agent.state_constraints["b"]
        for t in range(horizon):
            ineq_constraints.append(A@x[t*n : (t+1)*n, 0] - b)
            ineq_constraints_lb.append(-np.inf*np.ones((b.shape[0],1)))
            ineq_constraints_ub.append(np.zeros((b.shape[0],1)))
            ineq_names.extend([f'A{agent.id}_state_{t}']*b.shape[0])

    # Set the input constraints if there are any.
    if "A" in agent.input_constraints and "b" in agent.input_constraints:
        A = agent.input_constraints["A"]
        b = agent.input_constraints["b"]
        for t in range(horizon):
            ineq_constraints.append(A@u[t*q : (t+1)*q, 0] - b)
            ineq_constraints_lb.append(-np.inf*np.ones((b.shape[0],1)))
            ineq_constraints_ub.append(np.zeros((b.shape[0],1)))
            ineq_names.extend([f'A{agent.id}_input_{t}']*b.shape[0])
      
    # Add nonlinear constraints if there are any.      
    if agent.nonlinear_constraints is not None:
        for cstr in agent.nonlinear_constraints:
            if len(cstr.name_out()) != 1 and cstr.name_out()[0] != 'g':
                raise ValueError(f"Nonlinear constraints must have exactly one output named 'g'. Check Agent {agent.id}.")
            if len(cstr.name_in()) == 1 and cstr.name_in()[0] == 'x':
                for k in range(horizon):
                    ineq_constraints.append(cstr(x=x[k*n:(k+1)*n, 0])['g'])
                    ineq_constraints_lb.append(-np.inf*np.ones(cstr.size_out('g')))
                    ineq_constraints_ub.append(np.zeros(cstr.size_out('g')))
                    ineq_names.extend([f'{cstr.name_out()[0]}_{agent.id}_x_{k}']*cstr.size_out('g')[0])
            else: 
                raise NotImplementedError(f"Nonlinear constraints depending on {cstr.name_in()} are not implemented.")

    return ineq_constraints, ineq_constraints_lb, ineq_constraints_ub, eq_constraints, eq_constraints_lb, eq_constraints_ub, ineq_names, eq_names

   
def get_standard_MPC_constraints(agent, u, x, horizon, t=None, tol=1e-8):
    """Return the standard MPC constraints. Inequality constraints are upper bounded by zero.
    
    Deprecated call with x from x(0) to x(N) (i.e. containing x(0)) is passed on to deprecated function.
    
    Checks if the initial state is feasible up to the tolerance with respect to affine and nonlinear constraints on the state.

    The standard constraints are:
    * initial condition
    * dynamics
    * pointwise state and input constraints

    Parameters:
    - agent (Agent): Agent solving the optimisation problem.
    - u (casadi MX): Decision variable, the input sequence vertically stacked (at least) from u(0) to u(horizon-1).
    - x (casadi MX): Decision variable, the state sequence vertically stacked (at least) from x(1) to x(horizon).
    - horizon (int): Prediction horizon of the MPC optimisation problem.
    - t (int): Optional current time step used for warnings. (default is None)
    - tol (float): Tolerance for checking of initial state's feasibility.
    
    Returns:
    In a tuple with the following order
    - constraints (list): Contains the MX equation that defines the constraints.
    - constraints_lb (list): Contains the lower bound to the MX equation at the same index in 'constraints'. For inequality constraints this is always -inf.
    - constraints_ub (list): Contains the upper bound to the MX equation at the same index in 'constraints'. If there is an upper bound, it is set to 0 and the value of the constraint     shifted.
    - eq_constraints (list): Contains the MX equation that defines the equality constraints.
    - eq_constraints (list): Contains the lower bound, i.e. 0, to the MX equation at the same index in 'constraints'.
    - eq_constraints (list): Contains the upper bound, i.e. 0, to the MX equation at the same index in 'constraints'.
    - ineq_constraints_names (list): Contains the names of the inequality constraints.
    - eq_constraints_names (list): Contains the names of the equality constraints.
    """
    # Define shorthands for state and input dimensions.
    n = agent.state_dim
    
    if x.shape[0] > n*horizon:
        raise ValueError("The state decision variable should start at x(1) and end at x(horizon+1).")
    
    # Collect the constraints and a corresponding upper and lower bound in a list.
    # Same indices should correspond to the same scalar constraint.
    ineq_constraints = []
    ineq_constraints_lb = []
    ineq_constraints_ub = []
    ineq_constraints_names = []
    eq_constraints = []
    eq_constraints_lb = []
    eq_constraints_ub = []
    eq_constraints_names = []
    
    # Get the constraints for the initial condition.
    eq_cstr, eq_cstr_lb, eq_cstr_ub = get_ic_MPC_constraints(agent, u, x, t, tol)
    eq_constraints.extend(eq_cstr)
    eq_constraints_lb.extend(eq_cstr_lb)
    eq_constraints_ub.extend(eq_cstr_ub)
    eq_constraints_names.extend(["initial condition"]*sum([c.shape[0] for c in eq_cstr_lb]))
    
    # Get the constraints for the dynamics as well as state and input constraints.
    cstr, cstr_lb, cstr_ub, eq_cstr, eq_cstr_lb, eq_cstr_ub, ineq_names, eq_names = get_system_MPC_constraints(agent, u, x, horizon)
    ineq_constraints.extend(cstr)
    ineq_constraints_lb.extend(cstr_lb)
    ineq_constraints_ub.extend(cstr_ub)
    ineq_constraints_names.extend(ineq_names)
    eq_constraints.extend(eq_cstr)
    eq_constraints_lb.extend(eq_cstr_lb)
    eq_constraints_ub.extend(eq_cstr_ub)
    eq_constraints_names.extend(eq_names)

    return ineq_constraints, ineq_constraints_lb, ineq_constraints_ub, eq_constraints, eq_constraints_lb, eq_constraints_ub, ineq_constraints_names, eq_constraints_names



def get_bounds_of_affine_constraint(A, b, solver='CLARABEL'):
    """
    Find per-dimension lower and upper bounds for x in R^n
    subject to A x <= b. We assume this defines a non-empty,
    bounded (closed) polytope.
    
    Requires cvxpy.

    Parameters
    ----------
    - A (np.ndarray (shape = (m, n))): 
        Matrix in the inequality constraints A x <= b.
    - b (np.ndarray (shape = (m,))) :
        Vector in the inequality constraints.
    - solver (str): Solver used in cvxpy. (default is 'CLARABEL')
    
    Returns
    -------
    - lower_bounds (np.ndarray of shape (n,)) : 
        The minimal feasible value of x[i] for each dimension i.
    - upper_bounds (np.ndarray of shape (n,)): 
        The maximal feasible value of x[i] for each dimension i.
    """
    import cvxpy
    
    A = np.asarray(A)
    b = np.asarray(b)
    m, n = A.shape

    lower_bounds = np.zeros(n)
    upper_bounds = np.zeros(n)

    # For each dimension i, solve two linear problems:
    #   (1) minimize x[i] subject to A x <= b
    #   (2) maximize x[i] subject to A x <= b
    for i in range(n):
        x_var = cvxpy.Variable(n, name="x")

        constraints = [A @ x_var <= b]

        # -----------------------
        # 1) Minimize x[i]
        # -----------------------
        objective_min = cvxpy.Minimize(x_var[i])
        prob_min = cvxpy.Problem(objective_min, constraints)
        result_min = prob_min.solve(solver=solver)

        if prob_min.status not in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]:
            raise ValueError(
                f"Minimization for dimension {i} failed with status: {prob_min.status}"
            )
        lower_bounds[i] = x_var[i].value

        # -----------------------
        # 2) Maximize x[i]
        # -----------------------
        x_var2 = cvxpy.Variable(n, name="x2")
        constraints2 = [A @ x_var2 <= b]
        objective_max = cvxpy.Maximize(x_var2[i])
        prob_max = cvxpy.Problem(objective_max, constraints2)
        result_max = prob_max.solve(solver=solver)

        if prob_max.status not in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]:
            raise ValueError(
                f"Maximization for dimension {i} failed with status: {prob_max.status}"
            )
        upper_bounds[i] = x_var2[i].value

    return lower_bounds, upper_bounds



def generate_reference_grid(agent, resol: int):
    """
    Generate reference grid points for the state and input, based on cooperation constraints.

    Args:
        agent: Agent with cooperation constraints defined.
        resol: Number of samples per dimension for state and input (uniform).

    Returns:
        dict with keys 'xT' and 'uT', each a list of numpy column vectors.
    """
    # Get bounds from cooperation constraints.
    x_lbs, x_ubs = get_bounds_of_affine_constraint(agent.cooperation_constraints['Ax'], agent.cooperation_constraints['bx'])
    u_lbs, u_ubs = get_bounds_of_affine_constraint(agent.cooperation_constraints['Au'], agent.cooperation_constraints['bu'])

    n = agent.state_dim
    m = agent.input_dim

    if len(x_lbs) != n or len(x_ubs) != n:
        raise ValueError(f"State bounds length mismatch with state dimension.")
    if len(u_lbs) != m or len(u_ubs) != m:
        raise ValueError(f"Input bounds length mismatch with input dimension.")

    # Generate evenly spaced grid points per dimension.
    x_axes = [np.linspace(x_lbs[i], x_ubs[i], resol) for i in range(n)]
    u_axes = [np.linspace(u_lbs[i], u_ubs[i], resol) for i in range(m)]

    # Mesh and flatten.
    x_mesh = np.meshgrid(*x_axes, indexing='ij')
    u_mesh = np.meshgrid(*u_axes, indexing='ij')

    x_coords = np.column_stack([x.ravel() for x in x_mesh])
    u_coords = np.column_stack([u.ravel() for u in u_mesh])

    # Generate Cartesian product of x and u.
    xT = []
    uT = []

    for x in x_coords:
        x_col = np.vstack(x.reshape(-1, 1))  # shape (n, 1)
        for u in u_coords:
            u_col = np.vstack(u.reshape(-1, 1))  # shape (m, 1)
            xT.append(x_col)
            uT.append(u_col)

    return {'xT': xT, 'uT': uT}


def generate_lhs_reference_grid(agent, num_samples: int, seed: int = None):
    """
    Generate reference points using Latin Hypercube Sampling (LHS) from the cooperation constraint set.

    Args:
        agent: The agent with cooperation constraints defined.
        num_samples: Number of total reference points (xT, uT) to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        dict with keys 'xT' and 'uT', each a list of column vectors (np.ndarray with shape (n, 1) or (m, 1)).
    """
    from scipy.stats import qmc

    if seed is not None:
        np.random.seed(seed)

    # Get bounds for state and input references.
    x_lbs, x_ubs = get_bounds_of_affine_constraint(agent.cooperation_constraints['Ax'], agent.cooperation_constraints['bx'])
    u_lbs, u_ubs = get_bounds_of_affine_constraint(agent.cooperation_constraints['Au'], agent.cooperation_constraints['bu'])

    n = agent.state_dim
    m = agent.input_dim

    dim = n + m  # Total number of variables in each sample.

    # Generate LHS samples in [0, 1]^dim and scale to the correct ranges.
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    sample = sampler.random(n=num_samples)  # Shape: (num_samples, dim)

    # Rescale to the [lb, ub] bounds.
    x_bounds = np.column_stack((x_lbs, x_ubs))
    u_bounds = np.column_stack((u_lbs, u_ubs))
    bounds = np.vstack((x_bounds, u_bounds))  # Shape: (n+m, 2)

    scaled_samples = qmc.scale(sample, bounds[:, 0], bounds[:, 1])  # (num_samples, dim)

    # Split into state and input lists of column vectors.
    xT = [scaled_samples[i, :n].reshape(-1, 1) for i in range(num_samples)]
    uT = [scaled_samples[i, n:].reshape(-1, 1) for i in range(num_samples)]

    return {'xT': xT, 'uT': uT}


def save_generic_terminal_ingredients(agent, filepath) -> None:
    """Save the generic terminal ingredients of an agent to a dill file."""
    terminal_ingredients = agent.terminal_ingredients
    if not filepath.endswith(".pkl"):
        filepath += ".pkl"
    with open(filepath, "wb") as file:
        dill.dump(terminal_ingredients, file)


def load_generic_terminal_ingredients(agent, filepath):
    """Load the generic terminal ingredients (X, Y, terminal set size) of an agent from a dill file."""
    # Check if the path ends with ".npz" and append if necessary
    if not filepath.endswith(".pkl"):
        filepath += ".pkl" 
    with open(filepath, "rb") as file:
        terminal_ingredients = dill.load(file)
    if hasattr(agent, 'terminal_ingredients'):
        agent.terminal_ingredients.update(terminal_ingredients)
    else:
        agent.terminal_ingredients = terminal_ingredients