# %% [markdown]
# # Example: Ships navigating a harbour area
#  
# This notebook sets up the simulation for the example in
#  
# > Distributed MPC for dynamic cooperation of without terminal constraints --- Matthias Köhler, Matthias A. Müller, and Frank Allgöwer
#  
# If turned on below, the simulation data is saved to a data file in the folder `./data/`.
# This is recommended if the exported python file is run in order to access the simulation data later.
# The data can be visualised using the accompanying notebook `harbour_evaluation.ipynb`.
#  
# The simulation data used in the paper is contained in the file `./data/harbour_data.dill`. 
# This data is animated in `harbour.mp4`.

# %%
"""Imports"""
import warnings
import mkmpc.mkmpc as mkmpc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import casadi as cas
import dill
from datetime import datetime
import time
import auxiliaries as aux

# %%
"""Main settings"""
start_time = time.time()  # Time the total execution of the script.
save_data = True  # Whether to save the data at the end.

# Decide whether to continuously save data during the simulation on the hard drive.
# Data of previous runs is overwritten; only one save file is kept.
continuous_saving_time = 5  # Save data every 'continuous_saving_time' seconds. If 0, no continuous saving is done.
generate_animation = True     # Whether to save the animation at the end.

# ---------------------------------------------------------------------------------------------------------
# MAS parameters
# ---------------------------------------------------------------------------------------------------------
MAS_type = 'vessel'
N = 60                           # Set the prediction horizon used in the MPC optimization problem.
h = 0.5                           # Set the step size of the discretization of the continuous-time dynamics.
num_agents = 3                     # Set number of agents.
method = 'RK4'                    # Choose the integration method: 'Euler', 'RK4', 'RK2' (where applicable)
distance = 0.4                      # Set the minimum distance between vessels.       
# ---------------------------------------------------------------------------------------------------------
# Simulation parameters.
# ---------------------------------------------------------------------------------------------------------
max_sim_time = int(30*60/h)                # Set the maximum simulation time step.
terminal_ingredients_type = 'without'   # Choice between 'set', 'equality', and 'without'.
cutoff_treshold = -1e-6                 # Stop the simulation if the value function falls below this threshold.
average_treshold = -1e-6                # Stop the simulation if the standard deviation of the value function falls below this threshold.
max_iter = 1000                         # Maximum number of iterations for ipopt. None uses ipopt's default.

admm_max_iter = 80                      # Number of ADMM iterations to solve the QP of each SQP iteration.
admm_penalty = 200                       # Penalty parameter for ADMM.

# sqp_max_iter = 25                        # Number of SQP iterations in the dSQP method.
# solver_dSQP = 'gurobi'                   # Solver for local QPs, e.g. 'osqp', 'qpoases', 'gurobi', 'ipopt' used in the dSQP method.

parallel = True                         # Whether to use parallelization for the local QPs.
# ---------------------------------------------------------------------------------------------------------
# Cooperative task.
# ---------------------------------------------------------------------------------------------------------
T = 1
coop_task = 'harbour'  # Set the cooperative task.
print(f"Cooperative task is '{coop_task}'.")
print(f"Last simulation time is {max_sim_time*h} s (~ {max_sim_time*h // 60} min) with {max_sim_time} simulation steps.")
print(f"Period length is {T*h} s (~ {T*h // 60} min) with {T} simulation steps.")
print(f"Prediction horizon is {N*h} s (~ {N*h // 60} min) with {N} prediction steps.")


# %%
def vessel_warm_start_at_t0(agents, N, T):
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
            x_ws[i*n:(i+1)*n] = np.copy(x)
        
        x = np.copy(agent.current_state)
        for i in range(T):
            xT_ws[i*n:(i+1)*n] = x.copy()
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

# %%
"""Initialise data saving."""
data = {}
data['cooperative_task'] = {}
data['cooperative_task']['type'] = coop_task
data['MAS_parameters'] = {}
data['MAS_parameters']['num_agents'] = num_agents
data['MAS_parameters']['MAS_type'] = MAS_type
data['MAS_parameters']['h'] = h
data['MAS_parameters']['collision_distance'] = distance
data['sim_data'] = {'max_sim_time': max_sim_time}
data['sim_pars'] = {'N': N,
                    'cutoff_threshold': cutoff_treshold,
                    'average_treshold': average_treshold,
                    'terminal_ingredients_type': terminal_ingredients_type,
                    'max_iter': max_iter,
                    'T': T,
                    'admm_max_iter': admm_max_iter, 
                    'admm_penalty': admm_penalty}

# %%
def get_vessel_MAS(data, method):
    # Introduce a scaling factor to scale the lengths from m to 20 m.
    sf = 0.05

    # Set parameters.
    Mh = 493.77*sf
    Mr = 493.77*sf
    Mw = 55.81*sf**2
    Dh = 29.23*sf
    Dr = 2173.7*sf
    Dw = 17.7*sf**2

    tauh_min = -500.*sf**2
    tauh_max = 1000.*sf**2
    tauh_tight = 0.1*sf**2
    tauw_min = -20.*sf**3
    tauw_max = 20.*sf**3
    tauw_tight = 0.1*sf**3
    vh_max = 9.*sf
    vh_min = -9.*sf
    vh_tight = 0.1*sf
    vr_max = 9.*sf
    vr_min = -9.*sf
    vr_tight = 0.1*sf
    omega_max = 0.3
    omega_min = -0.3
    omega_tight = 0.01
    #--------------------------------------------------
    # Save some of the parameters in the data dictionary.
    if True:
        data['MAS_parameters']['Mh'] = Mh
        data['MAS_parameters']['Mr'] = Mr
        data['MAS_parameters']['Mw'] = Mw
        data['MAS_parameters']['Dh'] = Dh
        data['MAS_parameters']['Dr'] = Dr
        data['MAS_parameters']['Dw'] = Dw
        data['MAS_parameters']['tauh_min'] = tauh_min
        data['MAS_parameters']['tauh_max'] = tauh_max
        data['MAS_parameters']['tauw_min'] = tauw_min
        data['MAS_parameters']['tauw_max'] = tauw_max
        data['MAS_parameters']['vh_max'] = vh_max
        data['MAS_parameters']['vh_min'] = vh_min
        data['MAS_parameters']['vr_max'] = vr_max
        data['MAS_parameters']['vr_min'] = vr_min
        data['MAS_parameters']['omega_max'] = omega_max
        data['MAS_parameters']['omega_min'] = omega_min
        data['MAS_parameters']['tauw_tight'] = tauw_tight
        data['MAS_parameters']['tauh_tight'] = tauh_tight
        data['MAS_parameters']['vh_tight'] = vh_tight
        data['MAS_parameters']['vr_tight'] = vr_tight
        data['MAS_parameters']['omega_tight'] = omega_tight
        data['MAS_parameters']['scaling_factor'] = sf

    coop_task = data['cooperative_task']['type']
    num_agents = data['MAS_parameters']['num_agents']
    h = data['MAS_parameters']['h']

    # Save information about agents in a list.
    data['agents'] = {}

    agents = []  # Initialize a list to collect the agents in.
    for i in range(num_agents):
        # Initialize the satellite agent.
        agents.append(mkmpc.Vessel(h, Mh, Mr, Mw, Dh, Dr, Dw, method=method))

    for agent in agents:
        data['agents'][f'A{agent.id}'] = {}

    """Set constraints for each agent."""
    for i, agent in enumerate(agents):

        #--------------------------------------------------------
        # Set state constraints.
        #--------------------------------------------------------
        A_hbr_x = np.vstack([
            np.hstack([[-1.,          0.        ], np.hstack([0.0]*(agent.state_dim-2))]),
            np.hstack([[-0.,          1.        ], np.hstack([0.0]*(agent.state_dim-2))]),
            np.hstack([[-0.31622777, -0.9486833 ], np.hstack([0.0]*(agent.state_dim-2))]),
            np.hstack([[ 0.14142136, -0.98994949], np.hstack([0.0]*(agent.state_dim-2))]),
            np.hstack([[ 1.,         -0.        ], np.hstack([0.0]*(agent.state_dim-2))]),
            np.hstack([[ 0.27472113,  0.96152395], np.hstack([0.0]*(agent.state_dim-2))])
            ])
        b_hbr = np.vstack([0., 8., -3.32039154, -1.34350288, 15., 8.92843666])
        Ax = np.array([
                    [0., 0.,  0.,  1.,  0.,  0.],
                    [0., 0.,  0., -1.,  0.,  0.],
                    [0., 0.,  0.,  0.,  1.,  0.],
                    [0., 0.,  0.,  0., -1.,  0.],
                    [0., 0.,  0.,  0.,  0.,  1.],
                    [0., 0.,  0.,  0.,  0., -1.]])
        bx = np.array([[vh_max],
                       [-vh_min],
                       [vr_max],
                       [-vr_min],
                       [omega_max],
                       [-omega_min]])

        agent.state_constraints['A'] = np.vstack([A_hbr_x, Ax])
        agent.state_constraints['b'] = np.vstack([b_hbr, bx])

        bx_tight = np.vstack([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        agent.cooperation_constraints['Ax'] = np.vstack([A_hbr_x, Ax])
        agent.cooperation_constraints['bx'] = np.vstack([b_hbr-1e-2, bx_tight])

        A_hbr_yT = np.array([
            [-1.,          0.        , 0.],
            [-0.,          1.        , 0.],
            [-0.31622777, -0.9486833 , 0.],
            [ 0.14142136, -0.98994949, 0.],
            [ 1.,         -0.        , 0.],
            [ 0.27472113,  0.96152395, 0.]])
        agent.cooperation_constraints['Ay'] = A_hbr_yT
        agent.cooperation_constraints['by'] = b_hbr-1e-2

        #--------------------------------------------------------
        # Set input constraints.
        #--------------------------------------------------------
        agent.input_constraints['A'] = np.array([
            [ 1,  0],
            [-1,  0],
            [ 0,  1],
            [ 0, -1]
            ])
        agent.input_constraints['b'] = np.array([
            [tauh_max],  [-tauh_min],
            [tauw_max],  [-tauw_min]
            ])
        agent.cooperation_constraints['Au'] = agent.input_constraints['A']
        agent.cooperation_constraints['bu'] = np.vstack([0., 0., 0., 0.])

    #--------------------------------------------------------
    # Set the stage costs.
    #--------------------------------------------------------
    for agent in agents:
        # Define the artificial equilibrium.
        x = cas.MX.sym('x', agent.state_dim)
        u = cas.MX.sym('u', agent.input_dim)
        xT = cas.MX.sym('x_c', agent.state_dim)
        uT = cas.MX.sym('u_c', agent.input_dim)

        Q = np.vstack([
            1.0 / (3.0**4),
            10.0 / (3.0**2),
            1.0 / ((np.pi/6)**4),
            0.01 / (9.0*sf)**4, 
            0.01 / (9.0*sf)**4,
            1.0 / (0.3**4)
        ])
        R = np.vstack([
            1 / ((50*sf**2)**4),
            0.001 / ((20.0*sf**3)**4)
        ])

        # Add stage cost to agents.
        agent.stage_cost = cas.Function(
            'stage_cost',
            [x, u, xT, uT],
            [
               Q[0]*((x[0] - xT[0])*cas.sin(xT[2]) + (x[1] - xT[1])*cas.cos(xT[2]))**4
             + Q[1]*((x[0] - xT[0])*cas.cos(xT[2]) - (x[1] - xT[1])*cas.sin(xT[2]))**2
             + Q[2]*(x[2]-xT[2])**4
             + Q[3]*(x[3]-xT[3])**4
             + Q[4]*(x[4]-xT[4])**4
             + Q[5]*(x[5]-xT[5])**4
             + R[0]*(u[0]-uT[0])**4
             + R[1]*(u[1]-uT[1])**4
             ],
            ['x', 'u', 'xT', 'uT'],
            ['l'])
        agent.stage_cost_weights = {'Q': Q, 'R': R}
        data['agents'][f'A{agent.id}']['stage_cost'] = {'Q': Q, 'R': R}

    #----------------------------------------
    # Define the communication topology.
    #----------------------------------------
    # Define an all-to-all topology.
    for i, agent in enumerate(agents):
        agent.neighbours = []
        for j in range(len(agents)):
            if j != i:
                agent.neighbours.append(agents[j])

    return agents


# %%
"""Multi-agent system"""
agents = get_vessel_MAS(data, method)
sf = data['MAS_parameters']['scaling_factor']
positions = [
    np.vstack([ 6.0, 7.0, np.radians(  90), 0.5*sf, 0.0, 0.0]),
    np.vstack([12.0, 4.0, np.radians(- 30), 0.5*sf, 0.0, 0.0]),
    np.vstack([ 4.0, 6.0, np.radians(  30), 0.5*sf, 0.0, 0.0]),
    np.vstack([ 4.0, 3.0, np.radians(   0), 0.0*sf, 0.0, 0.0]),
    np.vstack([ 7.0, 2.5, np.radians(  90), 0.0*sf, 0.0, 0.0]),
    np.vstack([ 4.0, 6.0, np.radians(  90), 0.0*sf, 0.0, 0.0]),
    np.vstack([ 0.2, 5.0, np.radians(  90), 0.0*sf, 0.0, 0.0]),
    np.vstack([ 4.0, 7.0, np.radians(  90), 0.0*sf, 0.0, 0.0]),
    np.vstack([ 6.0, 5.0, np.radians(  90), 0.0*sf, 0.0, 0.0])
    ]
data['sim_pars']['positions'] = positions
for idx, agent in enumerate(agents):
    agent.current_state = np.vstack([positions[idx]])

# %%
"""Cooperative task"""
## Define the task of navigating through a harbour.

# Decide whether ship can leave the harbour and new ships can enter.
spawning = True
data['cooperative_task']['spawning'] = spawning

for agent in agents:
    agent.data['stage_flag'] = 0  # Initalize the stage flag.
    # 0 = agent was not in the harbour.
    # 1 = agent was in the harbour and is guided to first point.
    # 2 = agent was in the harbour and is guided to second point.
    # 3 = agent was in the harbour and is guided to edge.

if len(agents) > 1:
    agents[1].data['stage_flag'] = 3  # Set the second agent to stage 1, i.e. it is guided to the edge.
    agents[1].data['leaving_heading'] = np.radians(-90)

def set_cooperative_task_to_harbour(t:int, agents:list, spawning:dict, weight:float=1.0, N=None, distance=0.0) -> int:
    """Design a cooperative task inspired by a busy harbour.
                            
    Arguments:
        - t (int): Current time step.
        - agents (list): List of active agents (mkmpc.Agent).
        - spawning (bool): Whether agents leave the harbour and 'new' agents enter.
            If an agent has visited the harbour and reaches the western boundary, it 'leaves' the harbour, i.e. it is reset and enters the harbour again from south-west.
        - weight (float): Multiplicative weight of the cooperation objective function. (default is 1.0)
        - N (int): Prediction horizon.
    """
    # Define the point in the harbour where the agents should go. Do not provide a heading.
    harbour_point = np.vstack([13., 5.5])
    target1 = np.vstack([14., 5.])
    target2 = np.vstack([12., 4.])
    exit_point = np.vstack([2., 7.])  # Define the 'exit' point.

    # Initialize the cooperation decision variables and define collision avoidance constraints.
    for agent in agents:
        if 'coop_task_initialized' in agent.data and agent.data['coop_task_initialized']:
            continue
        else:
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
            
            # Add collision avoidance constraints to the agent.
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

    # If called at the initial time step, initialize constraints. These are time-invariant.
    for i, agent in enumerate(agents):
        if 'coop_task_initialized' in agent.data and agent.data['coop_task_initialized']:
            continue
        else:
            # Retrieve the decision variables.
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
            
            # Add coupling constraints on the position of the agent in the cooperation output.
            enlarged_distance = distance + 0.1
            for nghbr in agent.neighbours:
                yT_nghbr = nghbr.named_cooperation_dec_vars[f'A{nghbr.id}_yT']
                for tau in range(T):   
                    cooperation_constraint_map.append(enlarged_distance**2 - cas.dot(yT[tau*p : tau*p + 2, 0] - yT_nghbr[tau*p : tau*p + 2, 0], yT[tau*p : tau*p + 2, 0] - yT_nghbr[tau*p : tau*p + 2, 0]))  
                agent.named_cooperation_dec_vars[yT_nghbr.name()] = yT_nghbr
            
            # Add the peninsula (an ellipsoid) that separates the harbour from the inlet.
            d = np.radians(-40)
            
            v = cas.MX.sym('x', agent.state_dim)
            peninsula = cas.Function('peninsula', [v], [-1e-4*((((v[0] - 9.5)*cas.cos(d) + (v[1] - 3.3)*cas.sin(d)) / 3.0)**8 + (((v[1] - 3.3)*cas.cos(d) - (v[0] - 9.5)*cas.sin(d)) / 1.7)**8 - 1)], ['x'], ['g'])
            agent.nonlinear_constraints = [peninsula]
                
            # Add a slightly wider peninsula (an ellipsoid) that separates the harbour from the inlet to the cooperation constraints.
            v = cas.MX.sym('v', agent.output_dim)
            peninsula = cas.Function('peninsula', [v], [1e-4*((((v[0] - 9.5)*cas.cos(d) + (v[1] - 3.3)*cas.sin(d)) / (3.0+1e-1))**8 + (((v[1] - 3.3)*cas.cos(d) - (v[0] - 9.5)*cas.sin(d)) / (1.7+1e-1))**8 - 1)], ['v'], ['value'])
            for tau in range(T):
                cooperation_constraint_map.append(-peninsula(yT[tau*p : (tau+1)*p, 0]))
                
            # Add the constraint to the agent.
            agent.cooperation_constraints['function'] = cas.Function(f'A{agent.id}_cooperation_constraint', agent.named_cooperation_dec_vars.values(), [cas.vertcat(*cooperation_constraint_map)], agent.named_cooperation_dec_vars.keys(), [f'A{agent.id}_value'])
            
            agent.data['coop_task_initialized'] = True  # Mark the agent as initialized.
    
    # Define the cooperation cost. If the agent has not been in the harbour, the goal is to reach the harbour. Otherwise, the goal is to leave the harbour.
    for agent in agents:
        p = agent.output_dim
        yT = agent.named_cooperation_dec_vars[f'A{agent.id}_yT']
        # Check where the agent is and assign the task.
        if agent.data['stage_flag'] == 0 and np.linalg.norm(agent.current_state[0:2] - harbour_point[0:2]) <= 0.3:
            # Agent is in the harbour and has not been there before.
            agent.data['stage_flag'] = 1
            # Check the heading of the agent and compute the new heading such that the vessel can turn.
            heading = agent.current_state[2]
            if type(heading) == cas.DM:
                heading = float(heading)
            if heading >= 0:
                desired_heading = heading + (np.radians(270) - heading % (2*np.pi))
            else:
                desired_heading = heading + (-np.radians(90) - heading % (-2*np.pi))
            agent.data['leaving_heading'] = desired_heading
        elif agent.data['stage_flag'] == 1 and np.linalg.norm(agent.current_state[0:2] - target1) <= 0.3:
            # Agent has reached first point in the harbour.
            agent.data['stage_flag'] = 2
        elif agent.data['stage_flag'] == 2 and np.linalg.norm(agent.current_state[0:2] - target2) <= 0.3:
            # Agent has reached second point in the harbour.
            agent.data['stage_flag'] = 3
            agent.data['desired_heading'] = agent.data['leaving_heading']
            
        cooperation_objective = cas.MX(0)
        
        if 'stage_flag' in agent.data and agent.data['stage_flag'] == 1:
            # Agent has been in the harbour and is guided to the first point.
            agent.data['desired_heading'] = agent.data['leaving_heading'] - np.radians(90)
            target = np.vstack([target1, agent.data['desired_heading']])
            # Build the cooperation cost using a Huber loss function. Exiting the harbour takes precedence, i.e. the cost has a larger weight. 
            for tau in range(T):
                cooperation_objective = (
                    cooperation_objective 
                    + 100*(1/1**2)*(          target[0] - yT[tau*p]   )**2 
                    + 100*(1/1**2)*(          target[1] - yT[tau*p+1] )**2 
                    + 100*(20/((np.pi)**2))*( target[2] - yT[tau*p+2] )**2
                )
        elif 'stage_flag' in agent.data and agent.data['stage_flag'] == 2:
            # Agent has been in the harbour and is guided to the second point.
            agent.data['desired_heading'] = agent.data['leaving_heading'] + np.radians(30)
            target = np.vstack([target2, agent.data['desired_heading']])
            for tau in range(T):
                cooperation_objective = (
                    cooperation_objective 
                    + 100*(1/1.0**2)*(        target[0] - yT[tau*p]   )**2 
                    + 100*(1/1.0**2)*(        target[1] - yT[tau*p+1] )**2 
                    + 100*(20/((np.pi)**2))*( target[2] - yT[tau*p+2] )**2
                )
        elif 'stage_flag' in agent.data and agent.data['stage_flag'] == 3:
            # Agent has been in the harbour and is guided to the exit of the inlet.
            agent.data['desired_heading'] = agent.data['leaving_heading']
            target = np.vstack([exit_point, agent.data['desired_heading']])  # Define a target position at the exit of the inlet.
            # Build the cooperation cost using a Huber loss function. Exiting the harbour takes precedence, i.e. the cost has a larger weight. 
            for tau in range(T):
                cooperation_objective = (
                    cooperation_objective 
                    + (400/10**2)*(      target[0] - yT[tau*p]   )**2 
                    + (15/3**2)*(        target[1] - yT[tau*p+1] )**2 
                    + (1/((np.pi)**4))*( target[2] - yT[tau*p+2] )**4
                )
        else:
            # Agent has not been in the harbour.
            # agent.data['desired_heading'] = harbour_point[2]
            for tau in range(T):
                cooperation_objective = (
                    cooperation_objective 
                    + (400/10**2)*(        harbour_point[0] - yT[tau*p]   )**2 
                    + (15/3**2)*(          harbour_point[1] - yT[tau*p+1] )**2
                )

        # Create the objective function and assign it to the agent.
        cooperation_objective = weight*cooperation_objective/T
        agent.cooperation_objective_function = cas.Function(f'A{agent.id}_cooperation_objective_function', agent.named_cooperation_dec_vars.values(), [cooperation_objective], agent.named_cooperation_dec_vars.keys(), [f'A{agent.id}_value'])

    if spawning:
        # Check if an agent has left the harbour and reset it.
        # If multiple agents reach the edge at the same time, we would put them on top of each other, immediately destroying feasibility due to collision avoidance constraints.
        # Hence, we count how many agents are leaving the harbour and space them out accordingly.
        for agent in agents:               
            if agent.current_state[0] <= 1 and agent.data['stage_flag'] > 0:
                respawn_collision = False  # Check if another agent is in the way. If so, defer the agent to the next time step.
                # Try the lower position.
                for nghbr in agent.neighbours:
                    if np.linalg.norm(nghbr.current_state[0:2] - np.vstack([0.1, 3.55])) <= 0.8:
                        respawn_collision = True
                        break
                if not respawn_collision:
                    agent.current_state = np.vstack([0.1, 3.55, np.radians(90), .1, 0.0, 0.0])
                    agent.data['stage_flag'] = 0
                # Try the upper position.
                respawn_collision = False
                for nghbr in agent.neighbours:
                    if np.linalg.norm(nghbr.current_state[0:2] - np.vstack([0.1, 4.45])) <= 0.8:
                        respawn_collision = True
                        break
                    if not respawn_collision:
                        agent.current_state = np.vstack([0.1, 4.45, np.radians(90), .1, 0.0, 0.0])
                        agent.data['stage_flag'] = 0

coop_task_builder = set_cooperative_task_to_harbour
coop_kwargs={'t': 0, 'agents': agents, 'spawning': spawning, 'weight': 1.0, 'N': N, 'distance': distance}
data['cooperative_task']['T'] = T
data['cooperative_task']['kwargs'] = coop_kwargs

# Call the task builder to establish constraints.
set_cooperative_task_to_harbour(**coop_kwargs)

# %%
if terminal_ingredients_type == 'without':
    for agent in agents:
        # Initialize a tracking bound.
        agent.tracking_bound = 10
        print(f'A{agent.id} tracking bound: {agent.tracking_bound}')
        data['agents'][f'A{agent.id}']['tracking_bound'] = agent.tracking_bound
else:
    raise NotImplementedError(f"Terminal ingredients type '{terminal_ingredients_type}' is not implemented.")        

# %%
"""Simulation run"""
# Initalization:
positions = data['sim_pars']['positions']
for idx, agent in enumerate(agents):
    agent.current_state = np.vstack([positions[idx]])

# Initialize data tracking.
data['sim_data']['yT'] = {}  # Track the cooperation outputs.
data['sim_data']['xT'] = {}  # Track the cooperation state trajectory.
data['sim_data']['uT'] = {}  # Track the cooperation input trajectory.
data['sim_data']['x'] = {}  # Track the open-loop state prediction.
data['sim_data']['u'] = {}  # Track the open-loop input prediction.
data['sim_data']['tracking_cost'] = []  # Track the value of the tracking part.
data['sim_data']['cooperative_cost'] = []  # Track the value of the cooperation objective function part.
data['sim_data']['change_cost'] = []  # Track the value of the penalty on the change of the cooperation output part.
data['sim_data']['J'] = []  # Track the value of the cost.
    
for agent in agents:
    data['sim_data']['xT'][f'A{agent.id}'] = []
    data['sim_data']['yT'][f'A{agent.id}'] = []
    data['sim_data']['uT'][f'A{agent.id}'] = []
    data['sim_data']['x'][f'A{agent.id}'] = []
    data['sim_data']['u'][f'A{agent.id}'] = []
    
# Initialize the penalty weight for the change in the cooperation output.
for agent in agents:
    # Initialize an empty previously optimal cooperation output for each agent.
    agent.yT_pre = None
    agent.MPC_sol = None
    
    agent.penalty_weight = 1e-4/T  # Set the weight of the penalty on the change in the cooperation output.
    data['agents'][f'A{agent.id}']['penalty_weight'] = agent.penalty_weight

# Build the closed-loop state evolution of each agent and save it as an attribute of the agent.
for agent in agents:
    agent.cl_x = [agent.current_state.copy()]
    agent.cl_u = []
    
# Initialize a filestamp if continuous saving is activated.
if save_data and continuous_saving_time > 0:
    filestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filestamp = f'{filestamp}_quicksave_{coop_task}'

for t in range(0, max_sim_time+1):    
    print(f"{t}: -------------------------------------------------------------------------")  # Print the time step:
    if t == 0:
        warm_start = vessel_warm_start_at_t0(agents, N, T)
    else:
        warm_start = mkmpc.compute_ADMM_warm_start_dynamic_cooperative_DMPC(t, agents, T, N, max_iter, coop_task_builder, coop_kwargs, terminal_ingredients_type, 10, 2, keep_multiplier=False)
    
    coop_kwargs['t'] = t
    coop_kwargs['agents'] = agents
    
    # Generate and solve the optimization problem for MPC for dynamic cooperation.
    #-------------------- direct ADMM method --------------------
    res = mkmpc.solve_MPC_for_dynamic_cooperation_with_ADMM(
        admm_max_iter, 
        admm_penalty,
        t, 
        agents, 
        N=N, 
        T=T,
        feas_tol=1e-4,  # Default in IPOPT is 1e-4.
        warm_start=warm_start,
        terminal_ingredients_type=terminal_ingredients_type,
        coop_task_builder=coop_task_builder, 
        coop_kwargs=coop_kwargs,
        max_iter=max_iter,
        verbose=2,
        parallel=parallel,
        print_level = 0
    )
    # res = mkmpc.solve_MPC_for_dynamic_cooperation_decentrally(
    #     sqp_max_iter, 
    #     admm_max_iter, 
    #     admm_penalty,
    #     t, 
    #     agents, 
    #     N=N, 
    #     T=T,
    #     feas_tol=1e-8, 
    #     warm_start=warm_start, 
    #     solver=solver_dSQP,
    #     terminal_ingredients_type=terminal_ingredients_type,
    #     coop_task_builder=coop_task_builder, 
    #     coop_kwargs=coop_kwargs,
    #     max_iter=max_iter,
    #     verbose=2,
    #     parallel=parallel
    # )
    
    print(f'Solved at time step {t} with f* = {float(res["J"]):.5e}')
    
    data['sim_data']['tracking_cost'].append(res['tracking_cost'])
    data['sim_data']['cooperative_cost'].append(res['cooperative_cost'])
    data['sim_data']['change_cost'].append(res['change_cost']) 
    data['sim_data']['J'].append(res['J']) 
    
    for agent in agents:
        # # Keep track of open-loop solutions, reshaped ready for plotting.
        data['sim_data']['yT'][f'A{agent.id}'].append(agent.MPC_sol[f'A{agent.id}_yT'].reshape(T, agent.output_dim).T)
        data['sim_data']['xT'][f'A{agent.id}'].append(agent.MPC_sol[f'A{agent.id}_xT'].reshape(T, agent.state_dim).T)
        data['sim_data']['uT'][f'A{agent.id}'].append(agent.MPC_sol[f'A{agent.id}_uT'].reshape(T, agent.input_dim).T)
        data['sim_data']['u'][f'A{agent.id}'].append(agent.MPC_sol[f'A{agent.id}_u'].reshape(N, agent.input_dim).T)
        # The prediction starts with x(1|t), hence x(0|t) = x(t) needs to be prepended.
        data['sim_data']['x'][f'A{agent.id}'].append(np.hstack([np.array(agent.current_state), agent.MPC_sol[f'A{agent.id}_x'].reshape(N, agent.state_dim).T]))

        # Update the current state of the agents.
        agent.current_state = agent.dynamics(x=agent.current_state, u=agent.MPC_sol[f'A{agent.id}_u'][0:agent.input_dim])['x+']
        agent.cl_x.append(np.array(agent.current_state))  # Keep track of the current state.
        agent.cl_u.append(np.array(agent.MPC_sol[f'A{agent.id}_u'][0:agent.input_dim]))  # Keep track of the current input.
        
        # Set the previously optimal trajectory:
        agent.yT_pre = np.vstack([agent.MPC_sol[f'A{agent.id}_yT'][agent.output_dim :], agent.MPC_sol[f'A{agent.id}_yT'][0 : agent.output_dim]])
        
    # Stop the simulation if the cost falls below a threshold.
    if res['J'] <= cutoff_treshold:
        print(f"The value function has fallen below {cutoff_treshold} at time step {t}.")
        data['sim_data']['max_sim_time'] = t
        break
    # Stop the simulation if the cost has converged; i.e. the standard deviation over a window has fallen below a threshold.
    if t > 10 and np.std(data['sim_data']['J'][t-10:t]) <= average_treshold:
        print(f"The standard deviation of the value function has fallen below {average_treshold} at time step {t}.")
        data['sim_data']['max_sim_time'] = t
        break
    # Save the data after each specified time step.
    if save_data and t > 0 and continuous_saving_time > 0 and t % continuous_saving_time == 0:
        aux.save_data(data, agents, filestamp)
        
end_time = time.time()
elapsed = end_time - start_time
print(f"Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

# %% [markdown]
# ## Plotting

# %%
"""Colour palette"""
colours = [
    "#0072B2",  # Blue
    "#D55E00",  # Rich orange
    "#009E73",  # Strong green
    "#CC79A7",  # Soft magenta
    "#56B4E9",  # Light sky blue
    "#E69F00",  # Deep yellow-orange
    "#B22222",  # Firebrick red
    "#6A3D9A",  # Deep purple
    "#117733",  # Deep teal green
    "#88CCEE",  # Light cyan-blue
    "#DDCC77",  # Muted yellow-orange
]

# %%
"""Transform data."""
# Transform the costs into numpy arrays.
data['sim_data']['cooperative_cost'] = np.vstack(data['sim_data']['cooperative_cost']).flatten()
data['sim_data']['tracking_cost'] = np.vstack(data['sim_data']['tracking_cost']).flatten()
data['sim_data']['change_cost'] = np.vstack(data['sim_data']['change_cost']).flatten()
data['sim_data']['J'] = np.vstack(data['sim_data']['J']).flatten()

# Extract some parameters.
max_sim_time = data['sim_data']['max_sim_time']

# Transform the tracked closed-loop trajectories of each agent into a matrix.
for agent in agents:
    if type(agent.cl_x) == list:
        agent.cl_x = np.hstack(agent.cl_x)
        agent.cl_u = np.hstack(agent.cl_u)


# %%
"""Save data"""
if save_data:
    aux.save_data(data, agents)

# %%
"""Extract data"""
max_sim_time = data['sim_data']['max_sim_time']

# %%
"""Value function."""
# Plot from t1 to t2.
t1 = 0
t2 = max_sim_time+1

# Select a feasible start time (the end time is controlled below).
t1 = min(t1, max_sim_time+1)

# Draw the evolution in state space:
fig_V, ax_V = plt.subplots(figsize=(10, 6), num='state evolution')

stop_time = data['sim_data']['cooperative_cost'][t1:t2].shape[0]
ax_V.plot(range(t1, min(t2, stop_time)), data['sim_data']['cooperative_cost'][t1:t2], label='cooperative', color=colours[0])
ax_V.plot(range(t1, min(t2, stop_time)), data['sim_data']['tracking_cost'][t1:t2], label='tracking', color=colours[1])
ax_V.plot(range(max(t1,1), min(t2, stop_time)), data['sim_data']['change_cost'][max(t1,1):t2], label='change', color=colours[2])
ax_V.plot(range(t1, min(t2, stop_time)), data['sim_data']['J'][t1:t2], '--', label='J', color=colours[3])
    
ax_V.set_xlabel('time steps')
ax_V.set_title(f'Value function over time')
ax_V.legend()
ax_V.grid(True)

# Set the y-axis to logarithmic scale.
ax_V.set_yscale('log')
#ax_V.set_ylim((np.min(data["sim_data"]["J"]), np.max(data["sim_data"]["J"])))
# ax_V.set_ylim((500, 605))

print(f'Value function difference between the first and last time step: {data["sim_data"]["J"][-1] - data["sim_data"]["J"][0]}')
print(f'Value function at start: {data["sim_data"]["J"][0]:15.4e}')
print(f'Value function at stop : {data["sim_data"]["J"][-1]:15.4e}; diff: {data["sim_data"]["J"][-1] - data["sim_data"]["J"][0]:15.4e}')
print(f'Cooperation cost at start : {data["sim_data"]["cooperative_cost"][0]:15.4e}')
print(f'Cooperation cost at stop : {data["sim_data"]["cooperative_cost"][-1]:15.4e}; diff: {data["sim_data"]["cooperative_cost"][-1] - data["sim_data"]["cooperative_cost"][0]:15.4e}')

plt.show()


# %%
"""Compute constraints"""
def f(x, y, n, d):
    if n % 2 != 0:
        raise ValueError("n must be even")
    d = np.radians(d)
    x = x - 9.5
    y = y - 3.3
    return ((x*np.cos(d) + y*np.sin(d)) / 3)**n + ((y*np.cos(d) - x*np.sin(d)) / 1.7)**n - 1
x1 = np.linspace(-1, 16, 400)
x2 = np.linspace(1, 9, 400)
X1, X2 = np.meshgrid(x1, x2)
peninsula = f(X1, X2, 8, -40)

# %%
"""2D position"""
# Plot from t1 to t2.
t1 = 0#max_sim_time+1-max(2*T, 5)
t2 = agent.cl_x.shape[1]-1
step = 1

# Select a feasible start time (the end time is controlled automatically).
t1 = min(t1, max_sim_time+1)

# Draw the evolution in state space:
fig_cl, ax_cl = plt.subplots(figsize=(10, 10), num='state evolution')

for i, agent in enumerate(agents):
    cl_x = np.zeros(agent.cl_x.shape)
    cl_x = agent.cl_x
    ax_cl.plot(cl_x[0, t1 : t2+1:step], cl_x[1,t1 : t2+1:step], color=colours[i], label=f'A{agent.id}_x', 
               #marker='o', markersize=2, 
               linewidth=1.5)
    # Mark the initial state with a larger circle.
    ax_cl.plot(cl_x[0,t1], cl_x[1,t1], color=colours[i], marker='o', markersize=6)
    # Mark the final state with a cross.
    ax_cl.plot(cl_x[0,t2], cl_x[1,t2], color=colours[i], marker='x', markersize=6)

xlabel='$x_1$'
ylabel='$x_2$'

# Draw the peninsula.
avrt = [[0, 3.5], [4.5, 2.0], [15., 3.5], [15.0, 5.0], [4.5, 8.0], [0.0, 8.0]]
avrt = np.array(avrt)
avrt = np.vstack([avrt, avrt[0]])
ax_cl.plot(avrt[:, 0], avrt[:, 1], 'k', linewidth=1.5)
ax_cl.contour(X1, X2, peninsula, levels=[0], colors='black', linewidths=1.5)
ax_cl.set_xlim(-1, 16)
ax_cl.set_ylim(1, 9)

ax_cl.set_title(f'Closed-loop position from $t = {t1}$ to $t = {t2}$ with step {step}')
ax_cl.set_xlabel(xlabel)
ax_cl.set_ylabel(ylabel)
ax_cl.grid()

plt.show()

# %%
"""All states and inputs"""
# Define time range
t1 = 0
t2_state = max_sim_time + 1
t2_input = max_sim_time

# Number of states and inputs
num_states = 6
num_inputs = 2

# Agents to plot
agents2plot = agents[:]

# Select feasible start time
t1 = min(t1, max_sim_time + 1)

# Create a figure with multiple subplots (4 rows, 2 columns)
fig, axes = plt.subplots(4, 2, figsize=(12, 12), num='State & Input Evolution')

## --- Plot All 6 States ---
for idx_state in range(num_states):
    ax = axes[idx_state // 2, idx_state % 2]  # Get subplot position
    title_state = f'Closed-loop state $x_{idx_state+1}$ from t = {t1} to t = {t2_state}'

    for i, agent in enumerate(agents):
        if agent not in agents2plot:
            continue
        tf = min(t2_state, agent.cl_x.shape[1] - 1)
        if idx_state == 2:
            ax.plot(range(t1, tf+1), np.degrees(agent.cl_x[idx_state, t1:tf+1]), 
                    color=colours[i], label=f'{agent.id}_x{idx_state+1}', markersize=0, linewidth=2, marker='o')
        else:
            ax.plot(range(t1, tf+1), agent.cl_x[idx_state, t1:tf+1], 
                    color=colours[i], label=f'{agent.id}_x{idx_state+1}', markersize=0, linewidth=2, marker='o')

    if np.linalg.norm(ax.get_ylim()) < 1e-8:
        ax.set_ylim(-0.1, 0.1)
        
    ax.grid()
    ax.legend()
    ax.set_title(title_state)
    ax.set_xlabel('time steps')
    ax.set_ylabel(f'$x_{idx_state+1}$')

## --- Plot All 2 Inputs ---
for idx_input in range(num_inputs):
    ax = axes[3, idx_input]  # Get subplot position (last row)
    title_input = f'Closed-loop input $u_{idx_input+1}$ from t = {t1} to t = {t2_input}'


    for i, agent in enumerate(agents):
        if agent not in agents2plot:
            continue
        tf = min(t2_input, agent.cl_u.shape[1] - 1)
        ax.plot(range(t1, tf+1), agent.cl_u[idx_input, t1:tf+1], 
                color=colours[i], label=f'{agent.id}_u{idx_input+1}', markersize=0, linewidth=2, marker='o')

    if np.linalg.norm(ax.get_ylim()) < 1e-8:
        ax.set_ylim(-0.1, 0.1)
        
    ax.grid()
    ax.legend()
    ax.set_title(title_state)
    ax.set_xlabel('time steps')
    ax.set_ylabel(f'$u_{idx_input+1}$')

# Adjust layout and show plot
plt.tight_layout()
plt.show()



