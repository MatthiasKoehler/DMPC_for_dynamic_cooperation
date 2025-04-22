# %% [markdown]
# # Example: Crossing a narrow path
# 
# This notebook sets up the simulation for the example in Section VI.A of
# 
# > Distributed Model Predictive Control for Dynamic Cooperation of Multi-Agent Systems --- Matthias Köhler, Matthias A. Müller, and Frank Allgöwer
# 
# If turned on below, the simulation data is saved to a data file in the folder `./data/`.
# This is recommended if the exported python file is run in order to access the simulation data later.
# The data can be visualised using the accompanying notebook `narrow_path_evaluation.ipynb`.
# The simulation data used in the paper is contained in the file `./data/narrow_path_data.dill`. 

# %%
"""Imports"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import casadi as cas
import dill
from datetime import datetime
import auxiliaries as aux

# %%
"""Main settings"""
# ---------------------------------------------------------------------------------------------------------
# Data saving on hard drive to './data/'.
# ---------------------------------------------------------------------------------------------------------
save_data = True  # Whether to save the simulation data to a file.
save_interval_steps = 10  # Save data every 'save_interval_steps' steps. If 0, no continuous saving is done.

# ---------------------------------------------------------------------------------------------------------
# MAS parameters
# ---------------------------------------------------------------------------------------------------------
MAS_type = 'double_integrator'
N = 20                             # Set the prediction horizon used in the MPC optimization problem.
num_agents = 2                     # Set number of agents.

# ---------------------------------------------------------------------------------------------------------
# Simulation parameters.
# ---------------------------------------------------------------------------------------------------------
max_sim_time = 1                      # Set the last simulation time step.
terminal_ingredients_type = 'equality'  # Set terminal equality constraints.
cutoff_treshold = 1e-9                  # Stop the simulation if the value function falls below this threshold.
average_treshold = 1e-9                 # Stop the simulation if the standard deviation of the value function falls below this threshold.
max_iter = None                         # Maximum number of iterations for ipopt. None allows ipopt's default.

sqp_max_iter = 100                      # Number of SQP iterations.
admm_max_iter = 50                      # Number of ADMM iterations to solve the QP in each SQP iteration.  
admm_penalty = 200                      # Penalty parameter for ADMM.

solver = 'gurobi'                       # Solver for local QPs, e.g. 'osqp', 'qpoases', 'gurobi', 'ipopt'
parallel = True                         # Whether to use parallelization for the local QPs.

# ---------------------------------------------------------------------------------------------------------
# Cooperative task.
# ---------------------------------------------------------------------------------------------------------
T = 1                                   # Set the period length of the cooperative task.
coop_task = 'narrow_position_exchange'  # Fix the cooperative task.
# The cooperative task is defined in detail below.


# %%
"""Initialize data saving."""
data = {}
data['cooperative_task'] = {}
data['cooperative_task']['type'] = coop_task
data['MAS_parameters'] = {}
data['MAS_parameters']['num_agents'] = num_agents
data['MAS_parameters']['MAS_type'] = MAS_type
data['sim_data'] = {'max_sim_time': max_sim_time}
data['sim_pars'] = {'N': N,
                    'cutoff_threshold': cutoff_treshold,
                    'average_treshold': average_treshold,
                    'terminal_ingredients_type': terminal_ingredients_type,
                    'max_iter': max_iter,
                    'T': T,
                    'sqp_max_iter': sqp_max_iter,
                    'admm_max_iter': admm_max_iter, 
                    'admm_penalty': admm_penalty,
                    'solver': solver}

# %%
"""Multi-agent system"""
if MAS_type == 'double_integrator': 
    agents = aux.get_double_integrator_MAS(data)
else:
    raise ValueError(f'Multi-agent system type "{MAS_type}" is unknown!')

# %%
"""Cooperative task"""
# Define the positions that are exchanged. These are also set as the initial condition (with zeros appended.)
positions = [np.array([[-20.0], [0.0]]), np.array([[20.0], [0.0]])]
data['positions'] = positions

# Set the parameters for the narrow path.
horizontal_length = 12.
vertical_length = 1.2 
vertical_length_tightened = 1.15
exponent = 8

data['cooperative_task']['horizontal_length'] = horizontal_length
data['cooperative_task']['vertical_length'] = vertical_length
data['cooperative_task']['vertical_length_tightened'] = vertical_length_tightened
data['cooperative_task']['exponent'] = exponent
    
# Assign the target positions.
for i, agent in enumerate(agents):
        agent.target_pos = positions[(i+len(positions)//2) % len(positions)]

coop_task_builder = aux.set_cooperative_task_to_narrow_position_exchange
distance = 0.5
data['MAS_parameters']['collision_distance'] = distance
coop_kwargs = {'t': 0, 'agents': agents ,'weight': 1.0, 'N': N, 'data': data, 'distance': 0.5}
data['cooperative_task']['T'] = T
data['cooperative_task']['kwargs'] = coop_kwargs

# Call the task builder to establish constraints.
aux.set_cooperative_task_to_narrow_position_exchange(**coop_kwargs)
# Add a nonlinear constraint to the agents that defines the narrow band.
for agent in agents:
    v = cas.MX.sym('v', agent.state_dim)
    lower_state_blockage = cas.Function('lower_state_blockage', [v], [-((v[0] / horizontal_length)**exponent + ((v[1] + vertical_length) / 1)**exponent - 1)], ['x'], ['g'])
    upper_state_blockage = cas.Function('upper_state_blockage', [v], [-((v[0] / horizontal_length)**exponent + ((v[1] - vertical_length) / 1)**exponent - 1)], ['x'], ['g'])
    agent.nonlinear_constraints = [lower_state_blockage, upper_state_blockage]


# %%
"""Simulation run"""
# Set the initial state of the agents:
for idx, pos in enumerate(positions):
    agents[idx].current_state = np.vstack([pos, np.vstack([0.0]*(agents[idx].state_dim-2))])

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
if save_data and save_interval_steps > 0:
    filestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filestamp = f'{filestamp}_quicksave_{coop_task}'

for t in range(0, max_sim_time+1):    
    print(f"{t}: -------------------------------------------------------------------------")  # Print the time step:
    if t == 0:
        warm_start = aux.double_integrator_warm_start_at_t0(agents, N, T)
    else:
        warm_start = aux.compute_decentralized_following_warm_start_dynamic_cooperative_DMPC(agents, T, N, terminal_ingredients_type=terminal_ingredients_type)
    
    coop_kwargs['t'] = t
    coop_kwargs['agents'] = agents
    # Generate and solve the optimization problem for MPC for dynamic cooperation.
    res = aux.solve_MPC_for_dynamic_cooperation_decentrally(
        sqp_max_iter, admm_max_iter, admm_penalty, t, agents, 
        N=N, T=T, feas_tol=1e-8, 
        warm_start=warm_start, 
        solver=solver,
        terminal_ingredients_type=terminal_ingredients_type,
        coop_task_builder=coop_task_builder, 
        coop_kwargs=coop_kwargs,
        max_iter=max_iter,
        verbose=2,
        parallel=parallel
    )
    
    if 'J' not in res:
        res['J'] = res['f']
    print(f'Solved at time step {t} with f* = {float(res["J"]):.5e}')
    
    data['sim_data']['tracking_cost'].append(res['tracking_cost'])
    data['sim_data']['cooperative_cost'].append(res['cooperative_cost'])
    data['sim_data']['change_cost'].append(res['change_cost']) 
    data['sim_data']['J'].append(res['J']) 
    
    for agent in agents:
        # Keep track of open-loop solutions.
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
    # Stop the simulation if the cost has converged, i.e. the standard deviation over a window has fallen below a threshold.
    if t > 10 and np.std(data['sim_data']['J'][t-10:t]) <= average_treshold:
        print(f"The standard deviation of the value function has fallen below {average_treshold} at time step {t}.")
        data['sim_data']['max_sim_time'] = t
        break
    # Save the data after the specified interval of time steps.
    if save_data and t > 0 and save_interval_steps > 0 and t % save_interval_steps == 0:
        aux.save_data(data, agents, filestamp)


# %% [markdown]
# ## Plotting

# %%
"""Colour palette"""
colours = [
    "#0072B2",  # blue
    "#D55E00",  # orange
    "#009E73",  # green
    "#CC79A7",  # magenta
    "#56B4E9",  # light blue
    "#E69F00",  # yellow-orange
    "#B22222",  # red
    "#6A3D9A",  # purple
    "#117733",  # teal green
    "#88CCEE",  # cyan
    "#DDCC77",  # muted yellow-orange
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
ax_V.set_title('Value function over time')
ax_V.grid()
ax_V.legend()

# ax_V.set_yscale('log')  # Set the y-axis to logarithmic scale.

print(f'Value function difference between the first and last time step: {data["sim_data"]["J"][-1] - data["sim_data"]["J"][0]}')
print(f'Value function at start: {data["sim_data"]["J"][0]:15.4e}')
print(f'Value function at stop : {data["sim_data"]["J"][-1]:15.4e}; diff: {data["sim_data"]["J"][-1] - data["sim_data"]["J"][0]:15.4e}')
print(f'Cooperation cost at start : {data["sim_data"]["cooperative_cost"][0]:15.4e}')
print(f'Cooperation cost at stop : {data["sim_data"]["cooperative_cost"][-1]:15.4e}; diff: {data["sim_data"]["cooperative_cost"][-1] - data["sim_data"]["cooperative_cost"][0]:15.4e}')

plt.show()

# %%
"""2D position"""
horizontal_length = data['cooperative_task']['horizontal_length']
vertical_length = data['cooperative_task']['vertical_length']
vertical_length_tightened = data['cooperative_task']['vertical_length_tightened']
exponent = data['cooperative_task']['exponent']
def lower_x(x, y, n, horizontal, vertical):
    if n % 2 != 0:
        raise ValueError("n must be even")
    return (x / horizontal)**n + ((y + vertical) / 1)**n - 1

def upper_x(x, y, n, horizontal, vertical):
    if n % 2 != 0:
        raise ValueError("n must be even")
    return (x / horizontal)**n + ((y - vertical) / 1)**n - 1

x1 = np.linspace(-horizontal_length-1, horizontal_length+1, 400)
x2 = np.linspace(-3, 3, 400)
X1, X2 = np.meshgrid(x1, x2)

Blx = lower_x(X1, X2, exponent, horizontal_length, vertical_length)
Bux = upper_x(X1, X2, exponent, horizontal_length, vertical_length)
Bly = lower_x(X1, X2, exponent, horizontal_length, vertical_length_tightened)
Buy = upper_x(X1, X2, exponent, horizontal_length, vertical_length_tightened)
# Plot from t1 to t2.
t1 = 0
t2 = max_sim_time+1
step = 1

fig_cl, ax_cl = plt.subplots(figsize=(10, 5), num='state evolution')

for i, agent in enumerate(agents):
    cl_x = np.zeros(agent.cl_x.shape)
    cl_x = agent.cl_x
    
    t2 = min(t2, cl_x.shape[1]-1)
    
    ax_cl.plot(cl_x[0,:], cl_x[1,:], color=colours[i])  # Plot the 2D trajectory of the agent.
    ax_cl.plot(cl_x[0,t1], cl_x[1,t1], color=colours[i], marker='o', markersize=6)  # Mark the initial state with a circle.
    ax_cl.plot(cl_x[0,t2], cl_x[1,t2], color=colours[i], marker='x', markersize=6)  # Mark the final state with a cross.

ax_cl.set_xlabel('$x_1$')
ax_cl.set_ylabel('$x_2$')
ax_cl.contour(X1, X2, Blx, levels=[0], colors='black', linewidths=1)
ax_cl.contour(X1, X2, Bux, levels=[0], colors='black', linewidths=1)
ax_cl.grid()

ax_cl.set_xlim(-20.5, 20.5)
ax_cl.set_ylim(-0.5, 0.5)

plt.show()


