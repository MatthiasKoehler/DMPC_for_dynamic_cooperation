# %% [markdown]
# # Example: Synchronization and flocking
# 
# This notebook sets up the simulation for the example in Section VI.B of
# 
# > Distributed Model Predictive Control for Dynamic Cooperation of Multi-Agent Systems --- Matthias Köhler, Matthias A. Müller, and Frank Allgöwer
# 
# If turned on below, the simulation data is saved to a data file in the folder `./data/`.
# This is recommended if the exported python file is run in order to access the simulation data later.
# The data can be visualised using the accompanying notebook `quadrotor_evaluation.ipynb`.
# 
# The simulation data used in the paper is contained in the file `./data/quadrotor_data.dill`. 
# This data is animated in `quadrotor.mp4` and `quadrotor_3D.mp4`.

# %%
"""Imports"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import casadi as cas
import dill
from datetime import datetime
import auxiliaries as aux
import time
import cvxpy

print(f'CVXPY recognizes the following solvers: {cvxpy.installed_solvers()}')

# %%
"""Main settings"""
start_time = time.time()  # Time the total execution of the script.
# ---------------------------------------------------------------------------------------------------------
# Data saving on hard drive to './data/'.
# ---------------------------------------------------------------------------------------------------------
save_data = True  # Whether to save the simulation data to a file.
save_interval_steps = 10  # Save data every 'save_interval_steps' steps. If 0, no continuous saving is done.

# ---------------------------------------------------------------------------------------------------------
# MAS parameters
# ---------------------------------------------------------------------------------------------------------
MAS_type = 'quadrotor10'
N = 10                              # Set the prediction horizon used in the MPC optimization problem.
h = 0.1                             # Set the step size of the discretization of the continuous-time dynamics.
num_agents = 4                      # Set number of quadrotors.
load_terminal_ingredients = False   # If False, compute terminal ingredients. If True, load them from a file.
save_terminal_ingredients = False   # If True, save terminal ingredients to a file.
method = 'Euler'                    # Discretisation method: 'Euler', 'RK4', 'RK2'
distance = 0.4                      # Set the minimum distance between quadrotors.       
N_2nd_phase = 30                    # Set the prediction horizon used in the MPC optimization problem for the second phase. 
# ---------------------------------------------------------------------------------------------------------
# Simulation parameters.
# ---------------------------------------------------------------------------------------------------------
max_sim_time = 700                  # Set the maximum simulation time step.
terminal_ingredients_type = 'set'   # Choice between 'set', 'equality', and 'without'.
cutoff_treshold = -1e-6             # Stop the simulation if the value function falls below this threshold.
average_treshold = -1e-6            # Stop the simulation if the standard deviation of the value function falls below this threshold.
max_iter = None                     # Maximum number of iterations for ipopt. None allows ipopt's default.

sqp_max_iter = 10                   # Number of SQP iterations in the first phase.
sqp_max_iter_2nd_phase = 14         # Number of SQP iterations in the second phase.

admm_max_iter = 55                  # Number of ADMM iterations to solve the QP in each SQP iteration in the first phase.
admm_max_iter_2nd_phase = 100       # Number of ADMM iterations to solve the QP in each SQP iteration in the second phase.
admm_penalty = 25                   # Penalty parameter for the ADMM algorithm in the first phase.
admm_penalty_2nd_phase = 25         # Penalty parameter for the ADMM algorithm in the second phase.

solver = 'gurobi'                   # Solver for local QPs, e.g. 'osqp', 'qpoases', 'gurobi', 'ipopt'
parallel = True                     # Whether to use parallelization for the local QPs.

switching_time = 350                # Time step when the switching between the two phases of the cooperative task occurs.
# ---------------------------------------------------------------------------------------------------------
# Cooperative task.
# ---------------------------------------------------------------------------------------------------------
T = 50                              # Periodicity of the cooperative task.
T_2nd_phase = 1                     # Periodicity of the cooperative task in the second phase (should be set to 1).

coop_task = 'circle'

print(f"Cooperative task is '{coop_task}'.")
print(f"Last simulation time is {max_sim_time*h} s (~ {max_sim_time*h // 60} min) with {max_sim_time} simulation steps.")
print(f"Period length is {T*h} s (~ {T*h // 60} min) with {T} simulation steps.")
print(f"Prediction horizon is {N*h} s (~ {N*h // 60} min) with {N} prediction steps.")


# %%
"""Initialise data saving."""
data = {}
data['cooperative_task'] = {}
data['cooperative_task']['type'] = coop_task
data['MAS_parameters'] = {}
data['MAS_parameters']['num_agents'] = num_agents
data['MAS_parameters']['MAS_type'] = MAS_type
data['MAS_parameters']['h'] = h
data['sim_data'] = {'max_sim_time': max_sim_time}
data['sim_pars'] = {'N': N,
                    'cutoff_threshold': cutoff_treshold,
                    'average_treshold': average_treshold,
                    'terminal_ingredients_type': terminal_ingredients_type,
                    'max_iter': max_iter,
                    'T': T,
                    'sqp_max_iter': sqp_max_iter,
                    'admm_max_iter': admm_max_iter, 
                    'admm_penalty': admm_penalty}

# %%
"""Multi-agent system"""
agents = aux.get_quadrotor10_MAS(data)
positions = [
    # np.vstack([ -2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.vstack([ -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.vstack([ -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.vstack([  0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.vstack([  1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.vstack([  2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ]
data['sim_pars']['positions'] = positions
# Define an all-to-all topology.
for i, agent in enumerate(agents):
    agent.neighbours = []
    for j in range(len(agents)):
        if j != i:
            agent.neighbours.append(agents[j])


# %%
"""Cooperative tasks"""
## Define the task to following a circle trajectory with a shifted phase:
phase_shift = 45*np.pi/180  # Set a phase shift between agents.
radius_lb = 1.  # Set a lower bound for the radius.
radius_ub = 2.0   # Set an upper bound for the radius.
radius_softplus_weight = 1.  # Set a weight for the softplus function if the radius is penalized using softplus.
radius_softplus_decay = 100.  # Set a decay for the softplus function, i.e. a multiplicative factor for the exponent of the exponential function.
constrain_radius = True  # Decide whether to use a hard (True) or soft constraint (False) to constrain the radius.  

for agent in agents[1:]:
    agent.cooperation_neighbours = [agents[0]]  # Set the neighbours for the flocking task.
agents[0].cooperation_neighbours = agents[1:] 

def get_leader_reference(t, switching_time):
    rate = switching_time
    
    k = t - switching_time
    
    ref = [0., 0., 0.]
    
    if k <= rate:
        ref[0] = -10. + (k/rate)*20.
        ref[1] = -10. + (k/rate)*20.
    elif k <= 2*rate:
        ref[0] =  10. - ((k - rate)/rate)*20.
        ref[1] =  10. - ((k - rate)/rate)*20.

    return np.vstack(ref)

data['MAS_parameters']['collision_distance'] = distance
coop_kwargs = {
    't': 0, 
    'agents': agents, 
    'T': T, 
    'switching_time': switching_time, 
    'phase_shift': phase_shift, 
    'radius_lb': radius_lb, 
    'radius_ub': radius_ub, 
    'radius_softplus_weight': radius_softplus_weight,
    'radius_softplus_decay': radius_softplus_decay, 
    'constrain_radius': constrain_radius, 
    'get_leader_reference':get_leader_reference,
    'distance': distance}

coop_task_builder = aux.set_cooperative_task_to_circle
if 'cooperative_task' not in data:
    data['cooperative_task'] = {}
data['cooperative_task']['T'] = T
data['cooperative_task']['switching_time'] = switching_time
data['cooperative_task']['phase_shift'] = phase_shift
data['cooperative_task']['radius_lower_bound'] = radius_lb
data['cooperative_task']['radius_upper_bound'] = radius_ub
data['cooperative_task']['radius_constrained'] = constrain_radius
data['cooperative_task']['kwargs'] = coop_kwargs
if not constrain_radius:
    data['cooperative_task']['radius_softplus_weight'] = radius_softplus_weight
    data['cooperative_task']['radius_softplus_decay'] = radius_softplus_decay
    
# Call the task builder to establish constraints.
aux.set_cooperative_task_to_circle(**coop_kwargs)    


# %%
"""Terminal ingredients"""
if load_terminal_ingredients: 
    aux.load_generic_terminal_ingredients(agents[0], './data/quadrotor_terminal_ingredients')
    for agent in agents[1:]:
        agent.terminal_ingredients = agents[0].terminal_ingredients
    print(f'Loaded terminal ingredients with the following set sizes:')
    for agent in agents:
        print(f'A{agent.id}: {agent.terminal_ingredients['size']}')
else:    
    aux.compute_terminal_ingredients_for_quadrotor(
        agent=agents[0], 
        data=data, 
        grid_resolution=1000, 
        num_decrease_samples=1000, 
        alpha = 0.5,
        alpha_tol = 1e-9,
        references_are_equilibria=False,
        compute_size_for_decrease=True,
        compute_size_for_constraints=True,
        epsilon=1e-2,
        verbose=2,
        solver='MOSEK')
    for agent in agents[1:]:
        agent.terminal_ingredients = agents[0].terminal_ingredients
if save_terminal_ingredients:
    aux.save_generic_terminal_ingredients(agents[0], './data/quadrotor_terminal_ingredients')

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
if save_data and save_interval_steps > 0:
    filestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filestamp = f'{filestamp}_quicksave_{coop_task}'

for t in range(0, max_sim_time+1):    
    print(f"{t}: -------------------------------------------------------------------------")  # Print the time step:
    if t == 0:
        warm_start = aux.quadrotor_warm_start_at_t0(agents, N, T)
    elif t == switching_time:
        # Pop outdated multipliers.
        for agent in agents:
            agent.MPC_sol.pop(f'A{agent.id}_ineq_mult')
            agent.MPC_sol.pop(f'A{agent.id}_consensus_mult')
            if N_2nd_phase != N or T_2nd_phase != T:
                agent.MPC_sol.pop(f'A{agent.id}_eq_mult')
            
        # Compute a new warm start.
        warm_start = aux.quadrotor_warm_start_at_switching_time(agents, N_2nd_phase, T_2nd_phase, N, T, terminal_ingredients_type)
                
        # Update the preediction horizon, and the periodicity.
        N = N_2nd_phase
        T = T_2nd_phase
        coop_kwargs['T'] = T
        
        # Update the iteration numbers for the second phase.
        admm_max_iter = admm_max_iter_2nd_phase 
        data['sim_pars']['admm_max_iter_2nd_phase'] = admm_max_iter_2nd_phase
        sqp_max_iter = sqp_max_iter_2nd_phase
        data['sim_pars']['sqp_max_iter_2nd_phase'] = sqp_max_iter_2nd_phase
        admm_penalty = admm_penalty_2nd_phase
        data['sim_pars']['admm_penalty_2nd_phase'] = admm_penalty_2nd_phase

    else:
        warm_start = aux.compute_decentralized_following_warm_start_dynamic_cooperative_DMPC(agents, T, N, terminal_ingredients_type=terminal_ingredients_type)

    coop_kwargs['t'] = t
    coop_kwargs['agents'] = agents
    # Generate and solve the optimization problem for MPC for dynamic cooperation.
    res = aux.solve_MPC_for_dynamic_cooperation_decentrally(
        sqp_max_iter, 
        admm_max_iter, 
        admm_penalty,
        t, 
        agents, 
        N=N, 
        T=T,
        warm_start=warm_start, 
        solver=solver,
        terminal_ingredients_type=terminal_ingredients_type,
        coop_task_builder=coop_task_builder, 
        coop_kwargs=coop_kwargs,
        max_iter=max_iter,
        verbose=2,
        parallel=parallel
    )
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
    if save_data and t > 0 and save_interval_steps > 0 and t % save_interval_steps == 0:
        aux.save_data(data, agents, filestamp)

end_time = time.time()
elapsed = end_time - start_time

print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n")

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
ax_V.set_title(f'Value function over time')
ax_V.legend()
ax_V.grid()

# ax_V.set_yscale('log')  # Set the y-axis to logarithmic scale.

print(f'Value function difference between the first and last time step: {data["sim_data"]["J"][-1] - data["sim_data"]["J"][0]}')
print(f'Value function at start: {data["sim_data"]["J"][0]:15.4e}')
print(f'Value function at stop : {data["sim_data"]["J"][-1]:15.4e}; diff: {data["sim_data"]["J"][-1] - data["sim_data"]["J"][0]:15.4e}')
print(f'Cooperation cost at start : {data["sim_data"]["cooperative_cost"][0]:15.4e}')
print(f'Cooperation cost at stop : {data["sim_data"]["cooperative_cost"][-1]:15.4e}; diff: {data["sim_data"]["cooperative_cost"][-1] - data["sim_data"]["cooperative_cost"][0]:15.4e}')

plt.show()

# %%
"""2D position"""
# Plot from t1 to t2.
t1 = 0
t2 = max_sim_time + 1
step = 1

# Select a feasible start time (the end time is controlled automatically).
t1 = min(t1, max_sim_time+1)

# Draw the evolution in state space:
fig_cl, ax_cl = plt.subplots(figsize=(10, 10), num='state evolution')

for i, agent in enumerate(agents):
    cl_x = agent.cl_x
    ax_cl.plot(cl_x[0, t1 : t2+1:step], cl_x[1,t1 : t2+1:step], color=colours[i], label=f'A{agent.id}_x', 
               #marker='o', markersize=2, 
               linewidth=1.5)
    # Mark the initial state with a larger circle.
    ax_cl.plot(cl_x[0,t1], cl_x[1,t1], color=colours[i], marker='o', markersize=6)
    # Mark the final state with a cross.
    ax_cl.plot(cl_x[0,t2], cl_x[1,t2], color=colours[i], marker='x', markersize=6)
    
    ax_cl.plot(data['sim_data']['yT'][f'A{agent.id}'][-1][0, :], data['sim_data']['yT'][f'A{agent.id}'][-1][1, :], color=colours[i], markersize=2, linewidth=1, marker='o', label=f'A{agent.id}_yT', alpha=0.25)

ax_cl.set_xlabel('$x_1$')    
ax_cl.set_ylabel('$x_2$')
ax_cl.set_title(f'Closed-loop position from $t = {t1}$ to $t = {t2}$ with step {step}')
ax_cl.grid()

plt.show()

# %%
"""Collision"""

# Plot from t1 to t2.
t1 = 0
t2 = max_sim_time + 1

fig_d, ax_d = plt.subplots(figsize=(6, 6))
ax_d.set_xlabel('$t$') 
ax_d.set_ylabel(f'distance')
ax_d.set_title(f'Distance in position from t = {t1} to t = {t2}')
ax_d.grid(True)

considered_pairs = {}
for i, agent in enumerate(agents):
    a1 = agent
    for neighbour in agent.neighbours:
        a2 = neighbour
        if (a2.id, a1.id) in considered_pairs:
            continue
        else:
            considered_pairs[(a1.id, a2.id)] = True

    distances = []
    for t in range(t1, t2+1):
        distances.append(np.linalg.norm(a1.cl_x[0:2, t] - a2.cl_x[0:2, t])) 

    ax_d.plot(range(t1, t2+1), distances, color=colours[i], label=f'||A{a1.id}_x[0:2] - A{a2.id}_x[0:2]||', markersize=0, linewidth=2, marker='o')
ax_d.plot(range(t1, t2+1), [data['MAS_parameters']['collision_distance']]*len(range(t1, t2+1)), color='black', label=f'boundary', linewidth=2, linestyle='--')

plt.show()


