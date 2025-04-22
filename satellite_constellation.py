# %% [markdown]
# # Example: Satellite constellation
# 
# This notebook sets up the simulation for the example in Section IV.B of
# 
# > Distributed Model Predictive Control for Dynamic Cooperation of Multi-Agent Systems --- Matthias Köhler, Matthias A. Müller, and Frank Allgöwer
# 
# If the flag below is set to True, the simulation data is saved to a data file in the folder `./data/`.
# This is recommended if the exported python file is run in order to access the simulation data later.
# 
# The data can be visualised using the accompanying notebook `satellite_constellation_evaluation.ipynb`.
# 
# The simulation data used in the paper is contained in the file `./data/satellite_constellation_data.dill`. 
# This data is animated in `satellite_constellation.mp4`.

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
MAS_type = 'satellites'
N = 3*47                            # Set the prediction horizon used in the MPC optimization problem.
h = 120                             # Set the step size of the discretization of the continuous-time dynamics.
num_agents = 5                      # Set number of agents.
scaling = 'Mm'                      # Scaling for the satellite example, i.e. 'm', 'km', 'Mm'.
method = 'RK4'                      # Discretisation method: 'Euler', 'RK4', 'RK2' (where applicable)
# ---------------------------------------------------------------------------------------------------------
# Simulation parameters.
# ---------------------------------------------------------------------------------------------------------
max_sim_time = 2000                     # Set the last simulation time step.
terminal_ingredients_type = 'equality'  # Choose terminal equality constraints.
cutoff_treshold = -1e-6                 # Stop the simulation if the value function falls below this threshold.
average_treshold = -1e-6                # Stop the simulation if the standard deviation of the value function falls below this threshold.

max_iter = None                         # Maximum number of iterations for ipopt. None allows ipopt's default.

sqp_max_iter = 5                        # Number of SQP iterations.
admm_max_iter = 40                      # Number of ADMM iterations to solve the QP in each SQP iteration.
admm_penalty = 2                        # Penalty parameter for ADMM.

solver = 'gurobi'                       # Solver for local QPs, e.g. 'osqp', 'qpoases', 'gurobi', 'ipopt'
parallel = True                         # Whether to use parallelization for the local QPs.
# ---------------------------------------------------------------------------------------------------------
# Cooperative task.
# ---------------------------------------------------------------------------------------------------------
T = 47                                  # Set the period length of the cooperative task.
theta_des = 45                          # Desired angle for the constellation task in degrees.     
deorbit_time = 750                      # Time step at which two satellites are removed from the constellation.

coop_task = 'constellation'

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
                    'admm_penalty': admm_penalty,
                    'solver': solver}

# %%
"""Multi-agent system"""
agents = aux.get_satellites_MAS(data, scaling, method)
# Define the initial conditions.
ics = []
sf = data['MAS_parameters']['scaling_factor']
for i, agent in enumerate(agents):
    ics.append(np.array([[0.0], [np.radians(i*25)], [0.], [np.sqrt(agent.mu/(agent.r0)**3)]]))
# Set the initial condition.
for idx, agent in enumerate(agents):
    agent.current_state = ics[idx]
    print(f'Orbital radius for agent {agent.id}: {agent.r0 + agent.current_state[0,0]:.6e} {agent.scaling} (r0 + {agent.current_state[0,0]:.1e} {agent.scaling}) with periodicity {T}.')


# %%
"""Cooperative tasks"""
## Define the task of agreeing on an orbit and having a fixed distance between agents.

if len(agents) > len(ics):
    raise ValueError(f'The number of agents ({len(agents)}) exceeds the number of initial positions ({len(ics)}) of the {coop_task} task!')

data['positions'] = ics
    
coop_task_builder = aux.set_cooperative_task_to_constellation
deorbited_satellites = []
coop_kwargs={'t': 0, 'agents': agents, 'weight': 0.5, 'N': N, 'T': T, 'theta_des': theta_des, 'sf': data['MAS_parameters']['scaling_factor']}
data['cooperative_task']['T'] = T
data['cooperative_task']['kwargs'] = coop_kwargs
data['cooperative_task']['theta_des'] = theta_des

# Call the task builder to establish constraints.
aux.set_cooperative_task_to_constellation(**coop_kwargs)

# %%
"""Simulation run"""
# Initalization:
for idx, agent in enumerate(agents):
    agent.current_state = ics[idx]

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
        warm_start = aux.satellite_warm_start_at_t0(agents, N, T)
    else:
        warm_start = aux.compute_decentralized_following_warm_start_dynamic_cooperative_DMPC(agents, T, N, terminal_ingredients_type=terminal_ingredients_type)
            
    coop_kwargs['t'] = t
    coop_kwargs['agents'] = agents
    # Generate and solve the optimization problem for MPC for dynamic cooperation.
    res = aux.solve_MPC_for_dynamic_cooperation_decentrally(sqp_max_iter, admm_max_iter, admm_penalty,
            t, agents, N=N, T=T, feas_tol=1e-8, 
            warm_start=warm_start, 
            solver=solver,
            terminal_ingredients_type=terminal_ingredients_type,
            coop_task_builder=coop_task_builder, 
            coop_kwargs=coop_kwargs,
            max_iter=max_iter,
            verbose=2,
            parallel=parallel)
    print(f'Solved at time step {t} with f* = {float(res["J"]):.5e}')
    
    data['sim_data']['tracking_cost'].append(res['tracking_cost'])
    data['sim_data']['cooperative_cost'].append(res['cooperative_cost'])
    data['sim_data']['change_cost'].append(res['change_cost']) 
    data['sim_data']['J'].append(res['J']) 
    
    for agent in agents:
        # Keep track of open-loop solutions, reshaped ready for plotting.
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
        # For satellite agents, the second state (theta; angular position) wraps around 2pi and needs to be adjusted.
        # Here, shifting is not enough, since the agent's state increments theta and does not consider the modulo behaviour.
        # Hence, theta needs to be increased by 2pi when shifted.
        if isinstance(agent, aux.Satellite) and agent.state_dim == 4:
            agent.yT_pre[-2] = agent.yT_pre[-2] + 2*np.pi
        
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
        
    # In the constellation cooperative task, deorbit some satellites after some time.
    if coop_task == 'constellation' and (t == deorbit_time or res['cooperative_cost'] <= 1e-9) and not deorbited_satellites:
        # Save the data.
        aux.save_data(data, agents, filestamp + f'_before_deorbit_at_t_{t}') 
        # Remove some agents from the agents list.
        deorbited_satellites.append(agents.pop(1))
        deorbited_satellites.append(agents.pop(2))
        # Also update the neighbours.
        agents[0].neighbours = [agents[1]]
        for i, agent in enumerate(agents[1:-1]):
            i += 1
            agent.neighbours = [agents[i-1], agents[i+1]]
        agents[-1].neighbours = [agents[-2]]
        # Update the cooperative task.
        coop_task_builder(**coop_kwargs)
    
# In the constellation task, add the deorbited satellites again to the agents list.
if coop_task == 'constellation':
    for sat in deorbited_satellites:
        agents.append(sat)

end_time = time.time()
elapsed = end_time - start_time
print(f"Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

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

#ax_V.set_yscale('log')  # Set the y-axis to logarithmic scale.

print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n")

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
t2 = agent.cl_x.shape[1]-1
step = 1

# Select a feasible start time (the end time is controlled automatically).
t1 = min(t1, max_sim_time+1)

# Draw the evolution in state space:
fig_cl, ax_cl = plt.subplots(figsize=(10, 10), num='state evolution')

for i, agent in enumerate(agents):
    cl_x = np.zeros(agent.cl_x.shape)
    # For the constellation task, transform the polar coordinates into Cartesian coordinates.
    if coop_task == 'constellation':
        cl_x[0, t1 : t2+1:step] = (agent.cl_x[0, t1 : t2+1:step] + agent.r0) * np.cos(agent.cl_x[1, t1 : t2+1:step])
        cl_x[1, t1 : t2+1:step] = (agent.cl_x[0, t1 : t2+1:step] + agent.r0) * np.sin(agent.cl_x[1, t1 : t2+1:step])
    else:
        cl_x = agent.cl_x
    ax_cl.plot(cl_x[0, t1 : t2+1:step], cl_x[1,t1 : t2+1:step], color=colours[i], label=f'A{agent.id}_x', 
               #marker='o', markersize=2, 
               linewidth=1.5)
    # Mark the initial state with a larger circle.
    ax_cl.plot(cl_x[0,t1], cl_x[1,t1], color=colours[i], marker='o', markersize=6)
    # Mark the final state with a cross.
    ax_cl.plot(cl_x[0,t2], cl_x[1,t2], color=colours[i], marker='x', markersize=6)

if coop_task == 'constellation':
    xlabel=f'$x_1$ ({scaling})'
    ylabel=f'$x_2$ ({scaling})'
else:
    xlabel='$x_1$'
    ylabel='$x_2$'
    
if coop_task == 'constellation':
    sf = data['MAS_parameters']['scaling_factor']
    r0 = data['MAS_parameters']['r0']
    r_max = (data['MAS_parameters']['r_max'] + r0)*sf
    ax_cl.set_xlim(-r_max, r_max)
    ax_cl.set_ylim(-r_max, r_max)
    # Plot Earth
    circle = patches.Circle((0, 0), 6371e3*sf, facecolor='#4169E1', fill=True, edgecolor='#4169E1', linewidth=2) 
    ax_cl.add_patch(circle)

ax_cl.set_xlabel(xlabel)
ax_cl.set_ylabel(ylabel)
ax_cl.set_title(f'Closed-loop position from $t = {t1}$ to $t = {t2}$ with step {step}')
ax_cl.grid()
ax_cl.legend()

plt.show()


