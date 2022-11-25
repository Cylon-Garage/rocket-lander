import RocketLib as rl
import casadi as cs
import numpy as np
import matplotlib.pylab as plt


timestep = 0.2
n_horizon = 10
n_steps = 100
start_position = (-30, 60)
ref_state_fname = 'optimal_state.npy'
ref_input_fname = 'optimal_input.npy'


model = rl.get_model(tracking=True)
mpc = rl.get_mpc_optimizer(model, timestep, n_horizon)


optimal_trajectory = np.load(ref_state_fname)

pos_error = (model.x['pos'][0] - model.tvp['optimal_pos'][0]) ** 2 + \
            (model.x['pos'][1] - model.tvp['optimal_pos'][1]) ** 2
dpos_error = (model.x['dpos'][0] - model.tvp['optimal_dpos'][0]) ** 2 + \
             (model.x['dpos'][1] - model.tvp['optimal_dpos'][1]) ** 2

theta_error = (model.x['theta'] - model.tvp['optimal_theta']) ** 2
dtheta_error = (model.x['dtheta'] - model.tvp['optimal_dtheta']) ** 2
running_cost = pos_error + dpos_error + theta_error + dtheta_error
terminal_cost = running_cost
mpc.set_objective(mterm=terminal_cost, lterm=running_cost)
mpc.set_rterm(thrust=1)

optimal_trajectory = np.load(ref_state_fname)
