import RocketLib as rl
import casadi as cs
import numpy as np
import matplotlib.pylab as plt


timestep = 0.2
n_horizon = 50
n_steps = 100
path_n_horizon = int(n_steps * .5)
start_position = (-30, 60)
state_fname = 'optimal_state.npy'
input_fname = 'optimal_input.npy'

model = rl.get_model()
mpc = rl.get_mpc_optimizer(model, timestep, n_horizon)

running_cost = model.u['thrust'][0] + model.u['thrust'][1]
terminal_cost = cs.DM(0)
mpc.set_objective(mterm=terminal_cost, lterm=running_cost)
mpc.set_rterm(thrust=0.1)

rl.set_mpc_bounds(mpc)
mpc.setup()
mpc.set_initial_guess()

simulator, estimator = rl.get_simulator(model)
simulator.setup()
simulator.x0['pos'] = start_position
x0 = simulator.x0.cat.full()
estimator.x0 = x0
mpc.x0 = x0

simulator.reset_history()
simulator.x0 = x0
for _ in range(n_steps):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)
state = mpc.data['_x']
input = mpc.data['_u']

with open(state_fname, 'wb') as f:
    np.save(f, state)
with open(input_fname, 'wb') as f:
    np.save(f, input)

time = np.arange(n_steps) * timestep
fig, ax = rl.plot_trajectory(state, input, time)
plt.show()

fig, ax = rl.create_animation('rocket.gif', state, n_steps, True)
