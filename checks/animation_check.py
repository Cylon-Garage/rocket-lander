import RocketLib as rl
import numpy as np
import matplotlib.pylab as plt


# optimal_inputs = np.loadtxt('optimal_input2.csv', delimiter=',')
# optimal_states = np.loadtxt('optimal_state2.csv', delimiter=',')
# optimal_timestep = 0.2
# optimal_time = np.arange(optimal_states.shape[0]) * optimal_timestep

# rl.plot_trajectory(optimal_states, optimal_inputs, optimal_time)
# plt.show()
# # quit()

# n = 200
# optimal_states = rl.interpolate_data(optimal_states, n)
# optimal_inputs = rl.interpolate_data(optimal_inputs, n)
# rl.create_animation('test.gif', optimal_states, optimal_inputs)


# optimal_inputs = np.loadtxt('optimal_input0.csv', delimiter=',')
# optimal_states = np.loadtxt('optimal_state0.csv', delimiter=',')
optimal_inputs = np.load('optimal_input.npy')
optimal_states = np.load('optimal_state.npy')
optimal_timestep = 0.2
optimal_time = np.arange(optimal_states.shape[0]) * optimal_timestep

rl.plot_trajectory_matlab_example(optimal_states, optimal_inputs, optimal_time)
plt.show()
# quit()

n = 200
optimal_states = rl.interpolate_data(optimal_states, n)
optimal_inputs = rl.interpolate_data(optimal_inputs, n)
rl.create_animation_matlab_example('test.gif', optimal_states)
