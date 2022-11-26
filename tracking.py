import RocketLib as rl
import numpy as np
import matplotlib.pylab as plt


optimal_inputs = np.loadtxt('optimal_input.csv', delimiter=',')
optimal_states = np.loadtxt('optimal_state.csv', delimiter=',')
optimal_timestep = 0.2
optimal_time = np.arange(optimal_states.shape[0]) * optimal_timestep

# rl.plot_trajectory_matlab_example(optimal_states, optimal_inputs, optimal_time)
# plt.show()

n = 200
optimal_states = rl.interpolate_data(optimal_states, n)
optimal_inputs = rl.interpolate_data(optimal_inputs, n)
rl.create_animation('test.gif', optimal_states, optimal_inputs)
