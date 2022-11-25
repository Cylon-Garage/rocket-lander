import numpy as np
from scipy.integrate import solve_ivp
from RocketLib import state_function, get_linearized_state
import matplotlib.pylab as plt


sample_time = 0.2
epsilon = 1/50
epsilon = 0.01
g = 9.80665
x0 = np.array([-30, 60, 0, 0, 0, 0], dtype=float)
u = np.array([3, 1, g], dtype=float)
n = 10

x = x0
x_exact = x0
states = []
states_exact = []
A, B = get_linearized_state(x, u, sample_time, epsilon)
for i in range(n):
    x = A @ x + B @ u
    x_exact = solve_ivp(state_function,
                        t_span=[0, sample_time],
                        y0=x_exact,
                        args=(u,)).y.T[-1, :]
    states.append(x)
    states_exact.append(x_exact)

states = np.around(np.array(states), decimals=6)
states_exact = np.array(states_exact)
time = np.arange(n) * sample_time

fig, ax = plt.subplots(4, 1)
ax[0].plot(states_exact[:, 0], states_exact[:, 1], 'b')
ax[0].plot(states[:, 0], states[:, 1], 'r', linestyle='--', marker='.')

ax[1].plot(time, states_exact[:, 2], 'b')
ax[1].plot(time, states[:, 2], 'r', linestyle='--', marker='.')

ax[2].plot(states_exact[:, 3], states_exact[:, 4], 'b')
ax[2].plot(states[:, 3], states[:, 4], 'r', linestyle='--', marker='.')

ax[3].plot(time, states_exact[:, 5], 'b')
ax[3].plot(time, states[:, 5], 'r', linestyle='--', marker='.')
plt.show()
