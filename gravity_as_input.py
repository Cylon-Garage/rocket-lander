import RocketLib as rl
import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
from Parameters import *
np.set_printoptions(precision=3)

id = 2
n_states, n_inputs = 6, 3
n_steps = 150
epsilon = 0.01

n = 10
x_0 = np.array(get_x0(id))
u_0 = np.array([0.0, 0.0, G])
gravity_as_input = False
gravity_as_input = True

x = x_0
u = u_0
x_exact = x_0
A, B = rl.get_linearized_state(
    x, u, TIMESTEP, epsilon, rl.dragon_state_function, gravity_as_input)

states = []
states_exact = []
for i in range(n):
    x = A @ x + B @ u
    x_exact = solve_ivp(rl.dragon_state_function,
                        t_span=[0, TIMESTEP],
                        y0=x_exact,
                        args=(u, False)).y.T[-1, :]

    states.append(x)
    states_exact.append(x_exact)

states = np.around(np.array(states), decimals=6)
states_exact = np.array(states_exact)
time = np.arange(n) * TIMESTEP


fig, ax = plt.subplots()
ax.plot(states[:, 0], states[:, 1], 'g',
        marker='.', label='linearized solution')
ax.plot(states_exact[:, 0], states_exact[:, 1],
        'b', marker='.', linestyle='--', label='exact solution')
ax.grid()
ax.legend()
ax.set_xlabel('x displacement [m]')
ax.set_ylabel('y displacement [m]')
# plt.show()
fig.tight_layout()
fig.savefig('gravity_as_input_%s.png' % str(gravity_as_input), dpi=80)

# print(str(gravity_as_input))
