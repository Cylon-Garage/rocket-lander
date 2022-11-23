import casadi as cs
import opengen as og
import matplotlib.pylab as plt
import numpy as np

m = 1
L1 = 10
L2 = 5
I = 1 / 2 * m * (L1 ** 2)
g = 9.80665
n_states = 6
n_inputs = 2
n_horizon = 50
n_steps = 100
sampling_time = 0.2
Q, R, Qf = 1, 0.1, 1

optimal_trajectory = np.load('optimal_state.npy')
optimal_steps = optimal_trajectory.shape[0]


def dynamics_ct(x, y, theta, dx, dy, dtheta, u):
    t_fwd = u[0] + u[1]
    t_twist = u[1] - u[0]
    dx = dx
    dy = dy
    dtheta = dtheta
    d2x = -t_fwd / m * cs.sin(theta)
    d2y = -g + t_fwd / m * cs.cos(theta)
    d2theta = L2 / I * t_twist
    return dx, dy, dtheta, d2x, d2y, d2theta


def dynamics_dt(x, y, theta, dx, dy, dtheta, u):
    dx, dy, dtheta, d2x, d2y, d2theta = dynamics_ct(
        x, y, theta, dx, dy, dtheta, u)

    x = x + sampling_time * dx
    y = y + sampling_time * dy
    theta = theta + sampling_time * dtheta
    dx = dx + sampling_time * d2x
    dy = dy + sampling_time * d2y
    dtheta = dtheta + sampling_time * d2theta

    return x, y, theta, dx, dy, dtheta


u = cs.SX.sym('u', n_inputs * n_horizon)
z_0 = cs.SX.sym('z_0', n_states)
x, y, theta, dx, dy, dtheta = z_0[0], z_0[1], z_0[2], z_0[3], z_0[4], z_0[5]

cost = 0
u_prev = [0, 0]
constraints = []
for k in range(0, n_inputs * n_horizon, n_inputs):
    u_k = u[k:(k+n_inputs)]
    cost += u_k[0] + u_k[1]
    constraints.append(cs.fmin(y - 10, 0))
    x, y, theta, dx, dy, dtheta = dynamics_dt(x, y, theta, dx, dy, dtheta, u_k)
constraints.append(cs.fmin(x, 0))
constraints.append(cs.fmax(x, 0))
constraints.append(cs.fmin(dx, 0))
constraints.append(cs.fmax(dx, 0))
constraints.append(cs.fmin(y - 10, 0))
constraints.append(cs.fmax(y - 10, 0))
constraints.append(cs.fmin(dy, 0))
constraints.append(cs.fmax(dy, 0))
constraints.append(cs.fmin(theta, 0))
constraints.append(cs.fmax(theta, 0))
constraints.append(cs.fmin(dtheta, 0))
constraints.append(cs.fmax(dtheta, 0))
constraints = cs.vertcat(*constraints)

umin = [0.0] * (n_inputs*n_horizon)
umax = [8.0] * (n_inputs*n_horizon)
bounds = og.constraints.Rectangle(umin, umax)

problem = og.builder.Problem(u, z_0, cost) \
    .with_penalty_constraints(constraints).with_constraints(bounds)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("my_optimizers")\
    .with_build_mode("debug")\
    .with_tcp_interface_config()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("planning")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-5)
builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config)
# builder.build()


mng = og.tcp.OptimizerTcpManager('my_optimizers/planning')
mng.start()
# init_state = [-30, 60, 0, 0, 0, 0]
# state = init_state
# solution = mng.call(state, initial_guess=None)
# u = np.array(solution['solution']).reshape((-1, 2))
# print(u)

init_state = [-30, 60, 0, 0, 0, 0]
state = init_state
states = [state]
inputs = []
for k in range(n_steps):
    solution = mng.call(state, initial_guess=None)
    u = np.array(solution['solution']).reshape((-1, 2))[0, :]
    state = list(dynamics_dt(*state, u))
    states.append(state)
    inputs.append(u)


mng.kill()

states = np.array(states)
inputs = np.array(inputs)
time = np.arange(n_steps) * sampling_time
print(states[-5:, :])

fig, ax = plt.subplots(2, 1)

ax[0].plot(states[:, 0], states[:, 1])
ax[1].plot(time, inputs[:, 0])
ax[1].plot(time, inputs[:, 1])

# ax[0].set_xlim(-40, 40)
# ax[0].set_ylim(0, 80)
ax[1].set_ylim(0, 8)
plt.show()
