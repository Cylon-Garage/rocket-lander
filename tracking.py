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
n_horizon = 10
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


def stage_cost(x, y, theta, dx, dy, dtheta, z_ref, u, u_prev, k):
    du = [u[0] - u_prev[0], u[1] - u_prev[1]]
    cost = R * du[0] ** 2 + du[1] ** 2
    if k > 0:
        cost += Q*(x-z_ref[0])**2
        cost += Q*(y-z_ref[1])**2
        cost += Q*(theta-z_ref[2])**2
        cost += Q*(dx-z_ref[3])**2
        cost += Q*(dy-z_ref[4])**2
        cost += Q*(dtheta-z_ref[5])**2
    return cost


def terminal_cost(x, y, theta, dx, dy, dtheta, z_ref):
    cost = Qf*(x-z_ref[0])**2
    cost += Qf*(y-z_ref[1])**2
    cost += Qf*(theta-z_ref[2])**2
    cost += Qf*(dx-z_ref[3])**2
    cost += Qf*(dy-z_ref[4])**2
    cost += Qf*(dtheta-z_ref[5])**2
    return cost


u = cs.SX.sym('u', n_inputs * n_horizon)
z_0 = cs.SX.sym('z_0', n_states)
z_ref = cs.SX.sym('z_ref', n_states * n_horizon)
x, y, theta, dx, dy, dtheta = z_0[0], z_0[1], z_0[2], z_0[3], z_0[4], z_0[5]

cost = 0
u_prev = [0, 0]
for k in range(0, n_inputs * n_horizon, n_inputs):
    u_k = u[k:(k+n_inputs)]
    z_ref_k = z_ref[k:(k+n_states)]
    cost += stage_cost(x, y, theta, dx, dy, dtheta, z_ref_k, u_k, u_prev, k)
    x, y, theta, dx, dy, dtheta = dynamics_dt(x, y, theta, dx, dy, dtheta, u_k)
    u_prev = u_k

cost += terminal_cost(x, y, theta, dx, dy, dtheta, z_ref[-n_states:])

umin = [0.0] * (n_inputs*n_horizon)
umax = [8.0] * (n_inputs*n_horizon)
bounds = og.constraints.Rectangle(umin, umax)

problem = og.builder.Problem(u, cs.vcat(
    [z_0, z_ref]), cost).with_constraints(bounds)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("my_optimizers")\
    .with_build_mode("debug")\
    .with_tcp_interface_config()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("navigation")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-5)
builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config)
builder.build()


def get_reference_trajectory(state):
    reference_distance = np.linalg.norm(
        optimal_trajectory[:, :2] - state[:2], axis=1)
    nearest_index = np.argmin(reference_distance)
    idx = np.arange(nearest_index, nearest_index + n_horizon)
    idx[idx > (optimal_steps - 1)] = optimal_steps - 1
    return optimal_trajectory[idx, :]


mng = og.tcp.OptimizerTcpManager('my_optimizers/navigation')
mng.start()

init_state = [-30, 60, 0, 0, 0, 0]
state = init_state
states = [state]
inputs = []
for k in range(n_steps):
    state_ref = get_reference_trajectory(state).ravel().tolist()
    param = state + state_ref
    solution = mng.call(param, initial_guess=None)
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
