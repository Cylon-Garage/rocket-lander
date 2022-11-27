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

Q = cs.DM.eye(n_states) * [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
R = cs.DM.eye(n_inputs) * [0.1, 0.1]
Qf = cs.DM.eye(n_states) * [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
x_f = np.array([[0, 10, 0, 0, 0, 0]]).T

sampling_time = 0.2
n_steps = 100
n_horizon = 10

optimal_trajectory = np.load('optimal_state.npy')
optimal_steps = optimal_trajectory.shape[0]


def get_reference_trajectory(state):
    reference_distance = np.linalg.norm(
        optimal_trajectory[:, :2] - state[:2], axis=1)
    nearest_index = np.argmin(reference_distance)
    idx = np.arange(nearest_index, nearest_index + n_horizon)
    idx[idx > (optimal_steps - 1)] = optimal_steps - 1
    return optimal_trajectory[idx, :]


def dynamics_ct(x, u):
    t_fwd = u[0, 0] + u[1, 0]
    t_twist = u[1, 0] - u[0, 0]
    dx0 = x[3]
    dx1 = x[4]
    dx2 = x[5]
    dx3 = -t_fwd / m * cs.sin(x[2])
    dx4 = -g + t_fwd / m * cs.cos(x[2])
    dx5 = L2 / I * t_twist
    return [dx0, dx1, dx2, dx3, dx4, dx5]


def dynamics_dt(x, u):
    dx = dynamics_ct(x, u)
    # return cs.vcat([x[i] + sampling_time * dx[i] for i in range(n_states)])
    return np.array([x[i] + sampling_time * dx[i] for i in range(n_states)])


def stage_cost(x, u):
    return u[0] + u[1]


x_init = np.array([-30, 60, 0, 0, 0, 0])
u = np.array([[3, 1]]).T
x = x_init
states = np.zeros((n_steps + 1, n_states))
states[0, :] = x.squeeze()
for i in range(n_steps):
    x = dynamics_dt(x, u)
    states[i + 1, :] = x.squeeze()

# plt.show()

matlab = np.loadtxt('states.csv', delimiter=',').reshape((-1, 6))
plt.plot(matlab[:, 0], matlab[:, 1])
plt.plot(states[:, 0], states[:, 1])
plt.show()

# u_seq = cs.MX.sym('u', n_inputs, n_horizon)
# x_0 = cs.MX.sym('x_0', n_states)

# x_t = x_0
# total_cost = 0
# constraints = []
# for t in range(n_horizon):
#     total_cost += stage_cost(x_t, u_seq[:, t])
#     x_t = dynamics_dt(x_t, u_seq[:, t])
#     constraints.append(cs.fmin(x_t[1] - 10, 0))
# constraints.append(cs.fmin(x_t - x_f, 0))

# input_bounds = og.constraints.Rectangle([0] * n_horizon, [8] * n_horizon)
# constraints = cs.vertcat(*constraints)

# problem = og.builder.Problem(
#     u_seq, x_0, total_cost).with_penalty_constraints(constraints) \
#     .with_constraints(input_bounds)

# build_cfg = og.config.BuildConfiguration()\
#     .with_build_directory('python_build')\
#     .with_build_mode('debug')\
#     .with_tcp_interface_config()

# meta = og.config.OptimizerMeta()\
#     .with_optimizer_name('rocket_tracking')

# solver_cfg = og.config.SolverConfiguration()\
#     .with_tolerance(1e-5)

# builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_cfg, solver_cfg)
# builder.build()

# x_init = np.array([-30, 60, 0, 0, 0, 0])
# x = x_init
# x_ref = get_reference_trajectory(x)[0, :]
# param = np.hstack([x, x_ref]).tolist()
# mng = og.tcp.OptimizerTcpManager('python_build/rocket_tracking')
# mng.start()

# param = [-30, 60, 0, 0, 0, 0]
# # param = [0, 10, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0]
# solution = mng.call(param, initial_guess=None)
# print(solution)
