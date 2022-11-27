import RocketLib as rl
import numpy as np
import cvxpy as cp
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
np.set_printoptions(precision=3)

optimal_inputs = np.load('optimal_input.npy')
optimal_states = np.load('optimal_state.npy')
optimal_steps = optimal_states.shape[0]
n_states, n_inputs = 6, 3

timestep = 0.2
n_steps = 150
n_horizon = 10
epsilon = 0.01

G = 9.80665
MAX_THRUST = 8

x_0 = np.array([-30, 60, 0, 0, 0, 0], dtype=float)
x_0 = np.array([-30, 60, np.pi/4, -10, 10, 0], dtype=float)
u_0 = np.array([0, 0, G], dtype=float)

Q = np.diag([1, 1, 1, 1, 1, 1])
P = np.diag([1, 1, 1, 1, 1, 1])
R = np.diag([0.1, 0.1, 0])


def mpc(x_0: np.ndarray, u_last: np.ndarray, A: np.ndarray, B: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    x = cp.Variable((n_states, n_horizon + 1))
    u = cp.Variable((n_inputs, n_horizon))

    cost = 0
    constraints = [x[:, 0] == x_0.squeeze()]
    for t in range(n_horizon):
        du = u_last - u[:, t]

        cost += cp.quad_form(x[:, t] - x_ref[t, :], Q) + \
            cp.quad_form(du, R)
        constraints += [
            x[:, t + 1] == A @ x[:, t] + B @ u[:, t],
            x[1] >= rl.L1,

            u[[0, 1], t] >= 0,
            u[[0, 1], t] <= MAX_THRUST,
            u[2, t] == G,
        ]
    constraints += [
        x[1] >= rl.L1,
    ]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    j_star = problem.solve()
    x_star = x.value
    u_star = u.value
    return u_star


def get_reference_state(position: np.ndarray) -> np.ndarray:
    nearest_idx = np.argmin(np.linalg.norm(
        optimal_states[:, :2] - position, axis=1))
    idx = np.arange(nearest_idx, nearest_idx + n_horizon)
    idx[idx > (optimal_steps - 1)] = optimal_steps - 1
    return optimal_states[idx, :]


def run_tracker(rerun=False):
    state_fname = 'tracker_state.npy'
    input_fname = 'tracker_input.npy'

    if not rerun:
        states = np.load(state_fname)
        inputs = np.load(input_fname)
    else:
        x = x_0
        u = u_0
        inputs = [u]
        states = [x]
        for t in range(n_steps - 1):
            x_ref = get_reference_state(x[:2])
            A, B = rl.get_linearized_state(
                x, u, timestep, epsilon, rl.matlab_example_state_function)

            u_optimal = mpc(x, u, A, B, x_ref)
            if u_optimal is not None:
                u = u_optimal[:, 0]
            else:
                print('$$$$', t)
            x = solve_ivp(rl.matlab_example_state_function, t_span=[
                0, timestep], y0=x, args=(u,)).y[:, -1]
            states.append(x)
            inputs.append(u)
        states = np.array(states)
        inputs = np.array(inputs)

        with open(state_fname, 'wb') as f:
            np.save(f, states)
        with open(input_fname, 'wb') as f:
            np.save(f, inputs)
    return states, inputs


rerun = False
# rerun = True
states, inputs = run_tracker(rerun)


#         time = np.arange(states.shape[0]) * timestep


# fig, ax = rl.plot_trajectory_matlab_example(states, inputs, time)
# time = np.arange(optimal_states.shape[0]) * timestep
# linestyle = '--'
# ax[0].plot(optimal_states[:, 0], optimal_states[:, 1], linestyle=linestyle)
# ax[1].plot(time, np.degrees(optimal_states[:, 2]), linestyle=linestyle)
# ax[2].plot(time, optimal_inputs[:, 0], linestyle=linestyle)
# ax[2].plot(time, optimal_inputs[:, 1], linestyle=linestyle)
# plt.show()
# print(states)

n = 300
optimal_states = rl.interpolate_data(optimal_states, n)
optimal_inputs = rl.interpolate_data(optimal_inputs, n)
states = rl.interpolate_data(states, n)
inputs = rl.interpolate_data(inputs, n)
rl.create_animation_matlab_example('tracker.gif', states)
rl.create_tracking_animation('tracker_vs_planner.gif', states, optimal_states)
