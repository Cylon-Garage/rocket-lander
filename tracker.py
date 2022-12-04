import os
import shutil
import RocketLib as rl
import numpy as np
import cvxpy as cp
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
from Parameters import *
import utils
np.set_printoptions(precision=3)

id = 2
n_states, n_inputs = 6, 3
n_steps = 150
epsilon = 0.01

x_0 = np.array(get_x0(id))
u_0 = np.array([0.0, 0.0, G])

Q = np.diag([1, 1, 1, 1, 1, 1])
P = np.diag([1, 1, 1, 1, 1, 1])
Q = np.diag([1, 1, 50, 1, 1, 10])
P = np.diag([1, 1, 50, 1, 1, 10])
R = np.diag([0.1, 0.1, 0])

optimal_inputs = np.load('saved_states/optimal_input%u.npy' % id)
optimal_states = np.load('saved_states/optimal_state%u.npy' % id)
optimal_steps = optimal_states.shape[0]


def mpc(x_0: np.ndarray, u_last: np.ndarray, A: np.ndarray, B: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    x = cp.Variable((n_states, TRACKER_N_HORIZON + 1))
    u = cp.Variable((n_inputs, TRACKER_N_HORIZON))

    cost = 0
    constraints = [x[:, 0] == x_0.squeeze()]
    for t in range(TRACKER_N_HORIZON):
        du = u_last - u[:, t]

        cost += cp.quad_form(x[:, t] - x_ref[t, :], Q) + \
            cp.quad_form(du, R)
        constraints += [
            x[:, t + 1] == A @ x[:, t] + B @ u[:, t],
            x[1] >= DRAGON_CG_HEIGHT,

            u[[0, 1], t] >= 0,
            u[[0, 1], t] <= MAX_THRUST,
            u[2, t] == G,
        ]
    constraints += [
        x[1] >= DRAGON_CG_HEIGHT,
    ]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    j_star = problem.solve()
    x_star = x.value
    u_star = u.value
    return u_star


def get_reference_state(position: np.ndarray) -> np.ndarray:
    nearest_idx = np.argmin(np.linalg.norm(
        optimal_states[:, :2] - position, axis=1))
    idx = np.arange(nearest_idx, nearest_idx + TRACKER_N_HORIZON)
    idx[idx > (optimal_steps - 1)] = optimal_steps - 1
    return optimal_states[idx, :]


def run_tracker(id: int, rerun=False):
    state_fname = 'saved_states/tracker_state%u.npy' % id
    input_fname = 'saved_states/tracker_input%u.npy' % id

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
                x, u, TIMESTEP, epsilon, rl.dragon_state_function)

            u_optimal = mpc(x, u, A, B, x_ref)
            if u_optimal is not None:
                u = u_optimal[:, 0]
            else:
                print('*' * 25)
                print('optimization failed at %u timestep' % t)
                print('*' * 25)
                quit()
            x = solve_ivp(rl.dragon_state_function, t_span=[
                0, TIMESTEP], y0=x, args=(u,)).y[:, -1]
            states.append(x)
            inputs.append(u)
        states = np.array(states)
        inputs = np.array(inputs)

        with open(state_fname, 'wb') as f:
            np.save(f, states)
        with open(input_fname, 'wb') as f:
            np.save(f, inputs)
    return states, inputs


def plot_frames(
    id: int,
    states: np.ndarray,
    inputs: np.ndarray,
    time: np.ndarray,
    optimal_states: np.ndarray,
    optimal_inputs: np.ndarray,
    optimal_time: np.ndarray
):
    if os.path.isdir('plot_frames/%u' % id):
        shutil.rmtree('plot_frames/%u' % id)

    os.mkdir('plot_frames/%u' % id)

    for i in range(states.shape[0]):
        styles = {
            'linestyle': '-',
            'c1': 'g',
            'c2': 'g',
            'c3': 'g',
        }
        fig, ax = rl.plot(states[:i, :], inputs[:i, :],
                          time[:i], label='tracker', styles=styles, end_marker=True)

        styles = {
            'linestyle': '-',
            'c1': 'b',
            'c2': 'b',
            'c3': 'b',
        }
        fig, ax = rl.plot(optimal_states[:i, :], optimal_inputs[:i, :],
                          optimal_time[:i], styles, ax, label='optimal path', end_marker=True)

        ax[0].legend()
        ax[0].set_ylim([-60, 60])
        ax[1].set_ylim([-1, MAX_THRUST + 1])
        ax[2].set_ylim([-1, MAX_THRUST + 1])
        ax[0].set_xlim([0, np.ceil(time.max())])
        ax[1].set_xlim([0, np.ceil(time.max())])
        ax[2].set_xlim([0, np.ceil(time.max())])

        fig.tight_layout()
        fig.set_edgecolor('k')
        fig.set_linewidth(5)
        fig.savefig('plot_frames/%u/tracker_vs_optimal_%04u.png' %
                    (id, i), dpi=80)
        plt.close()


rerun = False
# rerun = True
styles = {
    'linestyle': '-',
    'c1': 'g',
    'c2': 'g',
    'c3': 'g',
}
states, inputs = run_tracker(id, rerun)
time = np.arange(states.shape[0]) * TIMESTEP
optimal_time = np.arange(optimal_states.shape[0]) * TIMESTEP


# fig, ax = rl.plot(states, inputs, time, label='tracker', styles=styles)
# styles = {
#     'linestyle': '--',
#     'c1': 'b',
#     'c2': 'b',
#     'c3': 'b',
# }
# fig, ax = rl.plot(optimal_states, optimal_inputs,
#                   optimal_time, styles, ax, label='optimal path', end_marker=True)
# ax[0].legend()
# ax[0].set_ylim([-60, 60])
# ax[1].set_ylim([-1, MAX_THRUST + 1])
# ax[2].set_ylim([-1, MAX_THRUST + 1])
# ax[0].set_xlim([0, np.ceil(time.max())])
# ax[1].set_xlim([0, np.ceil(time.max())])
# ax[2].set_xlim([0, np.ceil(time.max())])

# # plt.show()

# fig.set_edgecolor('k')
# fig.set_linewidth(5)
# fig.savefig('tracker_vs_optimal%u.png' % id, dpi=80)


n = 600
states = rl.interpolate_data(states, n)
inputs = rl.interpolate_data(inputs, n)
time = rl.interpolate_data(time.reshape((-1, 1)), n)
optimal_states = rl.interpolate_data(optimal_states, n)
optimal_inputs = rl.interpolate_data(optimal_inputs, n)
optimal_time = rl.interpolate_data(optimal_time.reshape((-1, 1)), n)

# plot_frames(id, states, inputs, time, optimal_states,
#             optimal_inputs, optimal_time)

# rl.animate(id, states, inputs, optimal_states, save_frames=False)
# rl.animate(id, states, inputs, optimal_states, save_frames=True)
utils.create_animation_video('animation_frames/%u' %
                             id, 'animation%u.mp4' % id)
