import RocketLib as rl
import numpy as np
import cvxpy as cp
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
from Parameters import *
import utils
np.set_printoptions(precision=3)

optimal_inputs = np.load('saved_states/optimal_input.npy')
optimal_states = np.load('saved_states/optimal_state.npy')
optimal_steps = optimal_states.shape[0]
n_states, n_inputs = 6, 3

n_steps = 150
epsilon = 0.01

x_0 = np.array([*INITIAL_POSITION, INITIAL_THETA, *INITIAL_VELOCITY, 0])
u_0 = np.array([0, 0, G], dtype=float)

Q = np.diag([1, 1, 1, 1, 1, 1])
P = np.diag([1, 1, 1, 1, 1, 1])
Q = np.diag([1, 1, 50, 1, 1, 10])
P = np.diag([1, 1, 50, 1, 1, 10])
R = np.diag([0.1, 0.1, 0])


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


def run_tracker(rerun=False):
    state_fname = 'saved_states/tracker_state.npy'
    input_fname = 'saved_states/tracker_input.npy'

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


rerun = False
# rerun = True
styles = {
    'linestyle': '-',
    'c1': 'g',
    'c2': 'g',
    'c3': 'g',
}
states, inputs = run_tracker(rerun)
time = np.arange(states.shape[0]) * TIMESTEP


fig, ax = rl.plot(states, inputs, time, label='tracker', styles=styles)
styles = {
    'linestyle': '--',
    'c1': 'b',
    'c2': 'b',
    'c3': 'b',
}
optimal_time = np.arange(optimal_states.shape[0]) * TIMESTEP
fig, ax = rl.plot(optimal_states, optimal_inputs,
                  optimal_time, styles, ax, label='optimal path')

ax[0].legend()
ax[0].set_ylim([-60, 60])
ax[1].set_ylim([-1, MAX_THRUST + 1])
ax[2].set_ylim([-1, MAX_THRUST + 1])

# plt.show()
fig.savefig('tracker_vs_optimal.png', dpi=80)


n = 600
states = rl.interpolate_data(states, n)
inputs = rl.interpolate_data(inputs, n)
optimal_states = rl.interpolate_data(optimal_states, n)
# rl.animate(states, inputs, optimal_states, save_frames=True)
# utils.create_animation_video('animation_frames', 'animation.mp4')
