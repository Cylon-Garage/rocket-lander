import RocketLib as rl
import numpy as np
import cvxpy as cp
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
np.set_printoptions(precision=3)

optimal_inputs = np.loadtxt('optimal_input2.csv', delimiter=',')
optimal_states = np.loadtxt('optimal_state2.csv', delimiter=',')
optimal_steps = optimal_states.shape[0]
n_states, n_inputs = 6, 3

timestep = 0.2
n_steps = 100
# n_steps = 3
n_steps = 100
n_horizon = 10
epsilon = 0.01

G = 9.80665
MAX_THRUST = 12
MAX_NOZZLE_ANGLE = 12.5

x_0 = np.array([-30, 60, np.pi / 4, -10, 10, 0], dtype=float)
x_0 = np.array([80, 200, np.pi / 4, -10, 10, 0], dtype=float)
u_0 = np.array([3, 1, G], dtype=float)

Q = np.diag([1, 1, 1, 1, 1, 1])
P = np.diag([1, 1, 1, 1, 1, 1])
R = np.diag([1, 1, 1])
R = np.diag([0.1, 0, 0])
# R = np.diag([0, 0, 0])


MAX_NOZZLE_ANGLE = np.radians(MAX_NOZZLE_ANGLE)


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

            u[0, t] >= 0,
            u[0, t] <= MAX_THRUST,
            u[1, t] >= -MAX_NOZZLE_ANGLE,
            u[1, t] <= MAX_NOZZLE_ANGLE,
            u[2, t] == G,

        ]
    cost += cp.quad_form(x[:, -1] - x_ref[-1, :], P)
    constraints += [
        x[1] >= rl.L1,
    ]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    j_star = problem.solve()
    u_star = u.value
    x_star = x.value
    # print(j_star)
    # print(u_star[:, 0])
    # print(x_star[:, 0])
    # print(x_star[:, -1])
    # print(x_ref[-1, :])
    return u_star[:, 0]


def get_reference_state(position: np.ndarray, last_idx: int) -> np.ndarray:
    nearest_idx = np.argmin(np.linalg.norm(
        optimal_states[:, :2] - position, axis=1)) + last_idx + 1
    idx = np.arange(nearest_idx, nearest_idx + n_horizon)
    idx[idx > (optimal_steps - 1)] = optimal_steps - 1
    return optimal_states[idx, :], idx[0]


def get_reference_state2(t: int) -> np.ndarray:
    idx = np.arange(t, t + n_horizon)
    idx[idx > (optimal_steps - 1)] = optimal_steps - 1
    return optimal_states[idx, :]


x = x_0
u = u_0
x_ref_idx = -1
mpc_inputs = [u]
states = [x]
for t in range(n_steps):
    # x_ref, x_ref_idx = get_reference_state(x[:2], x_ref_idx)
    x_ref = get_reference_state2(t)
    # print(x_ref_idx)
    A, B = rl.get_linearized_state(
        x, u, timestep, epsilon, rl.rocket_state_function)

    u = mpc(x, u, A, B, x_ref)
    x = solve_ivp(rl.rocket_state_function, t_span=[
                  0, timestep], y0=x, args=(u,)).y[:, -1]
    states.append(x)

    print(u[0], np.degrees(u[1]), u[2])
    print(optimal_inputs[t, 0], np.degrees(optimal_inputs[t, 1]))
    print('-' * 15)
    temp = np.array(states)
    plt.plot(optimal_states[:, 0], optimal_states[:, 1], 'b', linewidth=4)
    plt.plot(optimal_states[0, 0], optimal_states[0, 1], 'r', marker='o')
    plt.plot(x_ref[:, 0], x_ref[:, 1], 'm', '-o', linewidth=2.5)
    plt.plot(temp[:, 0], temp[:, 1], 'c', '-o', linewidth=1.5)
    plt.show()
    # print(temp.shape)
    # print(u)
    # print(x)
