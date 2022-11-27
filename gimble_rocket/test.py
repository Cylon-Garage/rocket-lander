import RocketLib as rl
import numpy as np
import cvxpy as cp
import matplotlib.pylab as plt
from scipy.integrate import solve_ivp
np.set_printoptions(precision=3)

G = 9.80665
MAX_THRUST = 12
MAX_NOZZLE_ANGLE = np.radians(12.5)
n_states, n_inputs = 6, 3

timestep = 0.2
n_steps = 100
# n_steps = 3
n_steps = 1
n_horizon = 10
epsilon = 0.01

Q = np.diag([1, 1, 1, 1, 1, 1])
P = np.diag([1, 1, 1, 1, 1, 1]) * 1
R = np.diag([1, 1, 1])
R = np.diag([0, 0, 0])

x_f = np.array([0, rl.L1, 0, 0, 0, 0], dtype=float)
x_0 = np.array([-30, 60, np.pi / 4, -10, 10, 0], dtype=float)
u_0 = np.array([0, 0, G], dtype=float)


def mpc(x_0: np.ndarray, u_last: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    x = cp.Variable((n_states, n_horizon + 1))
    u = cp.Variable((n_inputs, n_horizon))

    cost = 0
    constraints = [x[:, 0] == x_0.squeeze()]
    for t in range(n_horizon):
        cost += u[0, t]
        constraints += [
            x[:, t + 1] == A @ x[:, t] + B @ u[:, t],
            x[1] >= rl.L1,

            u[0, t] >= 0,
            u[0, t] <= MAX_THRUST,
            u[1, t] >= -MAX_NOZZLE_ANGLE,
            u[1, t] <= MAX_NOZZLE_ANGLE,
            u[2, t] == G,
        ]
    cost += cp.quad_form(x[:, -1] - x_f, P)
    constraints += [
        x[1] >= rl.L1,
    ]
    problem = cp.Problem(cp.Minimize(cost), constraints)
    j_star = problem.solve()
    u_star = u.value
    x_star = x.value
    print(j_star)
    # print(u_star[:, 0])
    # print(x_star[:, 0])
    # print(x_star[:, -1])
    # print(x_ref[-1, :])
    return u_star[:, 0]


x = x_0
u = u_0
mpc_inputs = [u]
states = [x]
for t in range(n_steps):
    A, B = rl.get_linearized_state(
        x, u, timestep * 2, epsilon, rl.rocket_state_function)
    u = mpc(x, u, A, B)
    print(u)
