import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Tuple
import matplotlib as mpl
np.set_printoptions(precision=3)

ROCKET_MASS = 1
ROCKET_WIDTH = 4
L1 = 10
L2 = 5
I = 1 / 2 * ROCKET_MASS * (L1 ** 2)
ROCKET_COLOR = (140/255, 151/255, 168/255)
FLAME_COLOR = (230/255, 133/255, 44/255)
FLAME_SCALE = 10


def matlab_example_state_function(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    t_fwd = u[0] + u[1]
    t_twist = u[1] - u[0]
    sin = np.sin(x[2])
    cos = np.cos(x[2])

    dx = np.zeros_like(x)
    dx[:3] = x[3:]
    dx[3] = -t_fwd * sin / ROCKET_MASS
    dx[4] = -u[2] + t_fwd * cos / ROCKET_MASS
    dx[5] = L2 / I * t_twist
    return dx


def rocket_state_function(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    angle = x[2] + u[1]
    sin1 = np.sin(angle)
    sin2 = np.sin(u[1])
    cos = np.cos(angle)

    dx = np.zeros_like(x)
    dx[:3] = x[3:]
    dx[3] = -u[0] * sin1 / ROCKET_MASS
    dx[4] = -u[2] + u[0] * cos / ROCKET_MASS
    dx[5] = -L1 * u[0] * sin2 / I
    return dx


def get_linearized_state(x: np.ndarray, u: np.ndarray, sample_time: float, epsilon: float, state_function):
    x = np.array(x).ravel()
    u = np.array(u).ravel()

    nx, nu = x.shape[0], u.shape[0]
    x_eps, u_eps = np.eye(nx) * epsilon, np.eye(nu) * epsilon

    x_plus = (np.tile(x, (nx, 1)) + x_eps).T
    x_minus = (np.tile(x, (nx, 1)) - x_eps).T
    u_plus = (np.tile(u, (nu, 1)) + u_eps).T
    u_minus = (np.tile(u, (nu, 1)) - u_eps).T
    states_plus, states_minus = np.zeros((nx, nx)), np.zeros((nx, nx))
    inputs_plus, inputs_minus = np.zeros((nx, nu)), np.zeros((nx, nu))
    for i in range(nx):

        states_plus[:, i] = solve_ivp(state_function,
                                      t_span=[0, sample_time],
                                      y0=x_plus[:, i],
                                      args=(u,)).y[:, -1]
        states_minus[:, i] = solve_ivp(state_function,
                                       t_span=[0, sample_time],
                                       y0=x_minus[:, i],
                                       args=(u,)).y[:, -1]
    for i in range(nu):

        inputs_plus[:, i] = solve_ivp(state_function,
                                      t_span=[0, sample_time],
                                      y0=x,
                                      args=(u_plus[:, i],)).y[:, -1]
        inputs_minus[:, i] = solve_ivp(state_function,
                                       t_span=[0, sample_time],
                                       y0=x,
                                       args=(u_minus[:, i],)).y[:, -1]
    A = (states_plus - states_minus) / (2 * epsilon)
    B = (inputs_plus - inputs_minus) / (2 * epsilon)
    return A, B


def get_thruster_plot_lines(center: np.ndarray, angle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos, -sin],
        [sin, cos],
    ])

    r = np.zeros((2, 2))
    r[0, 0] = L2 - 1
    r[1, 0] = L2 + 1
    l = np.array(r)
    l[:, 0] *= -1

    r[0, :] = R @ r[0, :]
    r[1, :] = R @ r[1, :]
    l[0, :] = R @ l[0, :]
    l[1, :] = R @ l[1, :]

    r[:, 0] += center[0]
    r[:, 1] += center[1]
    l[:, 0] += center[0]
    l[:, 1] += center[1]
    return r, l


def plot_trajectory_matlab_example(states: np.ndarray, inputs: np.ndarray, time: np.ndarray) -> None:
    linestyle = '-'
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(states[:, 0], states[:, 1], linestyle=linestyle)
    ax[1].plot(time, np.degrees(states[:, 2]), linestyle=linestyle)
    ax[2].plot(time, inputs[:, 0], linestyle=linestyle)
    ax[2].plot(time, inputs[:, 1], linestyle=linestyle)

    ax[0].set_ylabel('pos')
    ax[1].set_ylabel('angle')
    ax[2].set_ylabel('thrust')
    ax[0].xaxis.set_ticklabels([])
    ax[1].xaxis.set_ticklabels([])
    ax[2].set_xlabel('time [s]')
    fig.align_ylabels()
    fig.tight_layout()
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    return fig, ax


def plot_trajectory(states: np.ndarray, inputs: np.ndarray, time: np.ndarray) -> None:
    linestyle = '-'
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(states[:, 0], states[:, 1], linestyle=linestyle)
    # ax[0].plot(time, state[:, 1], linestyle=linestyle)
    ax[1].plot(time, np.degrees(states[:, 2]), linestyle=linestyle)
    ax[2].plot(time, inputs[:, 0], linestyle=linestyle)
    ax[2].plot(time, np.degrees(inputs[:, 1]), linestyle=linestyle)

    ax[0].set_ylabel('pos')
    ax[1].set_ylabel('angle')
    ax[2].set_ylabel('thrust')
    ax[0].xaxis.set_ticklabels([])
    ax[1].xaxis.set_ticklabels([])
    ax[2].set_xlabel('time [s]')
    fig.align_ylabels()
    fig.tight_layout()
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    return fig, ax


def create_animation_matlab_example(fname: str, state: np.ndarray, show=False) -> None:
    fig, ax = plt.subplots()
    circle = plt.Circle(state[0, :2].squeeze(), radius=L1, fc='y')
    ax.add_patch(circle)
    r, l = get_thruster_plot_lines(
        state[0, :2].squeeze(), state[0, 2].squeeze())
    bar1 = ax.plot(r[:, 0], r[:, 1], 'r', linewidth=3)
    bar2 = ax.plot(l[:, 0], l[:, 1], 'b', linewidth=3)
    ax.axhline(0, color='black')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-10, 100)
    ax.set_aspect(1)
    fig.tight_layout()

    def animate(i):
        circle.center = state[i, :2].squeeze()
        r, l = get_thruster_plot_lines(
            state[i, :2].squeeze(), state[i, 2].squeeze())
        bar1[0].set_data(r.T)
        bar2[0].set_data(l.T)

    anim = FuncAnimation(fig, animate, frames=state.shape[0], repeat=False)
    if show:
        plt.show()
    anim.save(fname, writer=PillowWriter(fps=25))
    return fig, ax


def interpolate_data(data: np.array, n: int) -> np.array:
    x = np.linspace(0, data.shape[0] - 1, n)
    xp = np.arange(data.shape[0])
    more_data = np.zeros((n, data.shape[1]))
    for i in range(data.shape[1]):
        more_data[:, i] = np.interp(x, xp, data[:, i])
    return more_data


def create_animation(fname: str, state: np.ndarray, inputs: np.ndarray, show: bool = False) -> None:
    flame_width = 0.1

    def get_plot_points(x, u):
        xc, yc = x[0], x[1]
        rocket_anchor = (xc - ROCKET_WIDTH / 2, yc - L1)
        flame_anchor = (xc - flame_width / 2, yc - L1)
        flame_rotation_point = (xc, yc - L1)
        return rocket_anchor, flame_anchor, flame_rotation_point

    fig, ax = plt.subplots()
    rocket_anchor, flame_anchor, flame_rotation_point = get_plot_points(
        state[0, :], inputs[0, :])
    rocket = mpatches.Rectangle(
        rocket_anchor,
        ROCKET_WIDTH,
        L1 * 2,
        angle=np.degrees(state[0, 2]),
        color=ROCKET_COLOR)
    flame = mpatches.Rectangle(
        flame_anchor,
        flame_width,
        inputs[0, 0]/12 * FLAME_SCALE,
        angle=np.degrees(inputs[0, 1]) + 180,
        rotation_point=flame_rotation_point,
        color=FLAME_COLOR)

    ax.add_patch(rocket)
    ax.add_patch(flame)
    ax.axhline(0, color='black')
    ax.set_xlim(-120, 120)
    ax.set_ylim(-5, 200)
    ax.set_aspect(1)

    def animate(i):
        rocket_anchor, flame_anchor, flame_rotation_point = get_plot_points(
            state[i, :], inputs[i, :])
        rocket.set_xy(rocket_anchor)
        rocket.set_angle(np.degrees(state[i, 2]))

        flame.rotation_point = flame_rotation_point
        flame.set_xy(flame_anchor)
        flame.set_angle(np.degrees(inputs[i, 1] + state[i, 2]) + 180)
        flame.set_height(inputs[i, 0]/12 * FLAME_SCALE)

    anim = FuncAnimation(fig, animate, frames=state.shape[0], repeat=False)
    if show:
        plt.show()
    anim.save(fname, writer=PillowWriter(fps=25))
    return fig, ax


if __name__ == '__main__':
    sample_time = 0.2
    x0 = np.array([-30, 60, 0, 0, 0, 0])
    u = np.array([3, 1])

    states = []
    for i in range(100):
        solution = solve_ivp(matlab_example_state_function,
                             t_span=[0, sample_time],
                             y0=x0,
                             args=(u,))
        x0 = solution.y.T[-1, :]
        states.append(x0)

    states = np.array(states)
    matlab = np.loadtxt('states.csv', delimiter=',').reshape((-1, 6))
    plt.plot(matlab[:, 0], matlab[:, 1], 'b')
    plt.plot(states[:, 0], states[:, 1], 'r')
    plt.show()
