import numpy as np
import casadi as cs
from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.estimator import StateFeedback
from do_mpc.simulator import Simulator
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Tuple

m = 1
L1 = 10
L2 = 5
I = 1 / 2 * m * (L1 ** 2)
g = 9.80665


def get_model(tracking=False) -> Model:
    model_type = 'continuous'
    model = Model(model_type)

    # init state variables
    pos = model.set_variable('_x', 'pos', (2, 1))
    theta = model.set_variable('_x', 'theta')
    dpos = model.set_variable('_x', 'dpos', (2, 1))
    dtheta = model.set_variable('_x', 'dtheta')

    # init input variables
    u = model.set_variable('_u', 'thrust', (2, 1))

    # init algebraic state variables
    ddpos = model.set_variable('_z', 'ddpos', (2, 1))
    ddtheta = model.set_variable('_z', 'ddtheta')

    # add tvp if tracking
    if tracking:
        optimal_pos = model.set_variable(
            var_type='_tvp', var_name='optimal_pos', shape=(2, 1))
        optimal_theta = model.set_variable(
            var_type='_tvp', var_name='optimal_theta')
        optimal_dpos = model.set_variable(
            var_type='_tvp', var_name='optimal_dpos', shape=(2, 1))
        optimal_dtheta = model.set_variable(
            var_type='_tvp', var_name='optimal_dtheta')

    # set right-hand-side
    model.set_rhs('pos', dpos)
    model.set_rhs('theta', dtheta)
    model.set_rhs('dpos', ddpos)
    model.set_rhs('dtheta', ddtheta)

    # set system dynamics
    t_fwd = u[0] + u[0]
    t_twist = u[1] - u[0]
    motion_equations = cs.vertcat(
        L2 * t_twist - I * ddtheta,
        -t_fwd * cs.sin(theta) - m * ddpos[0],
        -m * g + t_fwd * cs.cos(theta) - m * ddpos[1],
    )
    model.set_alg('motion_equations', motion_equations)
    model.setup()
    return model


def get_mpc_optimizer(model: Model, timestep: float, n_horizon: int) -> MPC:
    mpc = MPC(model)
    setup_mpc = {
        'n_horizon': n_horizon,
        'n_robust': 0,
        'open_loop': 0,
        't_step': timestep,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 3,
        'collocation_ni': 1,
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
    }
    mpc.set_param(**setup_mpc)
    return mpc


def set_mpc_bounds(mpc: MPC) -> None:
    mpc.bounds['lower', '_u', 'thrust'] = (0, 0)
    mpc.bounds['upper', '_u', 'thrust'] = (8, 8)
    mpc.bounds['lower', '_x', 'pos'][1] = 10

    mpc.terminal_bounds['lower', '_x', 'pos'] = (0, 10)
    mpc.terminal_bounds['upper', '_x', 'pos'] = (0, 10)

    mpc.terminal_bounds['lower', '_x', 'theta'] = 0
    mpc.terminal_bounds['upper', '_x', 'theta'] = 0

    mpc.terminal_bounds['lower', '_x', 'dpos'] = (0, 0)
    mpc.terminal_bounds['upper', '_x', 'dpos'] = (0, 0)

    mpc.terminal_bounds['lower', '_x', 'dtheta'] = 0
    mpc.terminal_bounds['upper', '_x', 'dtheta'] = 0


def get_simulator(model: Model) -> Tuple[StateFeedback, Simulator]:
    estimator = StateFeedback(model)
    simulator = Simulator(model)
    params_simulator = {
        # Note: cvode doesn't support DAE systems.
        'integration_tool': 'idas',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 0.2
    }

    simulator.set_param(**params_simulator)
    return simulator, estimator


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


def plot_trajectory(state: np.ndarray, input: np.ndarray, time: np.ndarray) -> None:
    linestyle = '-'
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(time, state[:, 0], linestyle=linestyle)
    ax[0].plot(time, state[:, 1], linestyle=linestyle)
    ax[1].plot(time, np.degrees(state[:, 2]), linestyle=linestyle)
    ax[2].plot(time, input[:, 0], linestyle=linestyle)
    ax[2].plot(time, input[:, 1], linestyle=linestyle)

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


def create_animation(fname: str, state: np.ndarray, n_steps: float, show=False) -> None:
    fig, ax = plt.subplots()
    circle = plt.Circle(state[0, :2].squeeze(), radius=L1, fc='y')
    ax.add_patch(circle)
    r, l = get_thruster_plot_lines(
        state[0, :2].squeeze(), state[0, 2].squeeze())
    bar1 = ax.plot(r[:, 0], r[:, 1], 'r', linewidth=3)
    bar2 = ax.plot(l[:, 0], l[:, 1], 'b', linewidth=3)
    ax.axhline(0, color='black')
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 80)
    ax.set_aspect(1)
    fig.tight_layout()

    def animate(i):
        circle.center = state[i, :2].squeeze()
        r, l = get_thruster_plot_lines(
            state[i, :2].squeeze(), state[i, 2].squeeze())
        bar1[0].set_data(r.T)
        bar2[0].set_data(l.T)

    anim = FuncAnimation(fig, animate, frames=n_steps, repeat=False)
    if show:
        plt.show()
    anim.save(fname, writer=PillowWriter(fps=25))
    return fig, ax
