import numpy as np
import os
import casadi
import RocketLib as rl
from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.estimator import StateFeedback
from do_mpc.simulator import Simulator
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Tuple
np.set_printoptions(precision=2)

ROCKET_MASS = 1
L1 = 10
I = 1 / 2 * ROCKET_MASS * (L1 ** 2)
G = 9.80665

MAX_THRUST = 12
MAX_NOZZLE_ANGLE = np.radians(20)


# MAX_NOZZLE_ANGLE = MAX_NOZZLE_ANGLE)


def get_model() -> Model:
    model_type = 'continuous'
    model = Model(model_type)

    # init state variables
    pos = model.set_variable('_x', 'pos', (2, 1))
    theta = model.set_variable('_x', 'theta')
    dpos = model.set_variable('_x', 'dpos', (2, 1))
    dtheta = model.set_variable('_x', 'dtheta')

    # init input variables
    u = model.set_variable('_u', 'inputs', (2, 1))

    # init algebraic state variables
    ddpos = model.set_variable('_z', 'ddpos', (2, 1))
    ddtheta = model.set_variable('_z', 'ddtheta')

    # set right-hand-side
    model.set_rhs('pos', dpos)
    model.set_rhs('theta', dtheta)
    model.set_rhs('dpos', ddpos)
    model.set_rhs('dtheta', ddtheta)

    # set system dynamics
    angle = theta + u[1]
    sin1 = np.sin(angle)
    sin2 = np.sin(u[1])
    cos = np.cos(angle)
    motion_equations = casadi.vertcat(
        ROCKET_MASS * ddpos[0] + u[0] * sin1,
        ROCKET_MASS * ddpos[1] + ROCKET_MASS * G - u[0] * cos,
        I * ddtheta + L1 * u[0] * sin2,
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
    mpc.bounds['lower', '_u', 'inputs'] = (0, -MAX_NOZZLE_ANGLE)
    mpc.bounds['upper', '_u', 'inputs'] = (MAX_THRUST, MAX_NOZZLE_ANGLE)
    mpc.bounds['lower', '_x', 'pos'][1] = L1

    mpc.terminal_bounds['lower', '_x', 'pos'] = (0, L1)
    mpc.terminal_bounds['upper', '_x', 'pos'] = (0, L1)

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


def get_optimal_path(rerun=False):
    state_fname = 'optimal_state.npy'
    input_fname = 'optimal_input.npy'

    if (os.path.isfile(state_fname) and os.path.isfile(input_fname) and not rerun):
        state = np.load(state_fname)
        input = np.load(input_fname)

    else:
        timestep = 0.2
        n_horizon = 50
        n_steps = 100

        model = get_model()
        mpc = get_mpc_optimizer(model, timestep, n_horizon)

        running_cost = model.u['inputs'][0]
        terminal_cost = casadi.DM(0)
        mpc.set_objective(mterm=terminal_cost, lterm=running_cost)
        mpc.set_rterm(inputs=0)

        set_mpc_bounds(mpc)
        mpc.setup()
        mpc.set_initial_guess()

        simulator, estimator = get_simulator(model)
        simulator.setup()
        simulator.x0['pos'] = [100.0, 60.0]
        simulator.x0['theta'] = np.pi/4
        simulator.x0['dpos'] = [-10.0, 1.0]
        simulator.x0['dtheta'] = [0.0]
        x0 = simulator.x0.cat.full()
        estimator.x0 = x0
        mpc.x0 = x0

        simulator.reset_history()
        simulator.x0 = x0
        for _ in range(n_steps):
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)
        state = mpc.data['_x']
        input = mpc.data['_u']

        with open(state_fname, 'wb') as f:
            np.save(f, state)
        with open(input_fname, 'wb') as f:
            np.save(f, input)
    return state, input


rerun = False
rerun = True
path_state, path_input = get_optimal_path(rerun)
path_time = np.arange(path_state.shape[0]) * 0.2


rl.plot_trajectory(path_state, path_input, path_time)
plt.show()

n = 200
path_state = rl.interpolate_data(path_state, n)
path_input = rl.interpolate_data(path_input, n)
rl.create_animation('test2.gif', path_state, path_input)
