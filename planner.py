import numpy as np
import casadi
import RocketLib as rl
from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.estimator import StateFeedback
from do_mpc.simulator import Simulator
import RocketLib as rl
from Parameters import *
import matplotlib.pylab as plt
from typing import Tuple, List
np.set_printoptions(precision=2)


def get_model() -> Model:
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

    # set right-hand-side
    model.set_rhs('pos', dpos)
    model.set_rhs('theta', dtheta)
    model.set_rhs('dpos', ddpos)
    model.set_rhs('dtheta', ddtheta)

    # set system dynamics
    rl.set_dragon_motion_equations(model)
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
    mpc.bounds['upper', '_u', 'thrust'] = (MAX_THRUST, MAX_THRUST)
    mpc.bounds['lower', '_x', 'pos'][1] = DRAGON_CG_HEIGHT

    mpc.terminal_bounds['lower', '_x', 'pos'] = (0, DRAGON_CG_HEIGHT)
    mpc.terminal_bounds['upper', '_x', 'pos'] = (0, DRAGON_CG_HEIGHT)

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
        'integration_tool': 'idas',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 0.2
    }

    simulator.set_param(**params_simulator)
    return simulator, estimator


def get_optimal_path(initial_state: List, rerun=False) -> Tuple[np.ndarray, np.ndarray]:
    state_fname = 'saved_states/optimal_state.npy'
    input_fname = 'saved_states/optimal_input.npy'

    if not rerun:
        states = np.load(state_fname)
        inputs = np.load(input_fname)

    else:
        n_steps = 150

        model = get_model()
        mpc = get_mpc_optimizer(model, TIMESTEP, PLANNER_N_HORIZON)

        running_cost = model.u['thrust'][0] + model.u['thrust'][1]
        terminal_cost = casadi.DM(0)
        mpc.set_objective(mterm=terminal_cost, lterm=running_cost)
        mpc.set_rterm(thrust=0.1)

        set_mpc_bounds(mpc)
        mpc.setup()
        mpc.set_initial_guess()

        simulator, estimator = get_simulator(model)
        simulator.setup()
        simulator.x0['pos'] = initial_state[:2]
        simulator.x0['theta'] = initial_state[2]
        simulator.x0['dpos'] = initial_state[3:5]
        simulator.x0['dtheta'] = initial_state[5]
        x0 = simulator.x0.cat.full()
        estimator.x0 = x0
        mpc.x0 = x0

        simulator.reset_history()
        simulator.x0 = x0
        for _ in range(n_steps):
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)
        states = mpc.data['_x']
        inputs = mpc.data['_u']

        with open(state_fname, 'wb') as f:
            np.save(f, states)
        with open(input_fname, 'wb') as f:
            np.save(f, inputs)
    return states, inputs


rerun = False
# rerun = True
states, inputs = get_optimal_path(X0, rerun)
time = np.arange(states.shape[0]) * TIMESTEP
fig, ax = rl.plot(states, inputs, time)
plt.show()
n = 600
states = rl.interpolate_data(states, n)
inputs = rl.interpolate_data(inputs, n)
rl.animate(states, inputs)
