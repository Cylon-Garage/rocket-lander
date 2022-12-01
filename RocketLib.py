import sys
import numpy as np
import casadi as cs
from scipy.integrate import solve_ivp
from do_mpc.model import Model
import pygame as pg
import matplotlib.pylab as plt
from Parameters import *
from typing import Tuple, Callable, Dict, Optional
np.set_printoptions(precision=3)


def dragon_state_function(t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    '''
    t: required variable for solve_ivp but not used in equations
    u: [thrust left, thrust right, gravity]
    '''
    t_fwd = u[0] + u[1]
    t_twist = u[1] - u[0]
    sin = np.sin(x[2])
    cos = np.cos(x[2])

    sin = np.sin(x[2])
    cos = np.cos(x[2])

    dx = np.zeros_like(x)
    dx[:3] = x[3:]
    dx[3] = -t_fwd * sin / DRAGON_MASS
    dx[4] = -u[2] + t_fwd * cos / DRAGON_MASS
    dx[5] = DRAGON_THRUST_ARM / DRAGON_I * t_twist
    return dx


def set_dragon_motion_equations(model: Model) -> None:
    u1, u2 = model.u['thrust'][0], model.u['thrust'][1]
    theta = model.x['theta']
    dd_x, dd_y, dd_theta = model.z['ddpos'][0], model.z['ddpos'][1], model.z['ddtheta']

    t_fwd = u1 + u2
    t_twist = u2 - u1
    sin = cs.sin(theta)
    cos = cs.cos(theta)
    motion_equations = cs.vertcat(
        -t_fwd * sin - DRAGON_MASS * dd_x,
        -DRAGON_MASS * G + t_fwd * cos - DRAGON_MASS * dd_y,
        DRAGON_THRUST_ARM * t_twist - DRAGON_I * dd_theta,
    )
    model.set_alg('motion_equations', motion_equations)
    return motion_equations


def get_linearized_state(x: np.ndarray, u: np.ndarray, sample_time: float, epsilon: float, state_function: Callable) -> Tuple[np.ndarray, np.ndarray]:
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


def interpolate_data(data: np.array, n: int) -> np.array:
    x = np.linspace(0, data.shape[0] - 1, n)
    xp = np.arange(data.shape[0])
    more_data = np.zeros((n, data.shape[1]))
    for i in range(data.shape[1]):
        more_data[:, i] = np.interp(x, xp, data[:, i])
    return more_data


def plot(states: np.ndarray, inputs: np.ndarray, time: np.ndarray, styles: Dict = {}, ax: Optional[plt.Axes] = None, label=''):
    linestyle = styles.setdefault('linestyle', '-')
    c1 = styles.setdefault('c1', 'b')
    c2 = styles.setdefault('c2', 'b')
    c3 = styles.setdefault('c3', 'b')

    if ax is None:
        fig, ax = plt.subplots(3, 1, figsize=(8, 6))
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
    else:
        fig = plt.gcf()
    ax[0].plot(time, np.degrees(states[:, 2]), color=c1,
               linestyle=linestyle, label=label)
    ax[1].plot(time, inputs[:, 0], color=c2, linestyle=linestyle)
    ax[2].plot(time, inputs[:, 1], color=c3, linestyle=linestyle)

    ax[0].set_ylabel('Angle [deg]')
    ax[1].set_ylabel('Thrust 1 [N]')
    ax[2].set_ylabel('Thrust 2 [N]')
    ax[0].xaxis.set_ticklabels([])
    ax[1].xaxis.set_ticklabels([])
    ax[2].set_xlabel('time [s]')
    fig.align_ylabels()
    fig.tight_layout()
    return plt.gcf(), ax


def animate(states: np.ndarray, inputs: np.ndarray, optimal_states: Optional[np.ndarray] = None, save_frames: bool = False):
    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 128, 0)
    blue = (0, 0, 255)
    pad_color = (182, 154, 63)
    water_color = (4, 48, 193)

    background = sprite = pg.sprite.Sprite()
    background.image = pg.image.load('images/sky.webp')
    background.image = pg.transform.scale(sprite.image, ANIMATION_SIZE)

    def get_sprite(fname: str, size: Tuple[int, int]) -> pg.sprite.Sprite:
        sprite = pg.sprite.Sprite()
        sprite.image = pg.image.load(fname)
        sprite.image = pg.transform.scale(sprite.image, size)
        sprite.orig_image = sprite.image
        return sprite

    dragon_size = np.array(
        [DRAGON_IMG_WIDTH, DRAGON_IMG_HEIGHT]) * ANIMATION_SCALE
    flame_size = np.array(
        (FLAME_IMG_WIDTH / 2, FLAME_IMG_HEIGHT)) * ANIMATION_SCALE
    flame_center1 = np.array(
        [DRAGON_IMG_WIDTH / 4, DRAGON_IMG_HEIGHT / 2.1]) * ANIMATION_SCALE
    flame_center2 = np.array(
        [DRAGON_IMG_WIDTH * 3 / 4, DRAGON_IMG_HEIGHT / 2.1]) * ANIMATION_SCALE

    def transform(points, x_shift):
        points[:, 0] += x_shift
        points[:, 1] *= -1
        points[:, 1] += ANIMATION_SIZE[1]
        return points

    state_scale = dragon_size[0] / DRAGON_WIDTH * \
        (DRAGON_PIXEL_CG_HEIGHT/DRAGON_PIXEL_CG_HEIGHT_OFFSET)
    states[:, :2] *= state_scale
    state_min_x, state_max_x = states[:, 0].min(), states[:, 0].max()
    x_shift = -state_min_x + \
        (ANIMATION_SIZE[0] - (state_max_x - state_min_x)) / 2
    states[:, :2] = transform(states[:, :2], x_shift)

    pad_width, pad_height = 10 * state_scale, 15
    pad_corner = np.array([[-pad_width/2, pad_height]])
    pad_corner = transform(pad_corner, x_shift)
    states[:, 1] -= pad_height

    if optimal_states is not None:
        optimal_states[:, :2] *= state_scale
        optimal_states[:, :2] = transform(optimal_states[:, :2], x_shift)
        optimal_states[:, 1] -= pad_height

    screen = pg.display.set_mode(ANIMATION_SIZE)
    screen.fill(white)
    timer = pg.time.Clock()

    dragon = get_sprite('images/dragon.png', dragon_size)
    optimal_dragon = get_sprite('images/dragon_optimal.png', dragon_size)
    flame1 = get_sprite('images/flame.png', flame_size)
    flame2 = get_sprite('images/flame.png', flame_size)

    def update_flames(i):
        flame_min, flame_max = flame_size * 0.1, flame_size * 1.2
        flame1_size = inputs[i, 0] / MAX_THRUST * \
            (flame_max - flame_min) + flame_min
        flame2_size = inputs[i, 1] / MAX_THRUST * \
            (flame_max - flame_min) + flame_min
        flame1_corner = (flame_center1[0] -
                         flame1_size[0] / 2, flame_center1[1])
        flame2_corner = (flame_center2[0] -
                         flame2_size[0] / 2, flame_center2[1])
        flame1.image = pg.transform.scale(flame1.orig_image, flame1_size)
        flame2.image = pg.transform.scale(flame2.orig_image, flame2_size)
        return flame1_corner, flame2_corner

    def display_loop(i):
        screen.fill(white)
        screen.blit(background.image, background.image.get_rect())
        if i > 1:
            pg.draw.lines(screen, green, False, states[:i, :2])
            if optimal_states is not None:
                pg.draw.lines(screen, blue, False, optimal_states[:i, :2])

        # create new dragon image and merge flames onto it
        dragon.image = pg.transform.rotate(dragon.orig_image, 0)
        if i < states.shape[0]:
            # update flames
            flame1_corner, flame2_corner = update_flames(i)
            angle = np.degrees(states[i, 2])
            offset = states[i, :2]
            dragon.image.blit(
                flame1.image, flame1.image.get_rect(topleft=flame1_corner))
            dragon.image.blit(
                flame2.image, flame2.image.get_rect(topleft=flame2_corner))
            optimal_dragon.image = pg.transform.rotate(
                optimal_dragon.orig_image, np.degrees(optimal_states[i, 2]))
            optimal_dragon.rect = optimal_dragon.image.get_rect()
            optimal_dragon.rect.center = optimal_states[i, :2]
            screen.blit(optimal_dragon.image, optimal_dragon.rect)
        else:
            angle = np.degrees(states[-1, 2])
            offset = states[-1, :2]

        # rotate merged dragon
        dragon.image = pg.transform.rotate(dragon.image, angle)
        dragon.rect = dragon.image.get_rect()

        # translate merged dragon
        dragon.rect.center = offset

        # draw everything onto screen
        screen.blit(dragon.image, dragon.rect)
        r = 5

        pg.draw.rect(
            screen,
            water_color,
            pg.Rect(0, ANIMATION_SIZE[1] - 10, ANIMATION_SIZE[0], 10)
        )
        pg.draw.rect(
            screen,
            pad_color,
            pg.Rect(*pad_corner.ravel(), pad_width, pad_height),
            border_top_left_radius=r,
            border_top_right_radius=r
        )
        if save_frames:
            pg.image.save(screen, 'animation_frames/%04u.png' % i)
    count = -1
    while True:
        count += 1
        if count <= states.shape[0]:
            display_loop(count)
        pg.display.update()
        timer.tick(40)
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_q):
                pg.quit()
                sys.exit()
