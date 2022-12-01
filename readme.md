# Rocket Landing Trajectory Tracking with Linear MPC
## Problem Statement
Starting from an initial state $\underline{x}_0$, we are given an optimal state trajectory $\underline{x}_{optimal}(t)$for our dragon capsule to track as well as terminal state requirements $\underline{x}_{terminal}$. The capsule states are defined below in equation 4 and $\underline{x}_{terminal}$ is simply $\underline{0}$. Meaning the landing site is at $(0, 0)$ and we want to reach it upright with zero linear and angular velocity. The capsule has two thrusters (shown below) which are limited to ${F_1}_{min}={F_2}_{min}=0$ and ${F_1}_{max}={F_2}_{max}=10N$.
## Dynamic Equations
<figure>
<p align="center">
<img src="https://github.com/Cylon-Garage/rocket-lander/blob/master/freebody.svg?raw=true" alt="Trulli" style="width:65%">
</p>
<figcaption align="center"><b>Dragon Capsule Freebody Diagram</b></figcaption>
</figure>

$$
\begin{align}
\sum F_x &= m \ddot{x} = -(F_1 + F_2) \cos\theta \\
\sum F_y &= m \ddot{y} = -mg + (F_1 + F_2) \sin\theta \\
\sum M_{cg} &= I \ddot{\theta} =  L_{thrust} (F_2 - F_1)
\end{align}
$$

## Nonlinear State Equations
$$
\begin{align}
\underline{x} &= \begin{bmatrix} x & y & \theta & \dot{x} & \dot{y} & \dot{\theta}\end{bmatrix}^T \\
\underline{u} &= \begin{bmatrix} F_1 & F_2 \end{bmatrix}^T \\
\end{align}
$$
$$
\begin{align}
f(\underline{x}, \underline{u}) = \dot{\underline{x}} = \begin{bmatrix}
x_4 \\ x_5 \\ x_6 \\
\frac{-(u_1 + u_2)}{m}  \sin x_3 \\
-g + \frac{-(u_1 + u_2)}{m}  \cos x_3 \\
\frac{L_{thruster}}{I} (u_2 - u_1)
\end{bmatrix}
\end{align}
$$
## Problems With Implementing Linear MPC
The state equations (6) are **nonlinear** due to the $\sin$ and $\cos$ functions. Those functions are also **non-convex**. Therefore, a linear MPC cannot directly be implemented on this system. The system needs linearized first.


## Linearization
### Linearizing Around a Fixed Point (Jacobian Linearization)
$$
\begin{align}
A &= \begin{bmatrix} \frac{\partial{f_i}}{\partial{x_j}} \end{bmatrix} \rvert_{x_{eq}, u_{eq}} \\
B &= \begin{bmatrix} \frac{\partial{f_i}}{\partial{u_j}} \end{bmatrix} \rvert_{x_{eq}, u_{eq}} \\
\text{where} \enspace f(\underline{x}_{eq}, \underline{x}_{eq}) &= \dot{\underline{x}}_{eq} = \underline{0}
\end{align}
$$
The state matrices $A$ and $B$ are obtained by evaluating the Jacobian at $\underline{x}_{eq}$. The fixed point $\underline{x}_{eq}, \underline{u}_{eq}$ is obtained by setting equation 6 to $\underline{0}$ and solving for $\underline{x}_{eq}, \underline{u}_{eq}$. One equation resulting from doing this is $\frac{-(u_1 + u_2)}{m}  \sin x_3 = 0$. Which means that $u_1 = -u_2$ and/or $\theta = n\pi$ where $n = 1, 2 \dots $. The min and max limits on the thrusters mean that $u_1$ can only equal $-u_2$ when $u_1 = u_2 = 0$. This limits the validity of our state matrices to two different situations. One where the dragon capsule is close to upright or upside down. The other is when both thrusters are off. These two states are too limiting and it will not be possible to follow a given state trajectory that will have a wide range of capsule angles that require a wide range of thrust inputs. A different linearization technique must be employed

### Finite Difference Linearization
I can across a thesis by [Ferrante](https://project-archive.inf.ed.ac.uk/msc/20172139/msc_proj.pdf) where he linearized his state equations using finite difference. This linearization is done at the current state, $\underline{x}$, so it is not limited to a fixed point like the technique above. The idea is similar to fixed point linearization as it also uses the Jacobian. However, here finite difference is used to obtain the Jacobian.
Assume you are at some state $\underline{x}(t)$ with input $\underline{u}(t)$. The following state $\underline{x}(t + \Delta t)$ after $\Delta t$ seconds can be found by integrating the system of ODEs (6). Lets add and subtract a small perturbation to $x_1(t)$ such that
$$
\underline{x}_+(t) = \begin{bmatrix} x + \epsilon \\ y \\ \theta \\ \dot{x} \\ \dot{y} \\ \dot{\theta}\end{bmatrix} \text{and} \enspace \underline{x}_-(t) = \begin{bmatrix} x - \epsilon \\ y \\ \theta \\ \dot{x} \\ \dot{y} \\ \dot{\theta}\end{bmatrix}
$$
We can again solve the system of ODEs to obtain $\underline{x}_+(t + \Delta t)$ and $\underline{x}_-(t + \Delta t)$. These two states can be used to approximate $\frac{\partial{f(t)}}{\partial{x_1}}$:
$$
\frac{\partial{f(t)}}{\partial{x_1}} \approx \Delta_{x_1}f(t) =  \frac{\underline{x}_+(t + \Delta t) - \underline{x}_-(t + \Delta t)}{2 \epsilon}
$$
This partial derivative approvimation can be used to estimate the state matrices $A$ and $B$ at time $t$ with
$$
\begin{align}
A(t) &\approx \begin{bmatrix}
\Delta_{x_1}f(t) &
\Delta_{x_2}f(t) &
\Delta_{x_3}f(t) &
\Delta_{x_4}f(t) &
\Delta_{x_5}f(t) &
\Delta_{x_6}f(t)
\end{bmatrix} \\
B(t) &\approx \begin{bmatrix}
\Delta_{u_1}f(t) &
\Delta_{u_2}f(t)
\end{bmatrix}
\end{align}
$$

Ferrante performed this approximation by indirectly calculating each $\underline{x}_+(t + \Delta t)$ and $\underline{x}_-(t + \Delta t)$ through a simulation. He used Box2D as his physics engine and calculated each finite difference from the resulting simulation state after using $\epsilon$ perturbations though a timestep $\Delta t$. I thought this could be simplified and replaced the simulator with [SciPy's solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html). This is similar to [Matlab's ODE45](https://www.mathworks.com/help/matlab/ref/ode45.html) solver.

#### Gravity
This finite difference linearization technique was new to me. It made sense that it would work but I had to verify it somehow. Lets call the true solution $\underline{\hat{x}}_{t+1}$. $A(t)$ and $B(t)$ are valid near $\underline{x}(t)$. So $\underline{x}_{t+1} = A \underline{x}_t + B \underline{u}_t \approx \underline{\hat{x}}_{t+1}$. So if you start with an initial state, $\underline{x}_0$ and evolved it through a short time horizon ($n_{steps}=10$ with $\Delta t=0.2$) the two trajectories should be similar. My initial tests did not show this as seen below.

<figure>
<p align="center">
<img src="https://" alt="Trulli" style="width:65%">
</p>
<figcaption align="center"><b>Initial Finite Difference Linearization Verification</b></figcaption>
</figure>

Inspecting equations 6, 10 and 11 reveal the issue. Gravity is never perturbed during the calculation of $A$ & $B$. This means that gravity is essentially subtracted out when calculating the finite differences. To resolve this, gravity has to be involed in the Jacobian. This is done by treating gravity as an input:
$$
\underline{u} = \begin{bmatrix} F_1 & F_2 & g\end{bmatrix}^T
$$
This modification results in $A \underline{x}_t + B \underline{u}_t \approx \underline{\hat{x}}_{t+1}$, verifying the finite difference linearization technique.
 
 <figure>
<p align="center">
<img src="https://" alt="Trulli" style="width:65%">
</p>
<figcaption align="center"><b>Treating Gravity as an Input</b></figcaption>
</figure>

## Obtaining an Optimal Track (Planning)
Nonlinear Model Predictive Control (NMPC) was utilized to obtain an optimal track. A NMPC optimizes the nonlinear dynamics equations directly without the need for linearization. The optimizations were performed by minimizing total thrust (fuel) used from an initial state $\underline{x}_0$ to final state $\underline{x}_f=\underline{0}$ The [do-mpc](https://www.do-mpc.com) library was used to implement the optimization. NMPCs were not covered in EE688 and I do not have a good understanding of it. It is outside the scope of this project and was solely used to obtain an optimal track.

## Model Predictive Controller
I implemented the tracking MPC using quadratic cost:
$$
\begin{align}
J &= \underline{x}_{error}^T(N) P \underline{x}_{error}(N)  + \sum_{n=0}^{N-1} 
[\underline{x}_{error}^T(n) Q \underline{x}_{error}(n) + \underline{u}_{delta}^T(n) R \underline{u}_{delta}(n)] \\
\text{where} \enspace \underline{x}_{error} &= \underline{x} - \underline{x}_{ref} \\
\text{and} \enspace \underline{u}_{delta} &= \underline{u}_{n} - \underline{u}_{n-1}
\end{align}
$$
I found through empirical testing that penalizing $\theta$ and $\dot{\theta}$ were much more important than penalizing position and velocity. Maintaining track was also more important that reducing changes in input. This resulted in the following penalty matrices:
$$
\begin{align}
P = Q &= \begin{bmatrix} 
1 & 0 & 0 &0 &0 &0 \\
0 & 1 & 0 &0 &0 &0 \\
0 & 0 & 50& 0& 0& 0 \\
0 & 0 & 0 &1 &0 &0 \\
0 & 0 & 0 &0 &1 &0 \\
0 & 0 & 0 &0 &0 &10 \\
\end{bmatrix} \\
R &= \begin{bmatrix} 
0.1 & 0 & 0 \\
0 & 0.1 & 0 \\
0 & 0 & 0.1 \\
\end{bmatrix} 
\end{align}
$$
The linearized state matrices are only valid near $\underline{x}_t$, so I kept the MPC horizon short ($n_{horizon}$). Towards the end of the simulation the horizon will extends beyond data available within the provided optimal track. This is handled by forcing all states beyond the optimal trajectory to equal $\underline{0}$. This data extension was enough to implicitly enforce terminal state requirements $\underline{x}_{terminal}$ Terminal state requirments were never explicitly enforced.

The MPC code itself is very simple and is very similar to the one I implemented on HW3. Given an initial state $\underline{x}_0$ and the following constants:
$$
\begin{align*}
n_{steps} &= 100 \\ n_{horizon} &= 10 \\ \Delta t &= 0.2 \\ \epsilon &= 0.01 \\ G &= 9.80665
\end{align*}
$$

### Main Program
```python
x = x_0
u = [0, 0, G]
x = x_0
u = u_0
inputs = [u]
states = [x]
for t in range(n_steps - 1):
    x_ref = get_reference_state(x[:2])
    A, B = rl.get_linearized_state(
        x, u, TIMESTEP, epsilon, rl.dragon_state_function)

    u_optimal = mpc(x, u, A, B, x_ref)
    u = u_optimal[:, 0]
    x = solve_ivp(rl.dragon_state_function, t_span=[
        0, TIMESTEP], y0=x, args=(u,)).y[:, -1]
```
#### MPC Program
```python
def mpc(
    x_0: np.ndarray, 
    u_last: np.ndarray, 
    A: np.ndarray, 
    B: np.ndarray, 
    x_ref: np.ndarray) -> np.ndarray:

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
```
**get_reference_state**, **dragon_state_function**, **get_linearized_state** are shown in the Appendix. All code is available on [GitHub](https://github.com/Cylon-Garage/rocket-lander)

## Tracking Results
### Track 1
$\underline{x}_0 = \begin{bmatrix} 200 & 100 & -45° & -7.1 & -7.1 & 0\end{bmatrix}^T$
https://user-images.githubusercontent.com/100429965/204759660-715809cd-440b-4427-94ce-727bdb4b9641.mp4
### Track 2
$\underline{x}_0 = \begin{bmatrix} 200 & 100 & -45° & -7.1 & -7.1 & 0\end{bmatrix}^T$
https://user-images.githubusercontent.com/100429965/204759660-715809cd-440b-4427-94ce-727bdb4b9641.mp4

## Appendix
#### get_reference_state
```python
def get_reference_state(
    position: np.ndarray) -> np.ndarray:

    nearest_idx = np.argmin(np.linalg.norm(
        optimal_states[:, :2] - position, axis=1))
    idx = np.arange(nearest_idx, nearest_idx + TRACKER_N_HORIZON)
    idx[idx > (optimal_steps - 1)] = optimal_steps - 1
    return optimal_states[idx, :]
```

#### dragon_state_function
```python
def dragon_state_function(
    t: float, 
    x: np.ndarray, 
    u: np.ndarray) -> np.ndarray:

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
```
#### get_linearized_state
```python
def get_linearized_state(
    x: np.ndarray, 
    u: np.ndarray, 
    sample_time: float, 
    epsilon: float, 
    state_function: Callable) -> Tuple[np.ndarray, np.ndarray]:

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
```


## TBD
$$
\underline{u} = \begin{bmatrix} F_1 \enspace F_2 \enspace g \end{bmatrix}^T
$$


$F_1$

$F_2$

$mg$

$L_{thruster}$

$\theta$

## TBD2


```
# MAIN PROGRAM
x = x_0
u = [0, 0, G]
FOR t in n_steps:
    x_ref = get_reference_state(x)
    A, B = get_linearized_state(x, u, time_step, epsilon)
    u_optimal = mpc(x, u, A, B, x_ref)
    x = evolve to the next state by solving the ODE using solve_ivp given x and u_optimal
```
```
# GET REFERENCE STATE
nearest_index = index of optimal track with the nearest position to current position
index = list from nearest_index to (nearest_index + n_horizon)
change all indices beyond available to last index
return all optimal track states for given index list
```
```
# GET LINEARIZED STATE
A = perturb each state, solve for the next state using IVP and calculate the finite difference
B = perturb each input, solve for the next state using IVP and calculate the finite difference
```
```
# MPC
initialize x optimization variable
initialize u optimization variable
cost = 0
x[0] == x_0
FOR t in n_horizon:
    du = u_last - u[t]
    cost += quadratic(x[t], Q) + quadratic(du, R)
    
    x[t + 1] == A x[t] + B u[u]
    capsule_height >= 0
    u[t] >= 0
    u[t] <= 10
    third input == gravity
solve for u
```