import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np

# Build parametric optimizer
# ------------------------------------
nu, nx, N, L, ts = 2, 3, 20, 0.5, 0.1
xref, yref, thetaref = 1, 1, 0
Q, Qtheta, R, Qf, Qthetaf = 10, 0.1, 1, 200, 2


def dynamics_dt(x, y, theta, u):
    theta_dot = (1/L) * (u[1] * cs.cos(theta) - u[0] * cs.sin(theta))
    # cost += r * cs.dot(u, u)
    x += ts * (u[0] + L * cs.sin(theta) * theta_dot)
    y += ts * (u[1] - L * cs.cos(theta) * theta_dot)
    theta += ts * theta_dot
    return x, y, theta


def stage_cost(x, y, theta, z_ref, u):
    return Q*((x-z_ref[0])**2 + (y-z_ref[1])**2) + Qtheta*(theta-z_ref[2])**2 + R * cs.dot(u, u)


def terminal_cost(x, y, theta, z_ref):
    return Qf*((x-z_ref[0])**2 + (y-z_ref[1])**2) + Qthetaf*(theta-z_ref[2])**2


u = cs.SX.sym('u', nu*N)
z0 = cs.SX.sym('z0', nx)
z_ref = cs.SX.sym('z_ref', nx)

(x, y, theta) = (z0[0], z0[1], z0[2])

cost = 0
for t in range(0, nu*N, nu):
    u_t = u[t:t+2]
    cost += stage_cost(x, y, theta, z_ref, u_t)
    x, y, theta = dynamics_dt(x, y, theta, u_t)

cost += terminal_cost(x, y, theta, z_ref)

umin = [-3.0] * (nu*N)
umax = [3.0] * (nu*N)
bounds = og.constraints.Rectangle(umin, umax)

problem = og.builder.Problem(u, cs.vcat(
    [z0, z_ref]), cost).with_constraints(bounds)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("my_optimizers")\
    .with_build_mode("debug")\
    .with_tcp_interface_config()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("navigation")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-5)
builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config)
builder.build()
# quit()

mng = og.tcp.OptimizerTcpManager('my_optimizers/navigation')
mng.start()

mng.ping()
x_init = [-1.0, 2.0, 0.0, xref, yref, thetaref]
solution = mng.call(x_init, initial_guess=[1.0] * (nu*N))
mng.kill()


# Plot solution
# ------------------------------------
time = np.arange(0, ts*N, ts)
u_star = solution['solution']
ux = u_star[0:nu*N:2]
uy = u_star[1:nu*N:2]

plt.subplot(211)
plt.plot(time, ux, '-o')
plt.ylabel('u_x')
plt.subplot(212)
plt.plot(time, uy, '-o')
plt.ylabel('u_y')
plt.xlabel('Time')
plt.show()
