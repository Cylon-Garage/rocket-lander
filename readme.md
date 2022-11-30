# Rocket Landing Trajectory Tracking with Linear MPC

## Dynamic Equations
<figure>
<p align="center">
<img src="https://github.com/Cylon-Garage/rocket-lander/blob/master/freebody.svg?raw=true" alt="Trulli" style="width:65%">
</p>
<figcaption align="center"><b>Dragon Capsule Freebody Diagram</b></figcaption>
</figure>

$$
\begin{align*}
\sum F_x &= m \ddot{x} = -(F_1 + F_2) \cos\theta \\
\sum F_y &= m \ddot{y} = -mg + (F_1 + F_2) \sin\theta \\
\sum M_{cg} &= I \ddot{\theta} =  L_{thrust} (F_2 - F_1)
\end{align*}
$$

## State Equations
$$
\begin{align*}
\underline{x} &= \begin{bmatrix} x \enspace y \enspace \theta \enspace \dot{x} \enspace \dot{y} \enspace \dot{\theta}\end{bmatrix}^T \\
\underline{u} &= \begin{bmatrix} F_1 \enspace F_2 \end{bmatrix}^T \\
\end{align*}
$$
$$
f(\underline{x}, \underline{u}) = \dot{\underline{x}} = \begin{bmatrix}
x_4 \\ x_5 \\ x_6 \\
\frac{-(F_1 + F_2)}{m}  \sin x_3 \\
-g + \frac{-(F_1 + F_2)}{m}  \cos x_3 \\
\frac{L_{thruster}}{I} * (F_2 - F_1)
\end{bmatrix}
$$

## Linearization

## Video
https://user-images.githubusercontent.com/100429965/204759660-715809cd-440b-4427-94ce-727bdb4b9641.mp4
## TBD
$$
\underline{u} = \begin{bmatrix} F_1 \enspace F_2 \enspace g \end{bmatrix}^T
$$


$F_1$

$F_2$

$mg$

$L_{thruster}$

$\theta$


