"""
This module hosts the different ODE solvers and integrators.
"""


from typing import Callable, Tuple

from .array import Array, zeros
from .roots import newton_raphson


def forward_euler(
    dfunc: Callable,
    a: float,
    b: float,
    step_size: float = 0.1,
    limit: int = 100,
):
    """
    Forward Euler scheme.
    """
    t = [a]
    y = [b]
    while a <= limit:
        k1 = step_size * dfunc(a, b)
        b = b + k1
        a = a + step_size
        t.append(a)
        y.append(b)

    return t, y


def backward_euler(
    dfunc: Callable,
    a: float,
    b: float,
    step_size: float = 0.1,
    limit: int = 100,
):
    """
    Backward Euler scheme.
    """
    N = int(limit / step_size)
    t = [a]
    y = [b]
    t_n = a
    y_n = b

    for i in range(N):

        def func(y_n1):
            return y_n + step_size * dfunc(t_n + step_size, y_n1) - y_n1

        y_nr = newton_raphson(func, y_n)
        y_n1 = y_n + step_size * dfunc(t_n + step_size, y_nr)
        t.append(t_n + step_size)
        y.append(y_n1)
        t_n = t_n + step_size
        y_n = y_n1

    return t, y


def predictor_corrector(
    dfunc: Callable,
    a: float,
    b: float,
    step_size: float = 0.1,
    limit: int = 100,
):
    """
    Predictor-Corrector scheme.
    """
    t = [a]
    y = [b]

    while a <= limit:
        k1 = step_size * dfunc(a, b)
        k2 = step_size * dfunc(a + step_size, b + k1)
        b = b + 0.5 * (k1 + k2)
        a = a + step_size
        t.append(a)
        y.append(b)

    return t, y


def runge_kutta_4th_order(
    func: Callable,
    t0: float,
    y0: float | Array,
    interval: Tuple[float, float],
    step_size: float = 1e-4,
):
    """
    The RK4 Runge-Kutta ODE solver
    """
    t_start, t_end = interval

    num_steps = int((t_end - t_start) / step_size)

    time_steps = zeros("d", num_steps + 1)
    y_evals = zeros("d", num_steps + 1)

    time_steps[0] = t0
    y_evals[0] = y0

    t = t0
    y = y0

    for i in range(num_steps):
        k1 = step_size * func(t, y)
        k2 = step_size * func(t + 0.5 * step_size, y + 0.5 * k1)
        k3 = step_size * func(t + 0.5 * step_size, y + 0.5 * k2)
        k4 = step_size * func(t + step_size, y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + step_size
        time_steps[i + 1] = t
        y_evals[i + 1] = y

    return time_steps, y_evals


# TODO: Higher order Runge-Kutta methods


def symplectic_euler(
    hamiltonian_gradient: Callable, q0, p0, step_size: float = 0.1, num_steps: int = 100
):
    """
    Symplectic Euler scheme.
    """
    q_vals = [[0.0] * len(q0) for _ in range(num_steps)]
    p_vals = [[0.0] * len(p0) for _ in range(num_steps)]

    q_vals[0] = q0[:]
    p_vals[0] = p0[:]

    for i in range(num_steps):
        p_vals[i + 1] = [
            p_i + step_size * gh_i
            for p_i, gh_i in zip(p_vals[i], hamiltonian_gradient(q_vals[i]))
        ]
        q_vals[i + 1] = [
            q_i + step_size * p_i for q_i, p_i in zip(q_vals[i], p_vals[i + 1])
        ]

    return q_vals, p_vals


def semi_implicit_euler(
    f1: Callable,
    f2: Callable,
    x0: float,
    y0: float,
    t0: float,
    dt: float,
    num_steps: int,
):
    """
    Semi-implicit Euler scheme.
    """
    X = []
    Y = []
    x = x0
    y = y0
    time_vals = [t0 + i * dt for i in range(num_steps)]

    for i in range(num_steps):
        X.append(x)
        Y.append(y)
        x += dt * f1(t0 + i * dt, y)
        y += dt * f2(t0 + i * dt, x)

    return time_vals, X, Y


def verlet(acceleration: Callable, x0, v0, dt, num_steps, t0=0):
    """
    Verlet scheme.
    """
    # Initialize the position and velocity
    X = [x0]
    T = [t0 + i * dt for i in range(num_steps)]

    # First step
    X.append(x0 + v0 * dt + 0.5 * acceleration(x0) * dt**2)

    # Subsequent steps
    for i in range(num_steps - 2):
        X.append(2 * X[-1] - X[-2] + acceleration(X[-1]) * dt**2)

    return T, X


def velocity_verlet(acceleration: Callable, x0, v0, dt, num_steps, t0=0):
    """
    Velocity Verlet scheme.
    """
    X = [x0]
    V = [v0]
    T = [t0 + i * dt for i in range(num_steps)]

    for i in range(num_steps - 1):
        # Update the position
        x_new = X[-1] + V[-1] * dt + 0.5 * acceleration(X[-1]) * dt**2

        # Update the velocity
        v_new = V[-1] + 0.5 * (acceleration(X[-1]) + acceleration(x_new)) * dt

        X.append(x_new)
        V.append(v_new)

    return T, X, V


def leap_frog(func: Callable, x0, p0, dt, tau, t0=0):
    """
    Leap-frog scheme for Molecular Dynamics.
    """
    X = [x0]
    P = [p0]
    n = int((tau - t0) / dt)

    # Generate the time steps
    Tx = [t0 + i * (tau - t0) / (n - 1) for i in range(n)]
    Tp = [t + 0.5 * dt for t in Tx]

    # Initialize the position and momentum
    P.append(P[-1] + 0.5 * dt * func(t0))

    # Leap-frog steps
    for i in range(1, n-1):
        X.append(X[-1] + dt * P[-1])
        P.append(P[-1] + dt * func(t0 + i * dt))
    
    # Final step
    X.append(X[-1] + dt * P[-1])
    
    # Final momentum
    p_fin = P[-1] + 0.5 * dt * func(tau -0.5 * dt)

    return Tx, Tp, X, P, p_fin


def shooting_method():
    pass


def finite_difference():
    pass
