"""
This module hosts the different ODE solvers and integrators.
"""


from typing import Callable, Tuple

from .datamodels import Array


def forward_euler():
    pass

def backward_euler():
    pass

def predictor_corrector():
    pass

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

    time_steps = Array.zeros("d", num_steps + 1)
    y_evals = Array.zeros("d", num_steps + 1)

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

def semi_implicit_euler():
    pass

def verlet():
    pass

def leap_frog():
    pass

def shooting_method():
    pass

def finite_difference():
    pass

