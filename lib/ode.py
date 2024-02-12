"""
This module hosts the different ODE solvers and integrators.
"""


from typing import Callable, Tuple

import matplotlib.pyplot as plt

from .datamodels import Array, Matrix


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


class HeatDiffusion:
    """
    This class models the heat diffusion equation
    by the Crank-Nicolson method.
    """

    def __init__(
        self,
        diffusion_coefficient: float,
        space_bounds: Tuple[float, float],
        time_bounds: Tuple[float, float],
        Nx: int,
        Nt: int,
    ) -> None:
        self.space_bounds = space_bounds
        self.time_bounds = time_bounds
        self.Nx = Nx
        self.Nt = Nt
        self.m = Nx - 1
        self.n = Nt
        self.diffusion_coefficient = diffusion_coefficient
        self.h = (space_bounds[1] - space_bounds[0]) / Nx
        self.k = (time_bounds[1] - time_bounds[0]) / Nt
        self.alpha = self.diffusion_coefficient * self.k / (self.h**2)
        self.u = Matrix(Nt, Nx)

    @staticmethod
    def initial_condition(func: Callable, x: float):
        return func(x)

    def boundary_conditions(self, t):
        return 0

    def crank_nicolson(self):
        """
        The Crank-Nicolson method for solving the heat diffusion equation.
        """
        A = Matrix(self.Nx, self.Nx)
        for i in range(self.Nx):
            A[i, i] = 2 + 2 * self.alpha
            if i > 0:
                A[i, i - 1] = -self.alpha
            if i < self.Nx - 1:
                A[i, i + 1] = -self.alpha

        for n in range(self.Nt - 1):
            b = Array.zeros("d", self.Nx)
            for j in range(1, self.Nx - 1):
                b[j] = self.u[n, j] + self.alpha * (
                    self.u[n, j + 1] - 2 * self.u[n, j] + self.u[n, j - 1]
                )
            b[0] = self.boundary_conditions(self.time_bounds[0] + n * self.k)
            b[-1] = self.boundary_conditions(self.time_bounds[0] + n * self.k)
            self.u[n + 1] = self.thomas_algorithm(A, b)

    def thomas_algorithm(self, A: Matrix, b: Array):
        # Implementation of the Thomas algorithm for solving tridiagonal matrices
        N = len(b)
        c = Array.zeros("d", N - 1)
        d = Array.zeros("d", N)
        x = Array.zeros("d", N)

        # Forward elimination
        c[0] = A[0, 1] / A[0, 0]
        d[0] = b[0] / A[0, 0]
        for i in range(1, N - 1):
            m = 1 / (A[i, i] - A[i, i - 1] * c[i - 1])
            c[i] = A[i, i + 1] * m
            d[i] = (b[i] - A[i, i - 1] * d[i - 1]) * m

        # Back substitution
        d[N - 1] = (b[N - 1] - A[N - 1, N - 2] * d[N - 2]) / (
            A[N - 1, N - 1] - A[N - 1, N - 2] * c[N - 2]
        )
        x[N - 1] = d[N - 1]
        for i in range(N - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]

        return x

    def solve(self, func: Callable):
        for i in range(self.Nx):
            self.u[0, i] = self.initial_condition(
                func, self.space_bounds[0] + i * self.h
            )

        self.crank_nicolson()

    def tabulate_solution(self):
        return print(self.u)

    def plot_solution(self):
        plt.imshow(self.u.data, cmap="hot", interpolation="nearest")
        plt.show()
