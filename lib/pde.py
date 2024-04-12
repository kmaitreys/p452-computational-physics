"""
This module hosts the specific PDE solvers and integrators.
For example, the heat diffusion equation solver and the 
solvers for the Laplace and Poisson equations.
"""

from typing import Callable, List, Tuple

import matplotlib.pyplot as plt

from .array import Array, zeros
from .matrix import Matrix


class HeatDiffusion:
    """
    This class models the heat diffusion equation
    by the Crank-Nicolson method.

    Parameters
    ----------
    initial_condition : Callable
        The initial condition of the heat diffusion equation.

    diffusion_coefficient : float
        The diffusion coefficient of the heat diffusion equation.

    space_bounds : Tuple[float, float]
        The spatial bounds of the heat diffusion equation.

    time_bounds : Tuple[float, float]
        The time bounds of the heat diffusion equation.

    Nx : int
        The number of spatial points.

    Nt : int
        The number of time points.

    Methods
    -------
    boundary_conditions(t)
        The boundary conditions of the heat diffusion equation.

    crank_nicolson()
        The Crank-Nicolson method for solving the heat diffusion equation.

    thomas_algorithm(A, b)
        The Thomas algorithm for solving tridiagonal matrices.

    solve()
        Solve the heat diffusion equation.

    tabulate_solution(at_time)
        Tabulate the solution at specific time points.

    plot_solution()
        Plot a heatmap of the solution.

    """

    def __init__(
        self,
        initial_condition: Callable,
        diffusion_coefficient: float,
        space_bounds: Tuple[float, float],
        time_bounds: Tuple[float, float],
        Nx: int,
        Nt: int,
    ) -> None:
        self.initial_condition = initial_condition
        self.diffusion_coefficient = diffusion_coefficient
        self.space_bounds = space_bounds
        self.time_bounds = time_bounds
        self.Nx = Nx
        self.Nt = Nt

        self.h = (self.space_bounds[1] - self.space_bounds[0]) / self.Nx
        self.k = (self.time_bounds[1] - self.time_bounds[0]) / self.Nt
        self.alpha = self.diffusion_coefficient * self.k / self.h**2

        self.u = Matrix(self.Nt, self.Nx)

    def boundary_conditions(self, t):
        """
        The boundary conditions of the heat diffusion equation.
        """
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
            b = zeros("d", self.Nx)
            for j in range(1, self.Nx - 1):
                b[j] = self.u[n, j] + self.alpha * (
                    self.u[n, j + 1] - 2 * self.u[n, j] + self.u[n, j - 1]
                )
            b[0] = self.boundary_conditions(self.time_bounds[0] + n * self.k)
            b[-1] = self.boundary_conditions(self.time_bounds[0] + n * self.k)
            self.u[n + 1] = self.thomas_algorithm(A, b)

    def thomas_algorithm(self, A: Matrix, b: Array):
        """
        Native Python implementation of the Thomas algorithm.
        It solves tridiagonal matrices of the form Ax = b.
        """
        N = len(b)
        c = zeros("d", N - 1)
        d = zeros("d", N)
        x = zeros("d", N)

        # Forward elimination
        c[0] = A[0, 1] / A[0, 0]
        d[0] = b[0] / A[0, 0]
        for i in range(1, N - 1):
            c[i] = A[i, i + 1] / (A[i, i] - A[i, i - 1] * c[i - 1])
        for i in range(1, N):
            d[i] = (b[i] - A[i, i - 1] * d[i - 1]) / (A[i, i] - A[i, i - 1] * c[i - 1])
        x[N - 1] = d[N - 1]
        for i in range(N - 1, 0, -1):
            x[i - 1] = d[i - 1] - c[i - 1] * x[i]
        return x

    def solve(self):
        for i in range(self.Nx):
            self.u[0, i] = self.initial_condition(self.space_bounds[0] + i * self.h)

        self.crank_nicolson()

    def tabulate_solution(self, at_time: List[float]):
        """
        Tabulate the solution at specific time points.
        """
        # Create a table so the output looks pretty
        print(f"{'Time':<10}{'Space':<10}{'Temperature':<10}")
        for t in at_time:
            n = int((t - self.time_bounds[0]) / self.k)
            sol = self.u[n]
            for i in range(self.Nx):
                # Print in scientific notation
                print(
                    f"{t:<10.2f}{self.space_bounds[0] + i * self.h:<10.2f}{sol[i]:<10.2e}"
                )
            print()

    def plot_solution(self):
        """
        Plot a heatmap of the solution.
        The time dimension is on the X-axis and
        the spatial dimension is on the Y-axis.
        This also overlays a contour plot of the solution.
        """
        # Set figure size
        plt.figure(figsize=(8, 6))

        # Scale the axes for correct units
        plt.xticks(
            ticks=[0, self.Nt - 1],
            labels=[self.time_bounds[0], self.time_bounds[1]],
        )
        # Plot y ticks reversed
        plt.yticks(
            ticks=[self.Nx - 1, 0],
            labels=[self.space_bounds[1], self.space_bounds[0]],
        )

        # Plot the heatmap
        plt.imshow(self.u.transpose().data, cmap="hot", interpolation="nearest")

        # Overlay the contour plot
        # plt.contour(self.u.transpose().data, cmap="hot")

        # Set the x-axis and y-axis labels
        plt.xlabel("Time")
        plt.ylabel("Space")

        # Set the title of the plot
        plt.title("Heat diffusion")

        # Show the colorbar
        plt.colorbar()

        # Show the plot
        plt.show()

class LaplaceEquation:
    pass
class PoissonEquation:
    """
    This class models the Poisson equation
    by the finite difference method.

    The Dirichlet boundary conditions are assumed.

    TODO: Add Neumann boundary conditions and extend the class.
    """

    def __init__(
        self,
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float],
        M: int,
        N: int,
    ):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.M = M
        self.N = N

        self.m = M + 1
        self.n = N + 1

        self.mn = self.m * self.n

        self.h = (self.x_bounds[1] - self.x_bounds[0]) / self.M
        self.k = (self.y_bounds[1] - self.y_bounds[0]) / self.N

        self.u = zeros("d", self.mn)

    def intial_condition(self, x, y):
        return 0

    def dirichlet_boundary_conditions(
        self, left: Callable, right: Callable, bottom: Callable, top: Callable
    ):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def thomas_algorithm(self, A: Matrix, b: Array):
        """
        Native Python implementation of the Thomas algorithm.
        It solves tridiagonal matrices of the form Ax = b.
        """
        N = len(b)
        c = zeros("d", N - 1)
        d = zeros("d", N)
        x = zeros("d", N)

        # Forward elimination
        c[0] = A[0, 1] / A[0, 0]
        d[0] = b[0] / A[0, 0]
        for i in range(1, N - 1):
            c[i] = A[i, i + 1] / (A[i, i] - A[i, i - 1] * c[i - 1])
        for i in range(1, N):
            d[i] = (b[i] - A[i, i - 1] * d[i - 1]) / (A[i, i] - A[i, i - 1] * c[i - 1])
        x[N - 1] = d[N - 1]
        for i in range(N - 1, 0, -1):
            x[i - 1] = d[i - 1] - c[i - 1] * x[i]
        return x

    def solve(self):
        # Set the mesh
        import numpy as np
        x = np.linspace(self.x_bounds[0], self.x_bounds[1], self.m)
        y = np.linspace(self.y_bounds[0], self.y_bounds[1], self.n)

        A = Matrix(self.mn, self.mn)
        b = zeros("d", self.mn)

        for i in range(1, self.m - 1):
            for j in range(1, self.n - 1):
                A[i + (j - 1) * self.m, i - 1 + (j - 1) * self.m] = 1 / self.h**2
                A[i + (j - 1) * self.m, i + 1 + (j - 1) * self.m] = 1 / self.h**2
                A[i + (j - 1) * self.m, i + (j - 1) * self.m] = (
                    -2 / self.h**2 - 2 / self.k
                )
                A[i + (j - 1) * self.m, i + (j - 2) * self.m] = 1 / self.k**2
                A[i + (j - 1) * self.m, i + j * self.m] = 1 / self.k**2
                b[i + (j - 1) * self.m] = self.intial_condition(x[i], y[j])

        for i in range(self.m):
            j = 0
            A[i + (j - 1) * self.m, i + (j - 1) * self.m] = 1
            b[i + (j - 1) * self.m] = self.bottom(x[i])
            j = self.n - 1
            A[i + (j - 1) * self.m, i + (j - 1) * self.m] = 1
            b[i + (j - 1) * self.m] = self.top(x[i])

        for j in range(1, self.n - 1):
            i = 0
            A[i + (j - 1) * self.m, i + (j - 1) * self.m] = 1
            b[i + (j - 1) * self.m] = self.left(y[j])
            i = self.m - 1
            A[i + (j - 1) * self.m, i + (j - 1) * self.m] = 1
            b[i + (j - 1) * self.m] = self.right(y[j])

        self.u = self.thomas_algorithm(A, b)

    def tabulate_solution(self):
        """
        Tabulate the solution at specific time points.
        """
        # Create a table so the output looks pretty
        print(f"{'Space':<10}{'Space':<10}{'Temperature':<10}")
        for j in range(self.n):
            for i in range(self.m):
                # Print in scientific notation
                print(
                    f"{self.x_bounds[0] + i * self.h:<10.2f}"
                    f"{self.y_bounds[0] + j * self.k:<10.2f}"
                    f"{self.u[i + j * self.m]:<10.2e}"
                )
            print()

    def plot_solution(self):
        # Make a 3D surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        import numpy as np
       
        x = np.linspace(self.x_bounds[0], self.x_bounds[1], self.m)
        y = np.linspace(self.y_bounds[0], self.y_bounds[1], self.n)

        # Create a meshgrid
        X, Y = np.meshgrid(x, y)

        self.u = np.array(self.u)

        # Plot the surface
        ax.plot_surface(X, Y, self.u.reshape(self.m, self.n).transpose().data)

        # Set the x-axis and y-axis labels
        ax.set_xlabel("Space")
        ax.set_ylabel("Time")

        # Set the title of the plot
        ax.set_title("Poisson equation")

        # Show the plot
        plt.show()


class WaveEquation:
    pass