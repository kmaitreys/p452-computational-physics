"""
DEPRECATED: This module is deprecated and will be removed in the future.

This module contains functions to
plotting utilities for the computational
physics course.

It is basically a wrapper around the
`matplotlib` library.
"""

from typing import Callable, Tuple

import matplotlib.pyplot as plt

from .array import Array, linspace, zeros
from .optimize import get_chebyshev_poly


def plot_data(
    x: Array,
    y: Array,
    title,
):
    """
    Plot two arrays.
    """
    plt.plot(x, y)
    plt.title(title)
    # plt.show()


def plot_function(
    func: Callable, bounds: Tuple[float, float], num_points: int, title: str
):
    """
    Plot a function using matplotlib.

    Parameters
    ----------
    func : Callable
        The function to plot.

    bounds : tuple[float, float]
        The lower and upper bounds of the x-axis.

    num_points : int
        The number of points to plot.

    title : str
        The title of the plot.
    """

    x = linspace("d", bounds[0], bounds[1], num_points)
    y = zeros("d", num_points)
    for i in range(num_points):
        y[i] = func(x[i])

    print(f"Minimal value of y: {min(y.abs())}")
    print(f"Maximal value of y: {max(y.abs())}")
    plt.plot(x, y)
    plt.title(title)
    plt.show()

def plot_poly(x: Array, y: Array, sol: Array, degree: int):
    """
    Plot a polynomial fit to a set of data.
    """
    y_fit = zeros("d", len(x))
    for i in range(len(x)):
        for j in range(degree+1):
            y_fit[i] += sol[j] * x[i] ** j

    plt.plot(x, y, "o",label="Data")
    plt.plot(x, y_fit, label="Fit")
    plt.legend()
    plt.grid()

def plot_chebyshev(x, sol):
    y_fit = zeros("d", len(x))

    for i in range(len(x)):
        for j in range(len(sol)):
            y_fit[i] += sol[j] * get_chebyshev_poly(x[i], j)

    plt.plot(x, y_fit, label="Fit")
    plt.legend()
    plt.grid()
        
