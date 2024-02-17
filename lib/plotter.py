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

from .datamodels import Array


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

    x = Array.linspace("d", bounds[0], bounds[1], num_points)
    y = Array.zeros("d", num_points)
    for i in range(num_points):
        y[i] = func(x[i])

    print(f"Minimal value of y: {min(y.abs())}")
    print(f"Maximal value of y: {max(y.abs())}")
    plt.plot(x, y)
    plt.title(title)
    plt.show()
