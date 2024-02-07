"""
This module contains functions to
plotting utilities for the computational
physics course.

It is basically a wrapper around the
`matplotlib` library.
"""

from typing import Callable

import matplotlib.pyplot as plt

from .datamodels import Array


def plot(func: Callable, bounds: tuple[float, float], num_points: int, title: str):
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
    print(f"Minimum value of y: {min(y)}")
    print(f"Maximum value of y: {max(y)}")
    plt.plot(x, y)
    plt.title(title)
    plt.show()
