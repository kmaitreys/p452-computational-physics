"""
This module contains functions to
integrate functions using various methods.

The following methods are available:
- Trapezoidal Rule
- Simpson's Rule
- Gaussian Quadrature
"""
from typing import Callable

from .datamodels import Array


def simpson(
    func: Callable,
    a: float,
    b: float,
    n: int = 10000,
    maximum_value_of_error_func: float | None = None,
) -> float:
    """
    Integrate a function using Simpson's rule.

    Parameters
    ----------
    func : Callable
        The function to integrate.

    a : float
        The lower limit of integration.

    b : float
        The upper limit of integration.

    n : int
        The number of intervals to use.

    Returns
    -------
    float
        The value of the integral.
    """
    if n % 2 != 0:
        n += 1

    h = (b - a) / n

    # create an array of x values
    x = Array.arange("d", a, b + h, h)
    for i in range(len(x)):
        if i == 0 or i == len(x) - 1:
            x[i] = func(x[i])
        if i % 2 == 0:
            x[i] = 2 * func(x[i])
        else:
            x[i] = 4 * func(x[i])

    if maximum_value_of_error_func is None:
        return print(f"The value of the integral is {h / 3 * sum(x)}.")
    else:
        max_error = ((b - a) ** 5) / 180 / n**4 * maximum_value_of_error_func

        return print(
            f"The value of the integral is {h / 3 * sum(x)} "
            f"with an error of {max_error}."
        )
