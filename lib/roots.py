"""
This module contains functions to
find the roots of non-linear equations
using various methods.
"""

from typing import Callable


def fixed_point(func: Callable, x0: float, tol: float, maxiter: int) -> float:
    """
    Find the root of a non-linear equation using the fixed point method.

    Parameters
    ----------
    func : Callable
        The function whose root is to be found.

    x0 : float
        The initial guess for the root.

    tol : float
        The tolerance for the root.

    maxiter : int
        The maximum number of iterations to perform.

    Returns
    -------
    float
        The root of the function.
    """

    def g(x):
        return x + func(x)

    x = x0
    for _ in range(maxiter):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return print(f"{func.__name__} has a root at {x_new}.")
        x = x_new
    raise RuntimeError(
        "Failed to converge. Try increasing the maximum number of iterations."
    )
