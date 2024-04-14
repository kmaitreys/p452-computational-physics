"""
This module contains functions to
integrate functions using various methods.

The following methods are available:
- Trapezoidal Rule
- Simpson's Rule
- Gaussian Quadrature
"""

import math

# import random
from typing import Callable, Generator, Tuple

from lib.array import arange
from lib.random import _lcg_generator


def get_max_steps(func, a, b, tol, method, *args):
    """
    Get the maximum number of steps to use in the integration.
    """
    func_max = get_function_maximum_derivative(func, a, b, *args)

    # Calculation of N from error calculation formula
    if method == "midpoint":
        N_mp = int(((b - a) ** 3 / 24 / tol * func_max) ** 0.5)

        if N_mp == 0:
            N_mp = 1

        return N_mp
    elif method == "trapezoidal":
        N_t = int(((b - a) ** 3 / 12 / tol * func_max) ** 0.5)

        if N_t == 0:
            N_t = 1

        return N_t


def get_function_maximum_derivative(func, a, b, order=2, *args):
    h = (b - a) / 1000
    x = [a + i * h for i in range(1000)]
    y = []

    if order == 2:
        for i in range(len(x)):
            # calculate the 2nd derivative of f(x) using the central difference method
            y.append(
                abs(
                    (
                        func(x[i] + h, *args)
                        - 2 * func(x[i], *args)
                        + func(x[i] - h, *args)
                    )
                    / h**2
                )
            )

        return max(y)

    if order == 4:
        for i in range(len(x)):
            # calculate the 4th derivative of f(x) using the central difference method
            y.append(
                abs(
                    (
                        func(x[i] + 2 * h, *args)
                        - 4 * func(x[i] + h, *args)
                        + 6 * func(x[i], *args)
                        - 4 * func(x[i] - h, *args)
                        + func(x[i] - 2 * h, *args)
                    )
                    / h**4
                )
            )

        return max(y)


# TODO: Encaspuale the following functions in a NewtonCotes class


def midpoint(func: Callable, bounds: Tuple[float, float], tol: float = 1e-6, *args):
    """
    Midpoint rule.
    """
    N = get_max_steps(func, bounds[0], bounds[1], tol, method="midpoint", *args)
    s = 0
    step_size = (bounds[1] - bounds[0]) / N

    for i in range(N):
        x = bounds[0] + (2 * i - 1) * step_size / 2
        s += func(x)

    sol = step_size * s

    return sol


def trapezoidal(func: Callable, bounds: Tuple[float, float], tol: float = 1e-6, *args):
    """
    Trapezoidal rule.
    """
    N = get_max_steps(func, bounds[0], bounds[1], tol, method="trapezoidal", *args)
    s = 0
    step_size = (bounds[1] - bounds[0]) / N

    for i in range(1, N + 1):
        s += func(bounds[0] + i * step_size) + func(bounds[0] + (i - 1) * step_size)

    sol = s * step_size / 2

    return sol


def simpson(
    func: Callable,
    bounds: Tuple[float, float],
    tol: float,
    maximum_value_of_error_func: float,
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
    a, b = bounds
    n = int(((b - a) ** 5 / 180 / tol * maximum_value_of_error_func) ** (1 / 4))

    if n % 2 != 0:
        n += 1

    h = (b - a) / n

    # create an array of x values
    x = arange("d", a, b + h, h)
    x[0] = func(x[0])
    x[-1] = func(x[-1])
    for i in range(1, len(x) - 1):
        if i % 2 == 0:
            x[i] = 2 * func(x[i])
        else:
            x[i] = 4 * func(x[i])

    return print(
        f"The value of the integral is {h / 3 * sum(x)} with {n} intervals used."
    )


def get_quadrature_method(bounds: tuple[float, float]) -> dict[str, Callable]:
    Legendre = {
        "p0": lambda x: 1,
        "p1": lambda x: x,
        "p2": lambda x: 1.5 * (x**2 - 1 / 3),
        "p3": lambda x: 2.5 * (x**3 - 3 / 5 * x),
        "p4": lambda x: (1 / 8) * (35 * x**4 - 30 * x**2 + 3),
        "p5": lambda x: (1 / 8) * (63 * x**5 - 70 * x**3 + 15 * x),
        "p6": lambda x: (1 / 16) * (231 * x**6 - 315 * x**4 + 105 * x**2 - 5),
        "p7": lambda x: (1 / 16) * (429 * x**7 - 693 * x**5 + 315 * x**3 - 35 * x),
        "p8": lambda x: (1 / 128)
        * (6435 * x**8 - 12012 * x**6 + 6930 * x**4 - 1260 * x**2 + 35),
        "p9": lambda x: (1 / 128)
        * (12155 * x**9 - 25740 * x**7 + 18018 * x**5 - 4620 * x**3 + 315 * x),
        "p10": lambda x: (1 / 256)
        * (
            46189 * x**10
            - 109395 * x**8
            + 90090 * x**6
            - 30030 * x**4
            + 3465 * x**2
            - 63
        ),
    }

    Laguerre = {
        "p0": lambda x: 1,
        "p1": lambda x: 1 - x,
        "p2": lambda x: (1 / 2) * (2 - 4 * x + x**2),
        "p3": lambda x: (1 / 6) * (6 - 18 * x + 9 * x**2 - x**3),
        "p4": lambda x: (1 / 24) * (24 - 96 * x + 72 * x**2 - 16 * x**3 + x**4),
        "p5": lambda x: (1 / 120)
        * (120 - 600 * x + 600 * x**2 - 200 * x**3 + 25 * x**4 - x**5),
        "p6": lambda x: (1 / 720)
        * (720 - 4320 * x + 5400 * x**2 - 2400 * x**3 + 450 * x**4 - 36 * x**5 + x**6),
    }

    Hermite = {
        "p0": lambda x: 1,
        "p1": lambda x: 2 * x,
        "p2": lambda x: 4 * x**2 - 2,
        "p3": lambda x: 8 * x**3 - 12 * x,
        "p4": lambda x: 16 * x**4 - 48 * x**2 + 12,
        "p5": lambda x: 32 * x**5 - 160 * x**3 + 120 * x,
        "p6": lambda x: 64 * x**6 - 480 * x**4 + 720 * x**2 - 120,
        "p7": lambda x: 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x,
        "p8": lambda x: 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680,
        "p9": lambda x: 512 * x**9
        - 9216 * x**7
        + 48384 * x**5
        - 80640 * x**3
        + 30240 * x,
        "p10": lambda x: 1024 * x**10
        - 23040 * x**8
        + 161280 * x**6
        - 403200 * x**4
        + 302400 * x**2
        - 30240,
    }
    if bounds == (-math.inf, math.inf):
        return Hermite
    elif bounds == (-1, 1):
        return Legendre
    elif bounds == (0, math.inf):
        return Laguerre
    else:
        raise ValueError(
            "Invalid limits of integration. "
            "Change the limits of integration to -inf and inf, -1 and 1, or 0 and inf."
        )


def gaussian_quadrature(func: Callable, bounds: Tuple[float, float]):
    if not (
        bounds == (-math.inf, math.inf) or bounds == (-1, 1) or bounds == (0, math.inf)
    ):
        raise ValueError(
            "Invalid limits of integration. "
            "Change the limits of integration to -inf and inf, -1 and 1, or 0 and inf."
        )

    # method = get_quadrature_method(bounds)


def monte_carlo_integration(
    rng: Generator, lower_bound, upper_bound, max_iter, func
):
    total = 0
    for _ in range(max_iter):
        x = next(rng) * (upper_bound - lower_bound) + lower_bound
        total += func(x)

    return (upper_bound - lower_bound) * total / max_iter


def monte_carlo_sampler(
    func: Callable,
    lower_bound: float,
    upper_bound: float,
    sample_range: Tuple,
    sample_step: float,
    rng_state: Tuple,
):
    sample_collections = list(range(sample_range[0], sample_range[1], sample_step))

    results = []

    for sample_size in sample_collections:
        random_genrator = _lcg_generator(rng_state[0], rng_state[1], rng_state[2], rng_state[3])
        result = monte_carlo_integration(
            random_genrator, lower_bound, upper_bound, sample_size, func
        )
        results.append(result)

    return sample_collections, results
