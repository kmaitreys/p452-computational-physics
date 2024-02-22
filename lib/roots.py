"""
This module contains functions to
find the roots of non-linear equations
using various methods.
"""

from typing import Callable, Tuple


def derivative(func: Callable, x: float, order: int = 1, h: float = 1e-8) -> float:
    if order == 1:
        dfunc = (func(x + h) - func(x)) / h
        return dfunc
    elif order == 2:
        d2func = (func(x + h) - 2 * func(x) + func(x - h)) / h**2
        return d2func


def bisection(func: Callable, bounds: Tuple[float, float], max_iter):
    """Approximate solution of f(x)=0 on interval [a,b] by bisection method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x)=0.
    bounds : tuple
        The interval [a,b] in which to search for a solution. The function
        returns None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    max_iter : (positive) integer
        The number of iterations to implement.

    Returns
    -------
    x_N : number
        The midpoint of the Nth interval computed by the bisection method. The
        initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
        midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iteration, the bisection method fails and return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> bisection(f,1,2,25)
    1.618033990263939
    >>> f = lambda x: (2*x - 1)*(x - 3)
    >>> bisection(f,0,1,10)
    0.5
    """
    a, b = bounds
    if func(a) * func(b) >= 0:
        print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1, max_iter + 1):
        m_n = (a_n + b_n) / 2
        f_m_n = func(m_n)
        if func(a_n) * f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif func(b_n) * f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        else:
            print("Bisection method fails.")
            return None
    return (a_n + b_n) / 2


def regula_falsi(func: Callable, bounds: Tuple[float, float], tol=1e-6, max_iter=1000):
    """
    Regula Falsi method to find root of a function.

    Parameters
    ----------
    func : function
        The function for which roots need to be found.
    bounds : tuple
        The interval in which to search for a solution. The function
        returns None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    tol : float
        The tolerance for the root.
    max_iter : int
        The maximum number of iterations to perform.

    Returns
    -------
    float
        The root of the function.
    """
    a, b = bounds
    if func(a) * func(b) >= 0:
        raise ValueError("Function has same signs at both ends of the interval.")

    for _ in range(max_iter):
        c = (a * func(b) - b * func(a)) / (func(b) - func(a))
        if abs(func(c)) < tol:
            return c
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c

    raise ValueError("Method did not converge within maximum iterations.")


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


def newton_raphson(
    func: Callable, x0: float, tol: float = 1e-6, maxiter: int = 1000
) -> float:
    """
    The Newton-Raphson method to find the root of a function.
    """
    x_n = x0
    k = 0
    x = x0 - func(x0) / derivative(func, x0, order=1)
    while abs(x - x_n) > tol and k < maxiter:
        x_n = x
        x = x - func(x) / derivative(func, x, order=1)
        k += 1

    return x


def secant(func: Callable, x0: float, x1: float, tol: float = 1e-8, maxiter: int = 100):
    """
    Secant method to find the root of a function.
    """
    x_prev = x0
    x_curr = x1

    for _ in range(maxiter):
        f_prev = func(x_prev)
        f_curr = func(x_curr)

        if abs(f_curr) < tol:
            return x_curr

        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)

        if abs(x_next - x_curr) < tol:
            return x_next
    
        x_prev = x_curr
        x_curr = x_next
    
    raise ValueError("Method did not converge within maximum iterations.")


def broyden():
    pass


def brent():
    pass
