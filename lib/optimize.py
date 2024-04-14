"""
Curve-fitting, regression, and optimization functions.
"""

from typing import Iterable

from .array import Array, ones, zeros
from .linalg import LUSolve
from .matrix import Matrix


def polyfit(x: Array, y: Array, degree: int):
    degree += 1

    # Define coefficient matrix
    A = Matrix(degree, degree)
    vec = zeros("d", degree)

    for i in range(degree):
        for j in range(degree):
            A[i, j] = sum(x ** (i + j))

    det_A = A.det()

    if det_A == 0:
        raise ValueError("Matrix is singular")

    for i in range(degree):
        vec[i] = sum(y * x**i)

    # Solve the system of equations
    sol = LUSolve(A, vec, method="doolittle")
    x = sol.solve()

    A_inv = A.invert()

    return x[:degree], A_inv


def get_condition_number(A: Matrix):
    """
    Compute the condition number of a matrix
    according to the Frobenius norm.
    """
    cond = A.norm() * A.invert().norm()
    return cond


def get_chebyshev_poly(x: float, degree: int):
    if degree == 0:
        if isinstance(x, Iterable):
            return ones("d", len(x))
        else:
            return 1.0
    elif degree == 1:
        return 2 * x - 1
    elif degree == 2:
        return 8 * x**2 - 8 * x + 1
    elif degree == 3:
        return 32 * x**3 - 48 * x**2 + 18 * x - 1
    elif degree == 4:
        return 128 * x**4 - 256 * x**3 + 160 * x**2 - 32 * x + 1


def chebyshev(x: Array, y: Array, degree: int):
    """
    Fit a polynomial to a set of data using the Chebyshev method.
    """
    degree += 1

    # Define coefficient matrix
    A = Matrix(degree, degree)
    vec = zeros("d", degree)

    for i in range(degree):
        for j in range(degree):
            coeff = get_chebyshev_poly(x, i) * get_chebyshev_poly(x, j)
            if not isinstance(coeff, Iterable):
                coeff = [coeff]
            tot = sum(coeff)
            A[i, j] = tot

    for i in range(degree):
        vec[i] = sum(y * get_chebyshev_poly(x, i))

    det_A = A.det()

    if det_A == 0:
        raise ValueError("Matrix is singular")

    # Solve the system of equations
    A_inv = A.invert()
    x = A_inv * vec

    return x, A_inv
