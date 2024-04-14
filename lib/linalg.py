import math
from copy import deepcopy
from math import sqrt
from typing import Callable

import matplotlib.pyplot as plt

from .array import Array, inner, norm, ones, zeros
from .matrix import Matrix


class GaussJordan:
    """
    Solve a system of linear equations using the Gauss-Jordan Elimination method.
    The system is represented as `Ax = b`, where `A` is a matrix of coefficients,
    `x` is the vector of unknowns, which we want to solve for, and `b` is the vector
    of constants.

    Attributes
    ----------
    matrix : Matrix
        The matrix of coefficients
    vector : Array
        The vector of constants


    Methods
    -------
    solve()
        Solve the system of linear equations using the Gauss-Jordan Elimination method.


    """

    def __init__(self, matrix: Matrix, vector: Array):
        self.matrix = matrix
        self.vector = vector
        self.augmat = Matrix(matrix.nrows, matrix.ncols + 1)
        for i in range(matrix.nrows):
            for j in range(matrix.ncols):
                self.augmat.data[i][j] = matrix.data[i][j]
            self.augmat.data[i][matrix.ncols] = vector[i]

    def solve(self):
        i = 0
        n = self.matrix.nrows
        m = n - 1
        solution = zeros("d", n)
        # Gaussian elimination
        while i < n:
            # Partial pivoting
            for p in range(i + 1, n):
                if abs(self.augmat[i, i] < abs(self.augmat[p, i])):
                    # Swap rows
                    self.augmat[i], self.augmat[p] = self.augmat[p], self.augmat[i]

            if self.augmat[i, i] == 0.0:
                raise ZeroDivisionError(
                    f"{self.augmat} has a zero diagonal element at {i}, {i}"
                )

            for j in range(i + 1, n):
                scaling_factor = self.augmat[j, i] / self.augmat[i, i]
                self.augmat[j] = self.augmat[j] - (scaling_factor * self.augmat[i])

            i += 1

        # Back substitution
        solution[m] = self.augmat[m, n] / self.augmat[m, m]

        for k in range(n - 2, -1, -1):
            solution[k] = self.augmat[k, n]
            for j in range(k + 1, n):
                solution[k] -= self.augmat[k, j] * solution[j]

            solution[k] /= self.augmat[k, k]

        return round(solution, 5)


class LUSolve:
    """
    Solve a system of linear equations using the LU Decomposition method.
    Both `crout` and `doolittle` methods are implemented.
    """

    def __init__(self, matrix: Matrix, vector: Array, method: str):
        self.matrix = matrix
        self.vector = vector
        self.method = method
        self.pivmat = None

    def _partial_pivot(self):
        for i in range(self.vector.length - 1):
            if self.matrix[i, i] == 0:
                for j in range(i + 1, self.vector.length):
                    if abs(self.matrix[j, i]) > abs(self.matrix[i, i]):
                        self.matrix[i], self.matrix[j] = self.matrix[j], self.matrix[i]
                        self.vector[i], self.vector[j] = self.vector[j], self.vector[i]

    def decompose(self):
        lower = Matrix(self.matrix.nrows, self.matrix.ncols)
        upper = Matrix(self.matrix.nrows, self.matrix.ncols)
        if self.method == "crout":
            for i in range(self.matrix.nrows):
                upper[i, i] = 1.0
                for j in range(i, self.matrix.nrows):
                    tmp_lower = lower[j, i]
                    for k in range(i):
                        tmp_lower -= lower[j, k] * upper[k, i]
                    lower[j, i] = tmp_lower
                for j in range(i + 1, self.matrix.nrows):
                    tmp_upper = upper[i, j]
                    for k in range(i):
                        tmp_upper -= lower[i, k] * upper[k, j]
                    upper[i, j] = tmp_upper / lower[i, i]
            return lower, upper

        elif self.method == "doolittle":
            for i in range(self.matrix.nrows):
                for k in range(i, self.matrix.nrows):
                    sum_1 = 0
                    for j in range(i):
                        sum_1 += lower[i, j] * upper[j, k]
                    upper[i, k] = self.matrix[i, k] - sum_1

                for k in range(i, self.matrix.nrows):
                    if i == k:
                        lower[i, i] = 1
                    else:
                        sum_2 = 0
                        for j in range(i):
                            sum_2 += lower[k, j] * upper[j, i]
                        lower[k, i] = (self.matrix[k, i] - sum_2) / upper[i, i]

            return lower, upper

    def solve(self):
        self._partial_pivot()
        lower, upper = self.decompose()

        # Forward substitution
        n = self.matrix.nrows
        y = zeros("d", n)
        for k in range(n):
            y[k] = self.vector[k]
            for j in range(k):
                y[k] -= lower[k, j] * y[j]
            y[k] /= lower[k, k]

        # Back substitution
        x = zeros("d", n)
        for k in range(n, 0, -1):
            x[k - 1] = y[k - 1]
            for j in range(k, n):
                x[k - 1] -= upper[k - 1, j] * x[j]
            x[k - 1] /= upper[k - 1, k - 1]

        return round(x, 5)


class Cholesky:
    def __init__(self, matrix: Matrix, b: Array):
        self.L = matrix
        self.b = Array("d", b)

    def decompose(self):
        for i in range(self.L.nrows):
            for k in range(i, self.L.nrows):
                if i == k:
                    sum_1 = sum(self.L[i, j] ** 2 for j in range(i))
                    self.L[i, i] = sqrt(self.L[i, i] - sum_1)
                elif i < k:
                    sum_2 = sum(self.L[i, j] * self.L[j, k] for j in range(i))
                    self.L[i, k] = (self.L[i, k] - sum_2) / self.L[i, i]
                    self.L[k, i] = self.L[i, k]

        return self.L

    def solve(self):
        y = zeros("d", self.b.length)

        # Forward sub
        for i in range(self.b.length):
            summ = 0
            for j in range(i):
                summ += self.L[i, j] * y[j]
            y[i] = (self.b[i] - summ) / self.L[i, i]

        # Backward sub
        for i in range(self.b.length - 1, -1, -1):
            summ = 0
            for j in range(i + 1, self.b.length):
                summ += self.L[i, j] * self.b[j]
            self.b[i] = (y[i] - summ) / self.L[i, i]

        return round(self.b, 5)


class GaussJacobi:
    pass


class GaussSeidel:
    def __init__(self, matrix: Matrix, b: Array, tol: float = 1e-6):
        self.matrix = matrix
        self.b = b
        self.x = zeros("d", b.length)
        self._x = zeros("d", b.length)
        self.tol = tol

    def solve(self):
        while True:
            delta = 0
            for i in range(self.matrix.nrows):
                self._x[i] = self.b[i]
                for j in range(self.matrix.nrows):
                    if i != j:
                        self._x[i] -= self.matrix[i, j] * self.x[j]
                self._x[i] /= self.matrix[i, i]
                delta += abs(self._x[i] - self.x[i])

            for i in range(self.matrix.nrows):
                self.x[i] = self._x[i]

            if delta < self.tol:
                break

        return round(self.x, 5)


class ConjugateGradient:
    def __init__(
        self,
        matrix: Matrix | Callable,
        b: Array,
        x0: Array = None,
        tol: float = 1e-10,
        max_iter: int = 10000,
    ):
        self.matrix = matrix
        self.b = b
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        self.residue = None

    def _get_gradients_and_residue(self):
        if isinstance(self.matrix, Matrix):
            if self.x0 is None:
                self.x0 = zeros("d", self.b.length)
            if not isinstance(self.x0, Array):
                self.x0 = Array("d", self.x0)

            r = self.b - self.matrix @ self.x0
            d = r
            self.residue = []
            count = 1

            while inner(r, r) > self.tol and count <= self.max_iter:
                k = inner(r, r)
                alpha = k / inner(d, self.matrix @ d)
                self.x0 = self.x0 + d * alpha
                r -= (self.matrix @ d) * alpha
                if count == 1:
                    d = r
                beta = inner(r, r) / k
                d = r + d * beta
                count += 1
                self.residue.append(sqrt(inner(r, r)))

            return self

        elif callable(self.matrix):
            r = self.b - self.matrix(self.x0)
            d = r
            self.residue = []
            count = 1

            while norm(r) > self.tol and count <= self.max_iter:
                k = inner(r, r)
                alpha = k / inner(d, self.matrix(d))
                self.x0 = self.x0 + d * alpha
                r = r - self.matrix(d) * alpha
                if count == 1:
                    d = r
                beta = inner(r, r) / k
                d = r + d * beta
                self.residue.append(norm(r))
                count += 1

            return self

    def solve(self, plot: bool = None):
        if isinstance(self.matrix, Matrix):
            inverse = Matrix(self.matrix.nrows, self.matrix.ncols)
            for i in range(self.matrix.nrows):
                e = zeros("d", self.matrix.nrows)
                e[i] = 1
                self._get_gradients_and_residue()
                inverse[i] = self.x0

            return self, inverse

        elif callable(self.matrix):
            vals = []
            res = []

            for i in range(self.b.length):
                e = zeros("d", self.b.length)
                e[i] = 1
                self._get_gradients_and_residue()
                vals.append(self.x0)
                res.append(self.residue)

            res = Matrix.from_list(res)
            vals = Matrix.from_list(vals).transpose()

            res = res**2

            residue = zeros("d", res.nrows)

            for i in range(res.nrows):
                residue[i] = sqrt(sum(res[i]))

            if plot is True:
                plt.plot(self.residue)
                plt.xlabel("Iterations")
                plt.ylabel("Residue")
                plt.yscale("log")
                plt.title("Conjugate Gradient Residue")
                plt.show()

            return self, vals, residue

    def plot_residue(self):
        plt.plot(self.residue)
        plt.xlabel("Iterations")
        plt.ylabel("Residue")
        plt.yscale("log")
        plt.title("Conjugate Gradient Residue")
        plt.show()


def conjugate_gradient(
    matrix: Matrix,
    b: Array,
    x0: Array = None,
    tol: float = 1e-10,
    max_iter: int = 10000,
):
    if x0 is None:
        x0 = zeros("d", b.length)

    if not isinstance(x0, Array):
        x0 = Array("d", x0)

    r = b - matrix @ x0
    d = r
    residue = []
    count = 1

    while inner(r, r) > tol and count <= max_iter:
        k = inner(r, r)
        alpha = k / inner(d, matrix @ d)
        x0 = x0 + d * alpha
        r -= (matrix @ d) * alpha
        if count == 1:
            d = r
        beta = inner(r, r) / k
        d = r + d * beta
        count += 1
        residue.append(sqrt(inner(r, r)))

    return x0, residue


def inverse_conjugate_gradient(
    matrix: Matrix, tol: float = 1e-10, max_iter: int = 10000
):
    inverse = Matrix(matrix.nrows, matrix.ncols)
    for i in range(matrix.nrows):
        e = zeros("d", matrix.nrows)
        e[i] = 1
        x, _ = conjugate_gradient(matrix.transpose(), e, tol=tol, max_iter=max_iter)
        inverse[i] = x

    return inverse


def conjugate_gradient_no_matrix(
    func: Callable, b: Array, tol: float = 1e-6, max_iter: int = 500
):
    x0 = zeros("d", b.length)
    r = b - func(x0)
    d = r
    residue = []
    count = 1

    while norm(r) > tol and count <= max_iter:
        k = inner(r, r)
        alpha = k / inner(d, func(d))
        x0 = x0 + d * alpha
        r = r - func(d) * alpha
        if count == 1:
            d = r
        beta = inner(r, r) / k
        d = r + d * beta
        residue.append(norm(r))
        count += 1

    return x0, residue


def inverse_conjugate_gradient_no_matrix(
    func, n, tol: float = 1e-6, max_iter: int = 500, plot: bool = True
):
    sol = []
    res = []

    for i in range(n):
        e = zeros("d", n)
        e[i] = 1
        x, r = conjugate_gradient_no_matrix(func, e, tol=tol, max_iter=max_iter)
        sol.append(x)
        res.append(r)

    res = Matrix.from_list(res)
    sol = Matrix.from_list(sol).transpose()

    res = res**2

    residue = zeros("d", res.nrows)

    for i in range(res.nrows):
        residue[i] = sqrt(sum(res[i]))

    if plot is True:
        plt.plot(r)
        plt.xlabel("Iterations")
        plt.ylabel("Residue")
        plt.yscale("log")
        plt.title("Conjugate Gradient Residue")
        plt.show()

    return sol, residue


def steepest_descent():
    pass


def gmres():
    pass


class QRFactorization:
    """Decompose a matrix into its QR factors using Householder or Gram-Schmidt method
    and return the eigenvalues and eigenvectors of the matrix.

    Attributes
    ----------
    matrix : Matrix
        The matrix to be decomposed

    Q : Matrix

    R : Matrix

    Methods
    -------
    householder()
        Perform QR decomposition using Householder method.

    gram_schmidt()
        Perform QR decomposition using Gram-Schmidt method.

    solve()
        Solve the system of linear equations using the QR decomposition.
    """

    def __init__(
        self,
        matrix: Matrix,
        method: str = "gram_schmidt",
        tol: float = 1e-10,
        max_iter: int = 10000,
    ):
        self.matrix = matrix
        self.Q = Matrix(matrix.nrows, matrix.ncols)
        self.R = Matrix(matrix.nrows, matrix.ncols)
        self.method = method
        self.tol = tol
        self.max_iter = max_iter

    def householder(self):
        raise NotImplementedError("Householder method not implemented yet")

    def gram_schmidt(self):
        n, m = self.matrix.nrows, self.matrix.ncols

        u = Matrix(n, n)

        u0 = self.matrix[:, 0]
        u0 = Array("d", u0)
        self.Q[:, 0] = u0 / norm(u0)
        u[:, 0] = u0

        for i in range(1, n):
            ui = self.matrix[:, i]
            ui = Array("d", ui)
            for j in range(i):
                Qj = self.Q[:, j]
                Qj = Array("d", Qj)
                ui -= inner(self.matrix[:, i], self.Q[:, j]) * Qj
                # u[:, i] = ui

            # ui = u[:, i]
            # ui = Array("d", ui)

            self.Q[:, i] = ui / norm(ui)

        for i in range(n):
            for j in range(m):
                self.R[i, j] = inner(self.Q[:, i], self.matrix[:, j])

        return self

    def solve(self):
        if self.method == "householder":
            self.householder()
        elif self.method == "gram_schmidt":
            self.matrix = deepcopy(self.matrix)
            matrix_new = deepcopy(self.matrix)

            diff = math.inf
            i = 0
            while (diff > self.tol) and (i < self.max_iter):
                self.matrix = deepcopy(matrix_new)
                self.gram_schmidt()
                matrix_new[:, :] = self.R @ self.Q
                diff = abs(matrix_new - self.matrix).max()

                i += 1
            eigenvals = matrix_new.diag()

            return eigenvals


def power_iteration(A: Matrix, tol: float = 1e-6, max_iter: int = 1000):
    lam_prev = 0
    x = ones("d", A.nrows)
    lam = 0.0
    for i in range(max_iter):
        x = A @ x / norm(A @ x)
        lam = inner(x, (A @ x)) / inner(x, x)

        if abs(lam - lam_prev) < tol:
            break

        lam_prev = lam

    return lam, x
