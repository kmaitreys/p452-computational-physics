from datamodels import Array, Matrix


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
        solution = Array.zeros("d", n)
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

        return solution


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
        lower, upper = self.decompose()
        
        # Forward substitution
        n = self.matrix.nrows
        y = Array.zeros("d", n)
        y[0] = self.vector[0] / lower[0, 0]
        for i in range(1, n):
            sum_1 = sum(lower[i, j] * y[j] for j in range(i))
            y[i] = (self.vector[i] - sum_1) / lower[i, i]
        
        # Back substitution
        x = Array.zeros("d", n)
        x[n-1] = y[n-1] / upper[n-1, n-1]
        for i in range(n-2, -1, -1):
            sum_2 = sum(upper[i, j] * x[j] for j in range(i+1, n))
            x[i] = (y[i] - sum_2) / upper[i, i]

        return x
    
    def decompose(self):
        lower = Matrix(self.matrix.nrows, self.matrix.ncols)
        upper = Matrix(self.matrix.nrows, self.matrix.ncols)
        if self.method == "crout":
            for k in range(self.matrix.nrows):
                upper[k, k] = 1.0
                for j in range(k, self.matrix.nrows):
                    sum_1 = sum(lower[k, s] * upper[s, j] for s in range(1, k-1))
                    lower[j, k] = self.matrix[k, k] - sum_1
                
                for j in range(self.matrix.nrows):
                    sum_2 = sum(lower[k, s] * upper[s, j] for s in range(1, k-1))
                    upper[k, j] = (self.matrix[k, j] - sum_2) / lower[k, k]

            return lower, upper
        elif self.method == "doolittle":
            for k in range(self.matrix.nrows):
                lower[k, k] = 1.0
                for j in range(k, self.matrix.nrows):
                    sum_1 = sum(lower[k, s] * upper[s, j] for s in range(1, k-1))
                    upper[k, j] = self.matrix[k, j] - sum_1
                
                for j in range(self.matrix.nrows):
                    sum_2 = sum(lower[k, s] * upper[s, j] for s in range(1, k-1))
                    lower[j, k] = (self.matrix[j, k] - sum_2) / upper[k, k]
            
            return lower, upper


lu = LUSolve(
    matrix=Matrix.from_list(
        [
            [0, 2, 5],
            [3, -1, 2],
            [1, -1, 3]
        ]
    ),
    vector=Array("d", [1, -2, 3]),
    method="crout"
)

sol = lu.solve()
print(sol)


class Cholesky:
    pass


class GaussJacobi:
    pass


class GaussSeidel:
    pass


def conjugate_gradient():
    pass


def steepest_descent():
    pass


def gmres():
    pass


def qr_decomposition():
    pass


def gram_schmidt():
    pass


def householder():
    pass


def crout():
    pass


def doolittle():
    pass
