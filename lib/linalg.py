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

        for k in range(n-2, -1, -1):
            solution[k] = self.augmat[k, n]
            for j in range(k+1, n):
                solution[k] -= self.augmat[k, j] * solution[j]
            
            solution[k] /= self.augmat[k, k]

        return solution



def lu_decomposition():
    pass


def cholesky_decomposition():
    pass


def gauss_jacobi():
    pass


def gauss_seidel():
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
