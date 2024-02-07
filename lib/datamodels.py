"""
This module defines the basic data models
like Arrays, Matrices, Vectors to be used
to model the problems for the computational
physics course.
"""


import array


def frange(start, stop=None, step=None):
    """
    Helper function to generate a range
    with step size as a float.
    """
    # if set start=0.0 and step = 1.0 if not specified
    start = float(start)
    if stop is None:
        stop = start + 0.0
        start = 0.0
    if step is None:
        step = 1.0

    num_of_steps = 0
    while True:
        next_step = float(start + num_of_steps * step)
        if step > 0 and next_step >= stop:
            break
        elif step < 0 and next_step <= stop:
            break
        yield next_step
        num_of_steps += 1


class Array:
    """
    A class to represent an array of numbers.
    Numbers can be of any type. This class
    wraps the array.array class to provide
    additional functionality.

    Methods
    -------
    zeros(typecode, size)
        Returns an array of zeros of given size and typecode.

    ones(typecode, size)
        Returns an array of ones of given size and typecode.

    arange(typecode, start, stop, step)
        Returns an array of numbers from start to stop with given step size and typecode.

    linspace(typecode, start, stop, num)
        Returns an array of numbers from start to stop, linearly spaced with given number of points and typecode.

    logspace(typecode, start, stop, num, base)
        Returns an array of numbers from start to stop, logarithmically spaced with given number of points and typecode.

    random(typecode, size)
        Returns an array of random numbers of given size and typecode.
    
    
    """

    @staticmethod
    def zeros(typecode, size):
        return array.array(typecode, [0] * size)

    @staticmethod
    def ones(typecode, size):
        return array.array(typecode, [1] * size)

    @staticmethod
    def arange(typecode, start, stop, step):
        return array.array(typecode, frange(start, stop, step))

    @staticmethod
    def linspace(typecode, start, stop, num):
        step = (stop - start) / (num - 1)
        return array.array(typecode, [start + step * i for i in frange(num)])

    @staticmethod
    def logspace(typecode, start, stop, num, base=10):
        step = (stop - start) / (num - 1)
        return array.array(typecode, [base ** (start + step * i) for i in frange(num)])

    @staticmethod
    def random(typecode, size):
        raise NotImplementedError("Random number generation is not implemented yet.")


class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0] * cols for _ in range(rows)]

    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    def __add__(self, other):
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrix dimensions must be the same")
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + other.data[i][j]
            return result
        else:
            raise ValueError("Matrix can only be added to another Matrix")

    def __sub__(self, other):
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrix dimensions must be the same")
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] - other.data[i][j]
            return result
        else:
            raise ValueError("Matrix can only be subtracted from another Matrix")

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Matrix dimensions must be compatible")
            result = Matrix(self.rows, other.cols)
            for i in range(self.rows):
                for j in range(other.cols):
                    for k in range(self.cols):
                        result.data[i][j] += self.data[i][k] * other.data[k][j]
            return result
        else:
            raise ValueError("Matrix can only be multiplied by another Matrix")

    def invert(self):
        if self.rows != self.cols:
            raise ValueError("Matrix must be square")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            result.data[i][i] = 1
        for i in range(self.rows):
            factor = 1 / self.data[i][i]
            for j in range(self.cols):
                self.data[i][j] *= factor
                result.data[i][j] *= factor
            for k in range(self.rows):
                if k != i:
                    factor = -self.data[k][i]
                    for j in range(self.cols):
                        self.data[k][j] += factor * self.data[i][j]
                        result.data[k][j] += factor * result.data[i][j]
        return result
    
    def __str__(self):
        return "\n".join(" ".join(str(x) for x in row) for row in self.data)

    def __repr__(self):
        return f"Matrix({self.rows}, {self.cols})"
    


