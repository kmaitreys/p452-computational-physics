"""
This module defines the basic data models
like Arrays, Matrices, Vectors to be used
to model the problems for the computational
physics course.
"""

from array import array
from typing import Self


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


class Array(array):
    """
    A class to represent an array of numbers.
    Numbers can be of any type. This class
    wraps the array class to provide
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

    def __new__(cls, typecode, iterable):
        return array.__new__(cls, typecode, iterable)

    def abs(self):
        return Array(self.typecode, (abs(x) for x in self))

    @property
    def length(self):
        return len(self)

    @staticmethod
    def zeros(typecode, size):
        return Array(typecode, [0] * size)

    @staticmethod
    def ones(typecode, size):
        return Array(typecode, [1] * size)

    @staticmethod
    def arange(typecode, start, stop, step):
        return Array(typecode, frange(start, stop, step))

    @staticmethod
    def linspace(typecode, start, stop, num):
        step = (stop - start) / (num - 1)
        return Array(typecode, [start + step * i for i in frange(num)])

    @staticmethod
    def logspace(typecode, start, stop, num, base=10):
        step = (stop - start) / (num - 1)
        return Array(typecode, [base ** (start + step * i) for i in frange(num)])

    @staticmethod
    def random(typecode, size):
        raise NotImplementedError("Random number generation is not implemented yet.")

    @staticmethod
    def inner(a, b):
        if len(a) != len(b):
            raise ValueError("Arrays must be of the same size")
        return sum([x * y for x, y in zip(a, b)])
    
    @staticmethod
    def norm(a):
        return Array.inner(a, a) ** 0.5

    def __add__(self, other: Self) -> Self:
        if isinstance(other, Array):
            return Array(self.typecode, [x + y for x, y in zip(self, other)])
        else:
            raise ValueError("Array can only be added to another Array")

    def __radd__(self, other: int | float) -> Self:
        return self.__add__(other)

    def __sub__(self, other: Self) -> Self:
        if isinstance(other, Array):
            return Array(self.typecode, [x - y for x, y in zip(self, other)])
        else:
            raise ValueError("Array can only be subtracted from another Array")

    def __rsub__(self, other: int | float) -> Self:
        return self.__sub__(other)

    def __isub__(self, other: Self) -> Self:
        return self.__sub__(other)

    def __mul__(self, other: Self | int | float) -> Self:
        if isinstance(other, (int, float)):
            return Array(self.typecode, [x * other for x in self])
        elif isinstance(other, Array):
            return Array(self.typecode, [x * y for x, y in zip(self, other)])
        else:
            raise ValueError(
                "Array can only be multiplied by a scalar or another Array"
            )

    def __rmul__(self, other: int | float) -> Self:
        return self.__mul__(other)

    def __truediv__(self, other: int | float) -> Self:
        if other == 0:
            raise ValueError("Division by zero")
        return Array(self.typecode, [x / other for x in self])
    
    def __round__(self, n):
        return Array(self.typecode, [round(x, n) for x in self])
    
    def __abs__(self):
        return Array(self.typecode, [abs(x) for x in self])
    
    def __pow__(self, other: int | float) -> Self:
        if isinstance(other, (int, float)):
            return Array(self.typecode, [x ** other for x in self])
        else:
            raise ValueError("Array can only be raised to a scalar power")

    def __repr__(self) -> str:
        # Make a more pretty representation of the array
        type_dict = {
            "b": "char",
            "B": "uchar",
            "h": "short",
            "H": "ushort",
            "i": "int",
            "I": "uint",
            "l": "long",
            "L": "ulong",
            "q": "longlong",
            "Q": "ulonglong",
            "f": "float",
            "d": "double",
        }
        return f"({list(self)}, {type_dict[self.typecode]})"


class Matrix:
    """
    The Matrix class.

    Attributes
    ----------
    nrows : int
        The number of rows in the matrix
    ncols : int
        The number of columns in the matrix
    data : list
        The list of Array objects representing the matrix
    rows : list
        The list of rows of the matrix
    cols : list
        The list of columns of the matrix
    trace : float
        The trace of the matrix
    
    Methods
    -------
    from_list(data)
        Create a matrix from a list of lists.
    
    transpose()
        Return the transpose of the matrix.
    
    diag(offset=0)
        Return the diagonal of the matrix.
    
    invert()
        Return the inverse of the matrix.

    """

    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.data = [Array("d", [0] * ncols) for _ in range(nrows)]

    @classmethod
    def from_list(cls, data):
        nrows = len(data)
        ncols = len(data[0])
        result = cls(nrows, ncols)
        for i in range(nrows):
            for j in range(ncols):
                result.data[i][j] = data[i][j]
        return result

    def transpose(self):
        result = Matrix(self.ncols, self.nrows)
        for i in range(self.nrows):
            for j in range(self.ncols):
                result.data[j][i] = self.data[i][j]
        return result

    def diag(self, offset=0):
        if self.nrows != self.ncols:
            raise ValueError("Matrix must be square")
        if offset > 0:
            return Array(self.data[0][offset:] + [0] * offset)
        elif offset < 0:
            return Array(
                [0] * -offset
                + [self.data[i][-offset] for i in range(-offset, self.nrows)]
            )
        else:
            return Array(self.data[i][i] for i in range(self.nrows))

    @property
    def rows(self):
        return [row for row in self.data]

    @property
    def cols(self):
        return [[self.data[i][j] for i in range(self.nrows)] for j in range(self.ncols)]

    @property
    def trace(self):
        if self.nrows != self.ncols:
            raise ValueError("Matrix must be square")
        return sum(self.data[i][i] for i in range(self.nrows))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            if isinstance(row, slice) and isinstance(col, slice):
                return [
                    [
                        self.data[i][j]
                        for j in range(
                            col.start or 0, col.stop or self.ncols, col.step or 1
                        )
                    ]
                    for i in range(
                        row.start or 0, row.stop or self.nrows, row.step or 1
                    )
                ]
            elif isinstance(row, slice):
                return [
                    self.data[i][col]
                    for i in range(
                        row.start or 0, row.stop or self.nrows, row.step or 1
                    )
                ]
            elif isinstance(col, slice):
                return [
                    self.data[row][j]
                    for j in range(
                        col.start or 0, col.stop or self.ncols, col.step or 1
                    )
                ]
            else:
                return self.data[row][col]
        else:
            return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row, col = key
            self.data[row][col] = value
        else:
            self.data[key] = value

    def __add__(self, other):
        if isinstance(other, Matrix):
            if self.nrows != other.nrows or self.ncols != other.ncols:
                raise ValueError("Matrix dimensions must be the same")
            result = Matrix(self.nrows, self.ncols)
            for i in range(self.nrows):
                for j in range(self.ncols):
                    result.data[i][j] = self.data[i][j] + other.data[i][j]
            return result
        else:
            raise ValueError("Matrix can only be added to another Matrix")

    def __sub__(self, other):
        if isinstance(other, Matrix):
            if self.nrows != other.nrows or self.ncols != other.ncols:
                raise ValueError("Matrix dimensions must be the same")
            result = Matrix(self.nrows, self.ncols)
            for i in range(self.nrows):
                for j in range(self.ncols):
                    result.data[i][j] = self.data[i][j] - other.data[i][j]
            return result
        else:
            raise ValueError("Matrix can only be subtracted from another Matrix")

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.ncols != other.nrows:
                raise ValueError("Matrix dimensions must be compatible")
            result = Matrix(self.nrows, other.ncols)
            for i in range(self.nrows):
                for j in range(other.ncols):
                    for k in range(self.ncols):
                        result.data[i][j] += self.data[i][k] * other.data[k][j]
            return result
        else:
            raise ValueError("Matrix can only be multiplied by another Matrix")

    def invert(self):
        if self.nrows != self.ncols:
            raise ValueError("Matrix must be square")
        result = Matrix(self.nrows, self.ncols)
        for i in range(self.nrows):
            result.data[i][i] = 1
        for i in range(self.nrows):
            factor = 1 / self.data[i][i]
            for j in range(self.ncols):
                self.data[i][j] *= factor
                result.data[i][j] *= factor
            for k in range(self.nrows):
                if k != i:
                    factor = -self.data[k][i]
                    for j in range(self.ncols):
                        self.data[k][j] += factor * self.data[i][j]
                        result.data[k][j] += factor * result.data[i][j]
        return result

    def __str__(self):
        return "\n".join(" ".join(str(x) for x in row) for row in self.data)

    def __repr__(self):
        return f"Matrix({self.nrows}, {self.ncols})"
