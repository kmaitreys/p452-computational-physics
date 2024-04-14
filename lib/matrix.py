"""
This module defines the Matrix class which is used to represent a matrix of numbers.
The Matrix class can be used to perform matrix operations such as addition, subtraction,
multiplication, and inversion.
"""

import ctypes

from .array import Array


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

    def __init__(self, nrows, ncols, data="zeros"):
        self.nrows = nrows
        self.ncols = ncols
        if data == "zeros":
            self.data = [Array("d", [0] * ncols) for _ in range(nrows)]
        elif data == "empty":
            raise NotImplementedError("Empty matrix not implemented")
            # Initialize an empty matrix using the ctypes module
            item_size = ctypes.sizeof(ctypes.c_double)
            buffer = (ctypes.c_double * (nrows * ncols * item_size))()
            self.data = [
                Array("d", buffer[i * ncols : (i + 1) * ncols]) for i in range(nrows)
            ]

    @classmethod
    def from_list(cls, data):
        nrows = len(data)
        ncols = len(data[0])
        result = cls(nrows, ncols)
        for i in range(nrows):
            for j in range(ncols):
                result.data[i][j] = data[i][j]
        return result
    
    @classmethod
    def from_array_list(cls, data):
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
            return Array("d", self.data[0][offset:] + [0] * offset)
        elif offset < 0:
            return Array(
                "d",
                [0] * -offset
                + [self.data[i][-offset] for i in range(-offset, self.nrows)],
            )
        else:
            return Array("d", [self.data[i][i] for i in range(self.nrows)])

    def det(self):
        if self.nrows != self.ncols:
            raise ValueError("Matrix must be square")
        if self.nrows == 1:
            return self.data[0][0]
        if self.nrows == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        
        det = 0
        for col in range(self.ncols):
            cofactor = (-1) ** col * self.data[0][col]
            minor = self._get_minor(0, col)
            minor = Matrix.from_array_list(minor)
            det += cofactor * minor.det()

        return det

    def _get_minor(self, row, col):
        return [
            row[:col] + row[col + 1:]
            for row in (self.data[:row] + self.data[row + 1:])
        ]


    @staticmethod
    def identity(n):
        result = Matrix(n, n)
        for i in range(n):
            result.data[i][i] = 1
        return result

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
            if isinstance(row, slice) and isinstance(col, slice):
                for i, r in enumerate(
                    range(row.start or 0, row.stop or self.nrows, row.step or 1)
                ):
                    for j, c in enumerate(
                        range(col.start or 0, col.stop or self.ncols, col.step or 1)
                    ):
                        self.data[r][c] = value[i][j]
            elif isinstance(row, slice):
                for i, r in enumerate(
                    range(row.start or 0, row.stop or self.nrows, row.step or 1)
                ):
                    self.data[r][col] = value[i]
            elif isinstance(col, slice):
                for j, c in enumerate(
                    range(col.start or 0, col.stop or self.ncols, col.step or 1)
                ):
                    self.data[row][c] = value[j]
            else:
                self.data[row][col] = value

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
        elif isinstance(other, Array):
            if self.ncols != other.length:
                raise ValueError("Matrix and Array dimensions must be compatible")
            result = Array("d", [0] * self.nrows)
            for i in range(self.nrows):
                for j in range(self.ncols):
                    result[i] += self.data[i][j] * other[j]
            return result

    def __matmul__(self, other):
        return self.__mul__(other)

    def __round__(self, n):
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.data[i][j] = round(self.data[i][j], n)
        return self

    def __abs__(self):
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.data[i][j] = abs(self.data[i][j])
        return self

    def max(self):
        max_val = self.data[0][0]
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.data[i][j] > max_val:
                    max_val = self.data[i][j]
        return max_val

    def __pow__(self, other):
        if isinstance(other, int | float):
            for i in range(self.nrows):
                for j in range(self.ncols):
                    self.data[i][j] **= other
            return self
        else:
            raise ValueError("Matrix can only be raised to a scalar power")

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
    
    def norm(self):
        norm = 0
        for i in range(self.nrows):
            for j in range(self.ncols):
                norm += self.data[i][j] ** 2
        return norm ** 0.5

    def __str__(self):
        # Round off the numbers to 5 decimal places
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.data[i][j] = round(self.data[i][j], 5)
        return "\n".join(
            " ".join("{:>{width}}".format(x, width=10) for x in row)
            for row in self.data
        )

    def __repr__(self):
        # Pretty print the matrix
        return "\n".join(" ".join(str(x) for x in row) for row in self.data)
