"""
This module implements an Array class that wraps the array class
to provide additional functionality. The Array class can be used
to perform element-wise operations on arrays of numbers.

Example
-------
>>> a = Array("d", [1, 2, 3, 4])
>>> b = Array("d", [5, 6, 7, 8])
>>> c = a + b
>>> print(c)
(6.0, 8.0, 10.0, 12.0)

>>> d = a * 2
>>> print(d)
(2.0, 4.0, 6.0, 8.0)
"""

from array import array
from typing import Self


class Array(array):
    """
    A class to represent an array of numbers.
    Numbers can be of any type. This class
    wraps the array class to provide
    additional functionality.

    Attributes
    ----------
    typecode : str
        The type of the numbers in the array

    Methods
    -------

    abs()
        Return the absolute value of the array.

    length()
        Return the length of the array.

    """

    def __new__(cls, typecode, iterable):
        return array.__new__(cls, typecode, iterable)

    def abs(self):
        return Array(self.typecode, (abs(x) for x in self))

    @property
    def length(self):
        return len(self)
    
    @classmethod
    def fromlist(cls, typecode, lst):
        return cls(typecode, lst)

    def __add__(self, other: Self) -> Self:
        if isinstance(other, Array):
            return Array(self.typecode, [x + y for x, y in zip(self, other)])
        elif isinstance(other, (int, float)):
            return Array(self.typecode, [x + other for x in self])
        else:
            raise ValueError(f"{type(other)} cannot be added to Array")

    def __radd__(self, other: int | float) -> Self:
        return self.__add__(other)

    def __sub__(self, other: Self) -> Self:
        if isinstance(other, Array):
            return Array(self.typecode, [x - y for x, y in zip(self, other)])
        elif isinstance(other, (int, float)):
            return Array(self.typecode, [x - other for x in self])
        else:
            raise ValueError(f"{type(other)} cannot be subtracted from Array")

    def __rsub__(self, other: int | float) -> Self:
        return self.__sub__(other)

    def __isub__(self, other: Self) -> Self:
        return self.__sub__(other)

    def __mul__(self, other: Self | int | float) -> Self:
        if isinstance(other, (int, float)):
            return Array(self.typecode, [x * other for x in self])
        elif isinstance(other, Array):
            if len(self) != len(other):
                raise ValueError("Arrays must be of the same size")
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
            return Array(self.typecode, [x**other for x in self])
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


def zeros(typecode: str, size: int):
    return Array(typecode, [0] * size)


def ones(typecode: str, size: int):
    return Array(typecode, [1] * size)


def arange(typecode: str, start: float | int, stop: float | int, step: float | int):
    return Array(typecode, frange(start, stop, step))


def linspace(typecode, start, stop, num):
    step = (stop - start) / (num - 1)
    return Array(typecode, [start + step * i for i in frange(num)])


def logspace(typecode, start, stop, num, base=10):
    step = (stop - start) / (num - 1)
    return Array(typecode, [base ** (start + step * i) for i in frange(num)])


def random(typecode, size):
    raise NotImplementedError("Random number generation is not implemented yet.")


def inner(a: Array, b: Array):
    if len(a) != len(b):
        raise ValueError("Arrays must be of the same size")
    return sum([x * y for x, y in zip(a, b)])


def norm(a: Array) -> float:
    return inner(a, a) ** 0.5
