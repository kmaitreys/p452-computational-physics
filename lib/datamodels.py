import array
import random


class Array:
    @staticmethod
    def zeros(typecode, size):
        return array.array(typecode, [0] * size)

    @staticmethod
    def ones(typecode, size):
        return array.array(typecode, [1] * size)

    @staticmethod
    def arange(typecode, start, stop, step):
        return array.array(typecode, range(start, stop, step))

    @staticmethod
    def linspace(typecode, start, stop, num):
        step = (stop - start) / (num - 1)
        return array.array(typecode, [start + step * i for i in range(num)])

    @staticmethod
    def logspace(typecode, start, stop, num, base=10):
        step = (stop - start) / (num - 1)
        return array.array(typecode, [base ** (start + step * i) for i in range(num)])

    @staticmethod
    def random(typecode, size):
        return array.array(typecode, [random.random() for _ in range(size)])
