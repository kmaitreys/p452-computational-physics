import array
import random


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
        return array.array(typecode, [random.random() for _ in range(size)])
