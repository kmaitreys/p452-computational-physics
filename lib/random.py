from typing import Iterator

from .array import zeros


def _lcg(m: int, a: int, c: int, seed: int) -> Iterator[int]:
    """
    Implements the linear congruential generator.
    """

    x = seed
    while True:
        yield x
        x = (a * x + c) % m


def random(
    start: int, stop: int, lcg_params: tuple = None, seed: int = 123_456_789, size=None
):
    """
    Return a random number in the range [start, stop)
    using a linear congruential generator.
    """
    if lcg_params is not None:
        a = lcg_params[0]
        m = lcg_params[1]
    else:
        m = 2**32
        a = 594_156_893
    c = 0
    gen = _lcg(m, a, c, seed)

    if size is None:
        return start + (next(gen) / m) % (stop - start)
    if size is not None:
        sequence = zeros("d", size)
        for i in range(size):
            sequence[i] = start + (next(gen) / m) % (stop - start)

        return sequence
