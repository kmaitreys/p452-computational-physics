# import math

from .array import Array


class LagrangeInterpolator:
    def __init__(self, x: Array, y: Array):
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        self.x_vals = x
        self.y_vals = y
        self.n = len(x)

    def _basis(self, i: int, x: float) -> float:
        """
        Helper function to calculate the basis polynomial.
        It is not intended to be called directly.
        """
        result = 1.0
        for j in range(self.n):
            if j != i:
                result *= (x - self.x_vals[j]) / (self.x_vals[i] - self.x_vals[j])
        return result

    def interpolate(self, x: float) -> float:
        result = 0.0
        for i in range(self.n):
            result += self.y_vals[i] * self._basis(i, x)
        return result




# interp = LagrangeInterpolator(Array("d", [0, 1/math.sqrt(3)]), )

# print(interp)
    

