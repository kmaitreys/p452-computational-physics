import array


class Array:
    def __init__(self, typecode, size):
        self.typecode = typecode
        self.size = size
        self.arr = array.array(typecode, [0] * size)

    def __str__(self):
        return str(self.arr)

    def zeros(self):
        return array.array(self.typecode, [0] * self.size)

    def arange(self, start, stop, step):
        return array.array(self.typecode, range(start, stop, step))


# Creating an array
arr = Array("i", 5)
print(arr)  # Output: array('i', [0, 0, 0, 0, 0])

# Creating an array filled with zeros
zeros_arr = arr.zeros()
print(zeros_arr)  # Output: array('i', [0, 0, 0, 0, 0])

# Creating an array using arange
arange_arr = arr.arange(1, 10, 2)
print(arange_arr)  # Output: array('i', [1, 3, 5, 7, 9])
