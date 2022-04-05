from abc import ABC
import numbers
import numpy as np
from fractions import Fraction


class TextBlock:
    def __init__(self, rows):
        assert isinstance(rows, list)
        self.rows = rows
        self.height = len(self.rows)
        self.width = max(map(len, self.rows))

    @classmethod
    def from_str(_cls, data):
        assert isinstance(data, str)
        return TextBlock(data.split('\n'))

    def format(self, width=None, height=None):
        if width is None: width = self.width
        if height is None: height = self.height
        return [f"{row:{width}}" for row in self.rows] + [' ' * width] * (height - self.height)

    @staticmethod
    def merge(blocks):
        return [" ".join(row) for row in zip(*blocks)]


class Matrix:
    """Общий предок для всех матриц."""

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    def __repr__(self):
        """Возвращает текстовое представление для матрицы."""
        text = [[TextBlock.from_str(f"{self[r, c]}") for c in range(self.width)] for r in range(self.height)]
        width_el = np.array(list(map(lambda row: list(map(lambda el: el.width, row)), text)))
        height_el = np.array(list(map(lambda row: list(map(lambda el: el.height, row)), text)))
        width_column = np.max(width_el, axis=0)
        width_total = np.sum(width_column)
        height_row = np.max(height_el, axis=1)
        result = []
        for r in range(self.height):
            lines = TextBlock.merge(
                text[r][c].format(width=width_column[c], height=height_row[r]) for c in range(self.width))
            for l in lines:
                result.append(f"| {l} |")
            if len(lines) > 0 and len(lines[0]) > 0 and lines[0][0] == '|' and r < self.height - 1:
                result.append(f'| {" " * (width_total + self.width)}|')
        return "\n".join(result)

    def empty_like(self, width=None, height=None):
        raise NotImplementedError

    def eye_element(self, element):
        if isinstance(element, Matrix):
            return element.eye_like()
        elif isinstance(element, np.number) or isinstance(element, numbers.Number):
            return 1
        elif isinstance(element, Fraction):
            return Fraction(1, 1)
        raise NotImplemented

    def eye_like(self, width=None, height=None):
        matix = self.empty_like()
        for i in range(min([matix.height, matix.width])):
            matix[i, i] = self.eye_element(matix[i, i])
        return matix

    def __getitem__(self, key):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, Matrix):
            assert self.width == other.width and self.height == other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r, c] = self[r, c] + other[r, c]
            return matrix
        elif isinstance(other, np.number) or isinstance(other, numbers.Number) or isinstance(other, int):
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r, c] = self[r, c] + other
            return matrix
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Matrix):
            assert self.width == other.width and self.height == other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r, c] = self[r, c] - other[r, c]
            return matrix
        elif isinstance(other, np.number) or isinstance(other, numbers.Number):
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r, c] = self[r, c] - other
            return matrix
        return NotImplemented

    #     def __rsub__(self, other):
    #         return self.__sub__(other)

    def __neg__(self):
        matrix = self.empty_like()
        for r in range(self.height):
            for c in range(self.width):
                matrix[r, c] = - self[r, c]
        return matrix

    def __mul__(self, other):
        return self.__matmul__(other)

    def __rmul__(self, other):
        return self.__matmul__(other)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            assert self.width == other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like(width=self.height, height=other.width)
            for r in range(self.height):
                for c in range(other.width):
                    acc = None
                    for k in range(self.width):
                        add = self[r, k] * other[k, c]
                        acc = add if acc is None else acc + add
                    matrix[r, c] = acc
            return matrix
        elif isinstance(other, np.number) or isinstance(other, numbers.Number):
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r, c] = self[r, c] * other
            return matrix
        return NotImplemented

    def __truediv__(self, other):
        return self * self.invert_element(other)

    def inverse(self):
        raise NotImplementedError

    def invert_element(self, element):
        if isinstance(element, numbers.Number) or isinstance(element, np.number):
            return 1 / element
        if isinstance(element, Fraction):
            return 1 / element
        if isinstance(element, Matrix):
            return element.inverse()
        raise TypeError

    def lu(self):
        assert self.width == self.height, f"Shapes does not match: {self.shape}"
        L = np.zeros((self.height, self.height), dtype=self.dtype)
        for r in range(self.height):
            L[r, r] = 1
        #         L = self.eye_like()
        U = np.zeros((self.height, self.height), dtype=self.dtype)

        for r in range(self.height):
            for c in range(self.height):
                if r <= c:
                    acc = None
                    for k in range(r):
                        add = L[r, k] * U[k, c]
                        acc = add if acc is None else (acc + add)
                    U[r, c] = self[r, c] if acc is None else (self[r, c] - acc)
                else:
                    acc = None
                    for k in range(c):
                        add = L[r, k] * U[k, c]
                        acc = add if acc is None else (acc + add)
                    L[r, c] = self[r, c] if acc is None else (self[r, c] - acc)
                    L[r, c] = L[r, c] * self.invert_element(U[c, c])
        return L, U

    def det(self):
        L, U = self.lu()
        acc = None
        for k in range(self.height):
            acc = U[k, k] if acc is None else acc * U[k, k]
        return acc

        def _LY_E_sol(self, L):
            ''' Solves LY=E'''

        Y = L.empty_like()
        for i in range(L.height):
            for j in range(L.width):
                Y[j, i] = 0 if i != j else 1
                for k in range(j):
                    Y[j, i] = Y[j, i] - L[j, k] * Y[k, i]
        return Y

    def _Ly_b_sol(self, L, b):
        ''' Solves Ly=b'''
        y = np.empty(L.height, dtype=self.dtype)
        for i in range(L.height):
            y[i] = b[i]
            for k in range(i):
                y[i] = y[i] - L[i, k] * y[k]
        return y

    def _Ux_y_sol(self, U, y):
        x = np.empty(U.height, dtype=self.dtype)
        for j in range(U.width - 1, -1, -1):
            x[j] = y[j]
            for k in range(j + 1, U.width):
                x[j] = x[j] - U[j, k] * x[k]
            x[j] = x[j] * self.invert_element(U[j, j])
        return x

    def _UX_Y_sol(self, U, Y):
        ''' Solves UX=Y'''
        X = Y.empty_like()
        for i in range(U.height):
            X[:, i] = self._Ux_y_sol(U, Y[:, i])
        return X

    def solve(self):
        pass


class FullMatrix(Matrix, ABC):
    """
    Заполненная матрица с элементами произвольного типа.
    """

    def __init__(self, data):
        """
        Создает объект, хранящий матрицу в виде np.ndarray `data`.
        """
        assert isinstance(data, np.ndarray)
        self.data = data

    def empty_like(self, width=None, height=None):
        dtype = self.data.dtype
        if width is None:
            width = self.data.shape[1]
        if height is None:
            height = self.data.shape[0]
        data = np.empty((height, width), dtype=dtype)
        return FullMatrix(data)

    @classmethod
    def zero(_cls, height, width, default=0):
        """
        Создает матрицу размера `width` x `height` со значениями по умолчанию `default`.
        """
        data = np.empty((height, width), dtype=type(default))
        data[:] = default
        return FullMatrix(data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, key):
        row, column = key
        return self.data[row, column]

    def __setitem__(self, key, value):
        row, column = key
        self.data[row, column] = value

    def lu(self):
        L, U = super().lu()
        return FullMatrix(L), FullMatrix(U)

    def inverse(self):
        '''LUX=E'''
        L, U = self.lu()
        Y = self._LY_E_sol(L)
        X = self._UX_Y_sol(U, Y)
        return (X)

