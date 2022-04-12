from abc import ABC, abstractproperty, abstractmethod
import numbers
import numpy as np
from fractions import Fraction
from typing import Union, Tuple, TypeVar, Generic, Dict, Iterable


def invert_element(element):
    if isinstance(element, numbers.Number) or isinstance(element, np.number):
        return 1 / element
    if isinstance(element, Fraction):
        return 1 / element
    if isinstance(element, Matrix):
        return element.inverse()
    raise TypeError


def norm_element(element, norm):
    if isinstance(element, Matrix):
        return element.norm(norm)
    # elif isinstance(element, np.number) or isinstance(element, numbers.Number):
    else:
        return abs(element)


def eye_element(element):
    if isinstance(element, Matrix):
        return element.eye_like()
    elif isinstance(element, np.number) or isinstance(element, numbers.Number):
        return 1
    elif isinstance(element, Fraction):
        return Fraction(1, 1)
    raise TypeError(f'Can`t find eye element for type {type(element)}')


def zero_element(element):
    if isinstance(element, Matrix):
        return element.zero_like()
    elif isinstance(element, np.number) or isinstance(element, numbers.Number):
        return 0
    elif isinstance(element, Fraction):
        return Fraction(0, 1)
    raise TypeError(f'Can`t find zero element for type {type(element)}')


def transpose_element(element):
    if isinstance(element, Matrix):
        return element.T
    else:
        return element


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


class Matrix(ABC):
    """Общий предок для всех матриц."""

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError

    @property
    @abstractmethod
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

    def __eq__(self, other):
        if type(self) != type(other) and not isinstance(other, Matrix):
            if not (isinstance(other, np.number) or isinstance(other, numbers.Number) or isinstance(other, int)):
                return False
            else:
                return self == self.eye_like() * other
        for i in range(self.height):
            for k in range(self.width):
                if self[i, k] != other[i, k]:
                    return False
        return True

    def __add__(self, other):
        if isinstance(other, Matrix):
            assert self.width == other.width and self.height == other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r, c] = self[r, c] + other[r, c]
            return matrix
        elif isinstance(other, np.number) or isinstance(other, numbers.Number) or isinstance(other, int):
            # matrix = self.empty_like()
            # for r in range(self.height):
            #     for c in range(self.width):
            #         matrix[r, c] = self[r, c] + other
            # return matrix
            return self + self.eye_like() * other
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Matrix):
            assert self.width == other.width and self.height == other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r, c] = self[r, c] - other[r, c]
            return matrix
        elif isinstance(other, np.number) or isinstance(other, numbers.Number):
            # matrix = self.empty_like()
            # for r in range(self.height):
            #     for c in range(self.width):
            #         matrix[r, c] = self[r, c] - other
            # return matrix
            return self - self.eye_like() * other
        return NotImplemented

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        matrix = self.empty_like()
        for r in range(self.height):
            for c in range(self.width):
                matrix[r, c] = - self[r, c]
        return matrix

    def __mul__(self, other):
        return self.__matmul__(other)

    def __rmul__(self, other):
        if not isinstance(other, Matrix):
            return self.__matmul__(other)
        raise ValueError

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            assert self.width == other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like(height=self.height, width=other.width)
            for r in range(self.height):
                for c in range(other.width):
                    acc = None
                    for k in range(self.width):
                        add = self[r, k] * other[k, c]
                        acc = add if acc is None else acc + add
                        # print(type(acc))
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
        return self * invert_element(other)

    def __abs__(self):
        matrix = self.empty_like()
        for i in range(matrix.height):
            for j in range(matrix.width):
                matrix[i, j] = abs(self[i, j])
        return matrix

    def norm(self, norm='F'):
        if norm == 'F':
            res = 0
            for i in range(self.height):
                for j in range(self.width):
                    res += (norm_element(self[i, j], norm)) ** 2
            if isinstance(res, Fraction):
                return np.sqrt(res.numerator / res.denominator)
            return np.sqrt(res)
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractmethod
    def empty_like(self, width=None, height=None):
        pass

    @abstractmethod
    def eye_like(self: 'Matrix') -> 'Matrix':
        pass

    @abstractmethod
    def zero_like(self: 'Matrix') -> 'Matrix':
        pass

    @abstractmethod
    def inverse(self: 'Matrix') -> 'Matrix':
        pass

    def lu(self):
        assert self.width == self.height, f"Shapes does not match: {self.shape}"
        L = self.eye_like()
        U = self.zero_like()
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
                    L[r, c] = L[r, c] * invert_element(U[c, c])
        return L, U

    def lup(self):
        assert self.width == self.height, f"Shapes does not match: {self.shape}"
        L = self.eye_like()
        P = self.eye_like()
        Pr = list(range(self.height))
        U = self.zero_like()
        for r in range(self.height):
            max_r = r + np.argmax([norm_element(self[Pr[i], r], 'F') for i in range(r, self.height)])
            new_r = Pr[max_r]
            Pr[max_r], Pr[r] = Pr[r], Pr[max_r]
            for c in range(self.width):
                if r <= c:
                    acc = None
                    for k in range(r):
                        add = L[r, k] * U[k, c]
                        acc = add if acc is None else (acc + add)
                    U[r, c] = self[new_r, c] if acc is None else (self[new_r, c] - acc)
                else:
                    acc = None
                    for k in range(c):
                        add = L[r, k] * U[k, c]
                        acc = add if acc is None else (acc + add)
                    L[r, c] = self[new_r, c] if acc is None else (self[new_r, c] - acc)
                    L[r, c] = L[r, c] * invert_element(U[c, c])
        for p in range(self.width):
            if Pr[p] != p:
                P[p, p] = 0
                P[Pr[p], p] = 1
        return L, U, P

    def det(self, n: int = None):
        if isinstance(n, int) and n < 1:
            return ValueError('n has to be int >=1 or None')
        L, U = self.lu()
        acc = None
        for k in range(self.height):
            if n is None:
                new_n = None
            else:
                new_n = n - 1
            if isinstance(U[k, k], Matrix) and new_n != 0:
                acc = U[k, k].det(new_n) if acc is None else acc * U[k, k].det(new_n)
            else:
                acc = U[k, k] if acc is None else acc * U[k, k]
        return acc

    @classmethod
    def _LY_B_sol(cls, l: 'Matrix', b: 'Matrix') -> 'Matrix':
        """ Solves LY=B"""
        if (l.width != b.height) or (l.height != l.width):
            raise ValueError
        Y = b.empty_like()
        for i in range(b.width):
            for j in range(l.width):
                Y[j, i] = b[j, i]
                for k in range(j):
                    Y[j, i] = Y[j, i] - l[j, k] * Y[k, i]
        # print((l * Y - b).norm())
        return Y

    @classmethod
    def _UX_Y_sol(cls, u: 'Matrix', y: 'Matrix') -> 'Matrix':
        """ Solves UX=Y """
        if (u.height != y.height) or (u.height != u.width):
            raise ValueError
        X = y.empty_like()
        for i in range(X.width):
            for j in range(u.width - 1, -1, -1):
                X[j, i] = y[j, i]
                for k in range(j + 1, u.width):
                    X[j, i] = X[j, i] - u[j, k] * X[k, i]
                X[j, i] = invert_element(u[j, j]) * X[j, i]
        return X

    def solve(self, b: 'Matrix'):
        return self.inverse() * b

    @property
    def T(self):
        matrix = self.empty_like(self.height, self.width)
        for i in range(matrix.width):
            for j in range(matrix.height):
                matrix[j, i] = transpose_element(self[i, j])
        return matrix


class FullMatrix(Matrix, ABC):
    """
    Заполненная матрица с элементами произвольного типа.
    """

    def __init__(self, data: Union[np.ndarray, list]):
        if isinstance(data, list):
            data = np.array(data)
        assert isinstance(data, np.ndarray)
        self.data = data

    def empty_like(self, width=None, height=None):
        dtype = self.data.dtype
        dtype = object  
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        data = np.empty((height, width), dtype=dtype)
        return FullMatrix(data)

    @classmethod
    def zero(cls, height, width, default=0):
        """
        Создает матрицу размера `width` x `height` со значениями по умолчанию `default`.
        """
        data = np.empty((height, width), dtype=type(default))
        data[:] = default
        return cls(data)

    def zero_like(self):
        matix = self.empty_like()
        for i in range(self.width):
            for j in range(self.height):
                matix[i, j] = zero_element(self[i, j])
        return matix

    def eye_like(self) -> 'FullMatrix':
        matix = self.empty_like()
        for i in range(self.height):
            for j in range(self.width):
                if i != j:
                    matix[i, j] = zero_element(self[i, j])
                else:
                    matix[i, i] = eye_element(self[i, i])
        return matix

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
        return L, U

    def inverse(self):
        """ PLUX=E """
        L, U, P = self.lup()
        # assert (P * L * U - self).norm() < 1e-10
        Y = self._LY_B_sol(L, P)
        X = self._UX_Y_sol(U, Y)
        # assert (L * U * X - P).norm() < 1e-10
        # L, U = self.lu()
        # Y = self._LY_B_sol(L, L.eye_like())
        # X = self._UX_Y_sol(U, Y)
        return X


class SymmetricMatrix(FullMatrix):
    def __init__(self, data: Union[np.ndarray, Iterable], side='u'):
        if isinstance(data, Iterable):
            data = np.array(data)
        if side not in ['u', 'd']:
            raise ValueError
        assert data.shape[0] == data.shape[1], f"Shapes does not match: {self.shape}"
        for i in range(data.shape[0]):
            for j in range(i+1, data.shape[1]):
                if side == 'u':
                    data[j, i] = transpose_element(data[i, j])
                elif side == 'd':
                    data[i, j] = transpose_element(data[j, i])
                else:
                    raise ValueError
            if isinstance(data[i, i], FullMatrix):
                data[i, i] = SymmetricMatrix(data[i, i].data, side=side)
            elif isinstance(data[i, i], BandMatrix):
                # data[i, i] = BandMatrix(data[i, i].data, side=side)
                raise NotImplementedError
        super().__init__(data)

    def ldl(self) -> Tuple['Matrix', 'Matrix']:
        L = self.eye_like()
        D = self.eye_like()
        for c in range(self.height):
            for r in range(c, self.width):
                if r == c:
                    acc = None
                    for k in range(r):
                        add = L[r, k] * D[k, k] * transpose_element(L[r, k])
                        acc = add if acc is None else (acc + add)
                    D[r, r] = self[r, r] if acc is None else (self[r, r] - acc)
                else:
                    acc = None
                    for k in range(c):
                        add = L[r, k] * D[k, k] * transpose_element(L[c, k])
                        acc = add if acc is None else (acc + add)
                    L[r, c] = self[r, c] if acc is None else (self[r, c] - acc)
                    L[r, c] = L[r, c] * invert_element(D[c, c])
        return L, D


class BandMatrix(FullMatrix):
    def __init__(self, data: Dict[int, Union[np.ndarray, Iterable]]):
        self.ds = list(sorted(data.keys(), reverse=True))
        new_data = []
        min_d = self.ds[np.argmin(abs(np.array(self.ds)))]
        N = abs(min_d) + len(data[min_d])
        self.N = N
        for d in self.ds:
            if len(data[d]) != N - abs(d):
                raise ValueError(f'{d} must contain {N - abs(d)} items')
            if -d in self.ds and (len(data[d]) != len(data[-d])):
                raise ValueError(f'{d} and {-d} must have the same length')
            new_data.append(np.concatenate([data[d], np.zeros(abs(d))]))
        super(BandMatrix, self).__init__(np.array(new_data))

    def __getitem__(self, key):
        row, column = key
        if column-row not in self.ds:
            return 0
        return self.data[self.ds.index(column-row), min(row, column)]

    def __setitem__(self, key, value):
        row, column = key
        if column - row not in self.ds:
            raise IndexError('You will break the band matrix')
        self.data[self.ds.index(column-row), min(row, column)] = value

    @property
    def shape(self):
        return self.N, self.N

    @property
    def T(self):
        raise NotImplementedError
