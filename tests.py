from MatrixLib import *
import numpy as np
from fractions import Fraction
import scipy as sp
import scipy.linalg

'''FullMatrix class testing'''
a_array = np.array([[1, 2], [3, 4]], dtype=float)
a = FullMatrix(a_array)

b_array = np.array([[a, a], [a, a]])
b = FullMatrix(b_array)

c_array = np.array([[Fraction(1, 1), Fraction(1, 2)], [Fraction(1, 3), Fraction(1, 4)]])
c = FullMatrix(c_array)

a_zero = FullMatrix(np.zeros_like(a_array))
a_eye = FullMatrix(np.eye(*np.shape(a)))

assert a.zero_like() == a_zero
assert b.zero_like() == FullMatrix(np.array([[a_zero, a_zero], [a_zero, a_zero]]))
assert c.zero_like() == a_zero

assert a.eye_like() == a_eye
assert b.eye_like() == FullMatrix(np.array([[a_eye, a_zero], [a_zero, a_eye]]))
assert c.eye_like() == a_eye

f = FullMatrix(np.array([[a, 1.0], [42, Fraction(1, 17)]]))
assert f.eye_like() == FullMatrix(np.array([[a_eye, 0], [0, 1]]))


'''LU и LUP на обычных'''
L, U = a.lu()
assert L * U == a
assert not any([L[i, i] != 1 for i in range(L.height)])
L, U, P = a.lup()
assert L * U == P * a
assert not any([L[i, i] != 1 for i in range(L.height)])

L, U = c.lu()
assert L * U == c
L, U, P = c.lup()
assert L * U == P * c


'''LU и LUP на блочных'''


def block_FullMatrix(m: int, n: int, start_from: int = 1):
    a = np.arange(start_from, start_from + m*n).astype(float).reshape((m, n))
    return FullMatrix(a)


for d_m in range(1, 4):
    d_test = np.array([[i+j*d_m+1 for j in range(d_m)] for i in range(d_m)], dtype=float)
    d = FullMatrix(np.array([[FullMatrix(np.array([[d_test[i, j]]])) for j in range(d_m)] for i in range(d_m)]))
    assert d.det() == np.linalg.det(d_test)
    L, U = d.lu()
    assert L * U == d
    L, U, P = d.lup()
    assert L * U == P * d

d_m = 2
b_m = 2
d_rand = np.random.randint(1, 100, (d_m, b_m, d_m, b_m))
d_array = 1 / d_rand
vf = np.vectorize(lambda x: Fraction(1, x))
d_array_frac = vf(d_rand)
d = FullMatrix(np.array([
    [FullMatrix(d_array_frac[i, :, j, :]) for j in range(d_m)] for i in range(d_m)
]))
real_det = np.linalg.det(d_array.reshape(b_m*d_m, b_m*d_m))

assert (d.det(2) - real_det) / real_det < 1e-8
L, U = d.lu()
assert L * U == d
L, U, P = d.lup()
assert L * U == P * d

'''нормы, для LUP '''

assert a.norm('F') == np.sqrt(np.sum(a.data**2))
fraction_to_float = np.vectorize(lambda x: x.numerator / x.denominator)
assert c.norm('F') == np.sqrt(np.sum(fraction_to_float(c.data)**2))

''' тест прогонов '''
''' _LY_B_sol и  _UX_Y_sol '''
l, u = FullMatrix(np.array([
    [1., 5., Fraction(np.random.randint(1, 10), 2)],
    [Fraction(np.random.randint(1, 10), 2), 1., 4.],
    [np.random.random(), Fraction(np.random.randint(1, 10), 3), 1.],
])).lu()
# print(l)
b = FullMatrix(np.array([[np.random.random()], [np.random.random()], [np.random.random()]]))
y = Matrix._LY_B_sol(l, b)
assert (l * y - b).norm() < 1e-10
x = Matrix._UX_Y_sol(u, y)
assert (u * x - y).norm() < 1e-10
assert (l * u * x - b).norm() < 1e-10

L, U = d.lu()
assert L * U == d
b = FullMatrix(np.array([[FullMatrix(np.random.rand(2, 2))], [FullMatrix(np.random.rand(2, 2))]]))
y = Matrix._LY_B_sol(L, b)
assert (L * y - b).norm() < 1e-10
x = Matrix._UX_Y_sol(U, y)
assert (U * x - y).norm() < 1e-10


''' solve '''
b = FullMatrix([[1], [0]])
assert (a * a.solve(b) - b).norm() < 1e-10
assert (c * c.solve(b) - b).norm() < 1e-10


''' тест обратных матриц '''

k = FullMatrix([
    [
        FullMatrix([[Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 4), Fraction(1, 5)]]),
        FullMatrix([[Fraction(1, 10), Fraction(1, 757)], [Fraction(1, 1), Fraction(1, 45)]]).eye_like()
    ],
    [
        FullMatrix([[Fraction(1, 4), Fraction(1, 47)], [Fraction(1, 52), Fraction(1, 24)]]).zero_like(),
        FullMatrix([[Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 4), Fraction(1, 5)]])
    ]
])
assert k.inverse().inverse() == k
assert k * k.inverse() == k.eye_like()

k = FullMatrix([
    [
        block_FullMatrix(2, 2, 1),
        block_FullMatrix(2, 2, 1)
    ],
    [
        0,
        block_FullMatrix(2, 2, 1)
    ]
])
assert (k - k.inverse().inverse()).norm() < 1e-10

k = FullMatrix([
    [
        FullMatrix([[Fraction(2, 1)]]),
        FullMatrix([[Fraction(3, 1)]])
    ],
    [
        FullMatrix([[Fraction(4, 1)]]),
        FullMatrix([[Fraction(1, 1)]])
    ]
])
assert k == k.inverse().inverse()

k = FullMatrix([
    [
        FullMatrix([[Fraction(1, 1), Fraction(2, 1)], [Fraction(5, 1), Fraction(6, 1)]]),
        FullMatrix([[Fraction(3, 1), Fraction(4, 1)], [Fraction(7, 1), Fraction(9, 1)]])
    ],
    [
        FullMatrix([[Fraction(10, 1), Fraction(11, 1)], [Fraction(12, 1), Fraction(13, 1)]]),
        FullMatrix([[Fraction(14, 1), Fraction(15, 1)], [Fraction(16, 1), Fraction(17, 1)]])
    ]
])

L, U = k.lu()
l, u, p = k.lup()
assert L*U == p*l*u == k
assert k.inverse().inverse() == k


'''SymmetricMatrix LDLT'''

for d_m in range(1, 3):
    for b_m in range(1, 3):
        d_rand = np.random.randint(1, 100, (d_m, b_m, d_m, b_m))
        d_array = 1 / d_rand
        vf = np.vectorize(lambda x: Fraction(x,1))
        d_array_frac = vf(d_rand)
        d = FullMatrix(np.array([
            [FullMatrix(d_array_frac[i, :, j, :]) for j in range(d_m)] for i in range(d_m)
        ]))
        real_l, real_u = FullMatrix(SymmetricMatrix(d.data).data).lu()
        L, D = SymmetricMatrix(d.data).ldl()
        assert L == real_l
        assert L * D * L.T == real_l * real_u


''' BandMatrix '''

p = {
    -1: [Fraction(-91), Fraction(17), 0],
    0: [Fraction(-93), Fraction(45), Fraction(88), Fraction(33)],
    1: [Fraction(91), Fraction(17), 0],
}
m = BandMatrix(p)
assert m.inverse() * m == m.eye_like()
assert m.inverse().inverse() == m


def FullFrac(data):
    new_data = [[Fraction(j) for j in i] for i in data]
    return FullMatrix(new_data)


p = {
    0: [FullFrac([[1, 2], [3, 4]]), FullFrac([[1, 2], [3, 4]]), FullFrac([[1, 2], [3, 4]])],
    # -1: [FullFrac([[5, 5], [3, 7]]), FullFrac([[1, 4], [3, 4]]),],
}

m = BandMatrix(p)

L, U, P = m.lup()
assert P * L * U == m
assert m.inverse() * m == m.eye_like()
