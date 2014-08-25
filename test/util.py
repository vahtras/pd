import numpy as np
from ..particles import PointDipole

def iterize(multi_line_string):
    return iter(multi_line_string.split('\n'))

EPSILON = 0.0005
ex = np.array([EPSILON/2, 0, 0])
ey = np.array([0, EPSILON/2, 0])
ez = np.array([0, 0, EPSILON/2])

def field_gradient(f):
    return np.array([gradx(f, ex), gradx(f, ey), gradx(f, ez)])

def gradx(f, eps):
    E_0 = f.__self__.local_field()
    f.__self__.set_local_field(E_0 + eps)
    f1 = f()
    f.__self__.set_local_field(E_0 - eps)
    f2 = f()
    f.__self__.set_local_field(E_0)
    return (f1 - f2)/EPSILON


def hess_zz(f):
    return hess(f, ez, ez)

def field_hessian(f):
    return np.array([
        [hess(f, ex, ex), hess(f, ex, ey), hess(f, ex, ez)],
        [hess(f, ey, ex), hess(f, ey, ey), hess(f, ey, ez)],
        [hess(f, ez, ex), hess(f, ez, ey), hess(f, ez, ez)]
        ])
        

def hess(f, e1, e2):
    E_0 = f.__self__.local_field()
    f.__self__.set_local_field(E_0 + e1 + e2)
    f12 = f()
    f.__self__.set_local_field(E_0 + e1 - e2)
    f12 -= f()
    f.__self__.set_local_field(E_0 - e1 + e2)
    f12 -= f()
    f.__self__.set_local_field(E_0 - e1 - e2)
    f12 += f()
    f.__self__.set_local_field(E_0)
    return f12/EPSILON**2


def random_scalar():
    import random
    return random.random()

def random_vector():
    return np.random.random(3)

def random_two_rank_triangular():
    return np.random.random(6)

def random_tensor():
    a = np.random.random((3, 3))
    a = 0.5*(a + a.T)
    return  a

def random_three_rank_triangular():
    return np.random.random(10)

def random_tensor2():
    b = np.random.random((3, 3, 3))
    b = b + b.transpose((1, 2, 0)) +  b.transpose((2, 0, 1)) +\
        b.transpose((1, 0, 2)) + b.transpose((2, 1, 0)) + b.transpose((0, 2, 1))
    return  b

class RandomPointDipole(PointDipole):
    
    def __init__(self, *args, **kwargs):
        PointDipole.__init__(self, *args, **kwargs)
        self._r = random_vector()
        self._q = random_scalar()
        self._p0 = random_vector()
        self._a0 = random_tensor()
        self._b0 = random_tensor2()
