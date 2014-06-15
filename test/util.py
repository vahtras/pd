import numpy as np

def iterize(multi_line_string):
    return iter(multi_line_string.split('\n'))

EPSILON = 0.001
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


