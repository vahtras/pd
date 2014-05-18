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
    f.__self__.local_field += eps
    f1 = f()
    f.__self__.local_field -= 2*eps
    f2 = f()
    f.__self__.local_field += eps
    return (f1 - f2)/EPSILON
