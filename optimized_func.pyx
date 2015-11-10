import numpy as np
cimport numpy as np


class SCFNotConverged( Exception ):
    def __init__(self, residual, threshold):
        self.residual = residual
        self.threshold = threshold

def field_at( np.ndarray[double,ndim=1] r,
        np.ndarray[double,ndim=1] r0,
        double q,
        np.ndarray[double,ndim=1] p):
    return monopole_field_at(r, r0, q) + dipole_field_at(r, r0, p)

def monopole_field_at( np.ndarray[double,ndim=1]r,
        np.ndarray[double,ndim=1]r0,
        double q):
    dr = r - r0
    dr2 = np.dot(dr, dr)
    if dr2 < .1: raise Exception("Nuclei too close")
    return q*dr/dr2**1.5

def dipole_field_at( np.ndarray[double,ndim=1]r,
        np.ndarray[double,ndim=1]r0,
        np.ndarray[double,ndim=1]p):
    dr = r - r0
    dr2 = np.dot(dr, dr)
    if dr2 < .1: raise Exception("Nuclei too close")
    return (3*dr*np.dot(dr, p) - dr2*p)/dr2**2.5

def evaluate_field_at_atoms_cython( 
        np.ndarray[long,ndim=1]particles,
        np.ndarray[double,ndim=1]E,
        np.ndarray[double,ndim=2]_r,
        np.ndarray[double,ndim=1]_q,
        np.ndarray[double,ndim=2]_p0,
        np.ndarray[double,ndim=3]_a0,
        np.ndarray[double,ndim=4]_b0,
        np.ndarray[double,ndim=2]_field,
        external=None):
    _Cell = None
    if _Cell is not None:
        for p in range(len(particles)):
            tmp = np.zeros( (3,) )
            for o in range(len(particles)):
                if particles[p] == particles[o]:
                    continue
                od = _p0[o] + np.dot(_a0[o], _field[o])\
                        + 0.5*np.dot( np.dot( _b0[o], _field[o]), _field[o] )
                tmp += field_at( _r[p], _r[o], _q[o], od )
            _field[p] = tmp 
    else:
        for p in range(len(particles)):
            tmp = np.zeros( (3,) )
            for o in range(len(particles)):
                if particles[p] == particles[o]:
                    continue
                od = _p0[o] + np.dot(_a0[o], _field[o])\
                        + 0.5*np.dot( np.dot( _b0[o], _field[o]), _field[o] )
                tmp += field_at( _r[p], _r[o], _q[o], od )
            _field[p] = tmp 

    if external is not None:
        for i in range(len(_field)):
            _field[i] += external
    return np.array( _field )

def solve_scf_for_external_cython( np.ndarray[long,ndim=1]particles,
        np.ndarray[double,ndim=1]E,
        np.ndarray[double,ndim=2]_r,
        np.ndarray[double,ndim=1]_q,
        np.ndarray[double,ndim=2]_p0,
        np.ndarray[double,ndim=3]_a0,
        np.ndarray[double,ndim=4]_b0,
        np.ndarray[double,ndim=2]_field,
        int max_it = 100,
        double threshold = 1e-8 ):
    """An optimized version of solve_scf_for_external which explicitly takes
    arrays of the particles properties, and outputs the local field at each pointa
    after the residual is below threshold"""
    E_p0 = np.zeros((len(particles), 3))
    for i in range(max_it):
        E_at_p = evaluate_field_at_atoms_cython( particles,
                E,
                _r,
                _q,
                _p0,
                _a0,
                _b0,
                _field,
                external=E )
        #print i, E_at_p
        for i in range( len(E_at_p) ):
            _field[i] = E_at_p[i]
        residual = np.linalg.norm( E_p0 - E_at_p)
        print residual, threshold
        if residual < threshold:
            return E_at_p, i, residual
        E_p0[:, :] = E_at_p

    raise SCFNotConverged(residual, threshold)


