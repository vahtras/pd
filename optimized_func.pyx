import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow
from libc.stdio cimport printf
from libc.stdlib cimport exit
from cython.parallel import prange
cimport cython
cimport openmp

class SCFNotConverged(Exception):
    def __init__(self, residual, threshold):
        self.residual = residual
        self.threshold = threshold

@cython.boundscheck( False )
def solve_scf_for_external_cython(
        np.ndarray[long,ndim=1]particles,
        np.ndarray[double,ndim=1]E,
        np.ndarray[double,ndim=2]_r,
        np.ndarray[double,ndim=1]_q,
        np.ndarray[double,ndim=2]_p0,
        np.ndarray[double,ndim=3]_a0,
        np.ndarray[double,ndim=4]_b0,
        np.ndarray[double,ndim=2]_field,
        int max_it = 100,
        double threshold = 1e-8,
        int num_threads = 1
        ):
    """An optimized version of solve_scf_for_external which explicitly takes
    arrays of the particles properties, and outputs the local field at each pointa
    after the residual is below threshold"""

    cdef int Nmax = particles.shape[0]

    cdef np.ndarray[double, ndim=2] E_p0 = np.zeros( (Nmax, 3))
    cdef np.ndarray[double, ndim=2] E_at_p = np.zeros( (Nmax, 3 ))
    cdef double[:,:] para_field = _field
    E_at_p[:, :] = _field

    cdef int i, o, p, k, j, l
    cdef double residual, dr2, tmp_dr, tmp_x, tmp_y, tmp_z

    cdef double rx, ry, rz
    cdef double px, py, pz

    cdef double ax, ay, az, bx, by, bz

#dipole components for particle o
    cdef double p_o_x, p_o_y, p_o_z

#monopole/dipole x, y, z component from particle o
    cdef double proj_o
    cdef double mx, my, mz
    cdef double dx, dy, dz
    cdef double tmp_E_x, tmp_E_y,tmp_E_z

    for i in range( max_it ):
#Parallelize over outer loop, over each particle p
        for p in prange( Nmax, num_threads = num_threads,
                nogil = True):
            tmp_E_x = 0
            tmp_E_y = 0
            tmp_E_z = 0
            for o in range( Nmax ):
                if particles[p] == particles[o]:
                    continue
                #od = c_od( od, _field, _p0, _a0, _b0 )
                ax = 0
                ay = 0
                az = 0
                bx = 0
                by = 0
                bz = 0
                for k in range(3):
                    ax = ax + _a0[o, 0, k ] * E_at_p[o, k]
                    ay = ay + _a0[o, 1, k ] * E_at_p[o, k]
                    az = az + _a0[o, 2, k ] * E_at_p[o, k]
                    for j in range(3):
                        bx = bx + 0.5*(_b0[o, 0, k, j ] * E_at_p[o, k]* E_at_p[o, j])
                        by = by + 0.5*(_b0[o, 1, k, j ] * E_at_p[o, k]* E_at_p[o, j])
                        bz = bz + 0.5*(_b0[o, 2, k, j ] * E_at_p[o, k]* E_at_p[o, j])
                p_o_x = _p0[o, 0] + ax + bx
                p_o_y = _p0[o, 1] + ay + by
                p_o_z = _p0[o, 2] + az + bz

                #printf( "p at o: %f, %f, %f\n", p_o_x, p_o_y , p_o_z ) 
                # + adot[ :] \
                        #+ 0.5*np.dot( np.dot( _b0[o], _field[o]), _field[o] )
#Calculate distance vectors
                rx = _r[p, 0] - _r[o, 0]
                ry = _r[p, 1] - _r[o, 1]
                rz = _r[p, 2] - _r[o, 2]
#This is the distance using norm using clib.math.sqrt
                dr2 = rx*rx + ry*ry + rz*rz

                mx = _q[o] * rx / pow(dr2, 1.5 )
                my = _q[o] * ry / pow(dr2, 1.5 )
                mz = _q[o] * rz / pow(dr2, 1.5 )

                proj_o = rx*p_o_x + ry*p_o_y + rz*p_o_z

                dx = (3*rx*proj_o - dr2 * p_o_x)/pow( dr2, 2.5 )
                dy = (3*ry*proj_o - dr2 * p_o_y)/pow( dr2, 2.5 )
                dz = (3*rz*proj_o - dr2 * p_o_z)/pow( dr2, 2.5 )

                tmp_E_x = tmp_E_x + mx + dx
                tmp_E_y = tmp_E_y + my + dy
                tmp_E_z = tmp_E_z + mz + dz

            _field[p,0] = tmp_E_x
            _field[p,1] = tmp_E_y
            _field[p,2] = tmp_E_z

        if E is not None:
            _field[p, 0] += E[0]
            _field[p, 1] += E[1]
            _field[p, 2] += E[2]

        E_at_p[:, :] = _field
        residual = np.linalg.norm( E_p0 - E_at_p)
        print residual, threshold
        if residual < threshold:
            return E_at_p, i, residual
        E_p0[:, :] = E_at_p
    raise SCFNotConverged(residual, threshold)


