"""Module defining polarizable point dipole class""" 
import numpy as np 
from numpy.linalg import norm 
from numpy import outer, dot, array, zeros

I_3 = np.identity(3)

class PointDipole(object):
    """ A point dipole object 
    """
    def __init__(self, *args, **kwargs):
        x, y, z, q, px, py, pz, a = args
        self.r = array([x, y, z])
        self.q = q
        self.p = array([px, py, pz])
        self.a = a*I_3
        self.b = np.zeros((3, 3, 3))
        self.args = args

        self.fmt = kwargs.get('fmt', "%10.5f")
        self.__local_field = kwargs.get('local_field', 0.0)

    def __str__(self):
        return self.fmt*len(self.args) % self.args

    def charge_energy(self):
        return -self.q*dot(self.local_field, self.r)

    def permanent_dipole_energy(self):
        return -dot(self.p, self.local_field)

    def alpha_induced_dipole_energy(self):
        return -0.5*dot(self.__local_field, dot(self.a, self.__local_field))

    def beta_induced_dipole_energy(self):
        e_field = self.local_field
        return -dot(e_field, dot(dot(self.b, e_field), e_field))/6

    def total_field_energy(self):
        return \
            self.charge_energy() + \
            self.permanent_dipole_energy() + \
            self.alpha_induced_dipole_energy() + \
            self.beta_induced_dipole_energy()

    def dipole_induced(self):
        e_field = self.local_field
        return dot(self.a, e_field) + 0.5*dot(dot(self.b, e_field), e_field)

    @property
    def local_field(self):
        return self.__local_field
    @local_field.setter
    def local_field(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("Must be array")
        self.__local_field = value


class PointDipoleList(list):
    """A list of dipole objects"""
    def __init__(self, pf):
        """Class constructor 
        pf: potential file object (or iterator)
        """
        units = pf.next()
        n, maxl, ipol, _ = map(int, pf.next().split())
        for i, line in enumerate(pf):
            if i == n: break
            args = tuple(map(float, line.split())[1:])
            self.append(PointDipole(*args))

    def charge(self):
        return sum([p.q for p in self])

    def dipole_tensor(self):
        n = len(self)
        _T = zeros((n, 3, n,  3))
        for i in range(n):
            ri = self[i].r
            for j in range(i):
                rj = self[j].r
                rij = ri - rj
                rij2 = dot(rij, rij)
                Tij = (3*outer(rij, rij) - rij2*I_3)/rij2**2.5
                _T[i, :, j, :] = Tij
                _T[j, :, i, :] = Tij
        return _T

    def __str__(self):
        for p in self:
            print p

    def alpha(self):
        # Solve the response equaitons
        n = len(self)
        aT = self.dipole_tensor().reshape((n, 3, 3*n))
        # evaluate alphai*Tij
        alphas = [pd.a for pd in self]
        for i, a in enumerate(alphas):
            aT[i, :, :] = dot(a, aT[i, :, :])
        #matrix (1 - alpha*T)
        L = np.identity(3*n) - aT.reshape((3*n, 3*n))
        #right-hand-side
        dE = array(alphas).reshape((3*n, 3))
        dpdE = np.linalg.solve(L, dE).reshape((n, 3, 3))
        _alpha = dpdE.sum(axis=0)
        return _alpha

    def alpha_iso(self):
        return np.trace(self.alpha())/3


            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('potfile')
    args = parser.parse_args()
    pdl = PointDipoleList(open(args.potfile))
    print pdl.alpha()
    print pdl.alpha_iso()
