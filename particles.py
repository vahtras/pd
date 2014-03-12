"""Module defining particle class"""
import numpy as np

class PointDipole(object):
    """ A point dipole object 
    """
    def __init__(self, *args, **kwargs):
        x, y, z, q, px, py, pz, a = args
        self.r = np.array([x, y, z])
        self.q = q
        self.p = np.array([px, py, pz])
        self.a = a
        self.args = args

        self.fmt = kwargs.get('fmt', "%10.5f")

    def __str__(self):
        return self.fmt*len(self.args) % self.args

class PointDipoleList(list):
    """A list of dipole objects"""
    def __init__(self, pf):
        with open(pf) as _pf:
            units = _pf.next()
            print "units", units
            n, maxl, ipol, _ = map(int, _pf.next().split())
            print n, maxl, ipol
            for line in _pf:
                args = tuple(map(float, line.split())[1:])
                self.append(PointDipole(*args))

    def charge(self):
        return sum([p.q for p in self])

    def alpha(self):
        return sum([p.a for p in self])

    def __str__(self):
        for p in self:
            print p

            
if __name__ == "__main__":
    a = PointDipole(0,0,0,1,2,3,4,5, fmt="%5.1f")
    print a
