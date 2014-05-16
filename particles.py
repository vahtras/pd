"""Module defining polarizable point dipole class""" 
import numpy as np 
from numpy.linalg import norm 
from numpy import outer, dot, array, zeros

I_3 = np.identity(3)

def header_to_dict(header):
    """Transfer header string in pot file to dict data"""

    header_dict = {
        "#atoms": 0, 
        "max_angmom": 0, 
        "iso_pol": False, 
        "full_pol": False, 
        "hyp_pol": False, 
        }

    header_data = map(int, header.split())
    header_dict["#atoms"] = header_data[0]
    header_dict["max_angmom"] = header_data[1]
    header_dict["iso_pol"] = len(header_data) > 2 and header_data[2] == 1
    header_dict["full_pol"] = len(header_data) > 2 and header_data[2] == 2
    header_dict["hyp_pol"] = len(header_data) > 3 and header_data[3] == 1

    return header_dict

def line_to_dict(header_dict, line):
    """Transfer line data to dictonary"""


    line_data = map(float, line.split()[1:])
    line_dict = {}

    line_dict['coordinates'] = line_data[:3]
    line_dict['dipole'] = None
    nextstart = 3

    max_angmom = header_dict.get('max_angmom', 0)
    iso_pol = header_dict.get('iso_pol', False)
    full_pol = header_dict.get('full_pol', False)
    hyp_pol = header_dict.get('hyp_pol', False)

    if max_angmom >= 0: 
        nextend = nextstart + 1
        line_dict['charge'] = line_data[nextstart]
        nextstart = nextend
    if max_angmom >= 1: 
        nextend = nextstart + 3
        line_dict['dipole'] = line_data[nextstart: nextend]
        nextstart = nextend
    if max_angmom >= 2: 
        nextend = nextstart + 6
        line_dict['quadrupole'] = line_data[nextstart: nextend]
        nextstart = nextend

    if iso_pol:
        nextend = nextstart + 1
        line_dict['iso_alpha'] = line_data[nextstart]
    else:
        line_dict['iso_alpha'] = 0

    if full_pol:
        nextend = nextstart + 6
        line_dict['alpha'] = line_data[nextstart: nextend]
    else:
        line_dict['alpha'] = np.zeros(6)

    if hyp_pol:
        nextend = nextstart + 18
        line_dict['beta'] = line_data[nextstart: nextend]
    else:
        line_dict['beta'] = np.zeros(27)
    
    return line_dict

class PointDipole(object):
    """ A point dipole object 
    """
    def __init__(self, *args, **kwargs):
        self.r = array(kwargs['coordinates'])
        self.q = kwargs['charge']
        self.p = array(kwargs["dipole"])
        self.a = kwargs["iso_alpha"]*I_3
        self.b = array(kwargs["beta"])
        self.args = args

        self.fmt = kwargs.get('fmt', "%10.5f")
        self.__local_field = kwargs.get('local_field', 0.0)

    def __str__(self):
        #for isotropic alpha
        value_line = list(self.r) + [self.q] + list(self.p) + [self.a[0, 0]]
        return "1" + self.fmt*len(value_line) % tuple(value_line)

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
        header_dict = header_to_dict(pf.next())
        for i, line in enumerate(pf):
            if i == header_dict["#atoms"]: break
            line_dict = line_to_dict(header_dict, line)
            self.append(PointDipole(**line_dict))

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
        return "\n\n".join([str(p) for p in self])

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
