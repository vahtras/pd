"""Module defining polarizable point dipole class""" 
import numpy as np 
from numpy.linalg import norm 
from numpy import outer, dot, array, zeros

I_3 = np.identity(3)
ORIGO = np.zeros(3)
BETA_ZERO = np.zeros((3, 3, 3))

class PointDipoleList(list):
    """A list class of ``PointDipole`` objects"""

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

    @staticmethod
    def from_string(potential):
        """Used to build the ``PointDipoleList`` object when the
           potential file is given as a triple quoted string with newlines
        """
        return PointDipoleList(iter(potential.split("\n")))
        
    def charge(self):
        """Returns sum of charges of particles in the list"""
        return sum([p.charge() for p in self])

    def dipole_tensor(self):
        """Calculates the dipole coupling, tensor, describing the
        electric field strength at a given particle due to
        another electric dipole:

        .. math::
            \mathbf{T}_{ij} = (3\mathbf{r}_{ij}\mathbf{r}_{ij}-r_{ij}^2\mathbf{1})/r_{ij}^5

        """

        n = len(self)
        _T = zeros((n, 3, n,  3))
        for i in range(n):
            ri = self[i]._r
            for j in range(i):
                rj = self[j]._r
                rij = ri - rj
                rij2 = dot(rij, rij)
                Tij = (3*outer(rij, rij) - rij2*I_3)/rij2**2.5
                _T[i, :, j, :] = Tij
                _T[j, :, i, :] = Tij
        return _T

    def __str__(self):
        return "\n\n".join([str(p) for p in self])

    def alpha_iso(self):
        r"""Return the isotropic polarizability

        .. math::
            \alpha_{iso} = \sum_k \alpha_{kk}/3
        """

        return np.trace(self.alpha())/3

    def alpha(self):
        dpdE = self.solve_Applequist_equation()
        return dpdE.sum(axis=0)

    def solve_Applequist_equation(self):
        # Solve the response equaitons
        n = len(self)
        dE = self.form_Applequist_rhs()
        L = self.form_Applequist_coefficient_matrix()
        dpdE = np.linalg.solve(L, dE).reshape((n, 3, 3))
        return dpdE

    def form_Applequist_rhs(self):
        n = len(self)
        alphas = [pd._a0 for pd in self]
        dE = array(alphas).reshape((n*3, 3))
        return dE

    def form_Applequist_coefficient_matrix(self):
        n = len(self)
        aT = self.dipole_tensor().reshape((n, 3, 3*n))
        # evaluate alphai*Tij
        alphas = [pd._a0 for pd in self]
        for i, a in enumerate(alphas):
            aT[i, :, :] = dot(a, aT[i, :, :])
        #matrix (1 - alpha*T)
        L = np.identity(3*n) - aT.reshape((3*n, 3*n))
        return L

    def solve_scf_for_external(self, E, max_it=100, threshold=1e-6):
        E_p0 = np.zeros((len(self), 3))
        for i in range(max_it):
            E_at_p =  self.evaluate_field_at_atoms(external=E)
            for p, Ep in zip(self, E_at_p):
                p.local_field = Ep
            residual = norm(E_p0 - E_at_p)
            if residual < threshold:
                return i, residual
            E_p0[:, :] = E_at_p
        raise Exception("SCF not converged")

    def evaluate_field_at_atoms(self, external=None):
        E_at_p =  [
            array(
                [o.field_at(p._r) for o in self if o is not p]
                ).sum(axis=0) 
            for p in self
            ]
        if external is not None:
            E_at_p = [external + p for p in E_at_p]

        return E_at_p

    def _intermediate_product_TB(self):
        """sum(jk)T[i,:, j, :]B[j, :, k, :]"""
        n = len(self)
        T = self.dipole_tensor().reshape(n, 3, n*3)
        B = self.solve_Applequist_equation().reshape(n*3, 3)
        TB = dot(T, B)
        return TB

    def _intermediate_C(self):
        """sum(jk)T[i,:, j, :]B[j, :, k, :]"""
        n = len(self)
        TB = self._intermediate_product_TB()
        C = np.array([I_3 + tb for tb in TB])
        return C
            


class PointDipole(object):
    r""" 
    A hyperpolarizable dipole object with dipole moment

    .. math::
        \bar{p} = \bar{p}^0 + \bar{\bar\alpha}\cdot\bar{E} + \frac12 \bar{\bar\beta}:\bar{E}\bar{E}
            
    """
    def __init__(self, *args, **kwargs):
        """
        fixed quantities: 
           r: coordinates
           q: charge
           _p0: permanent dipole
            _a0: polarizability tensor
            b: hyperpolarizability tensor
        
        derived quantities 
        
           dp: induced dipole moment
           p:  total dipole moment
        
        """

        self._r = array(kwargs.get('coordinates', np.zeros(3)))

        if 'charge' in kwargs:
            self._q = kwargs['charge']
        else:
            self._q = 0

        if "dipole" in kwargs:
            self._p0 = array(kwargs["dipole"])
        else:
            self._p0 = ORIGO
        self._a0 = kwargs.get("iso_alpha", 0)*I_3
        self.b = kwargs.get("beta", BETA_ZERO)
        self.args = args

        self.fmt = kwargs.get('fmt', "%10.5f")
        self.local_field = kwargs.get('local_field', np.zeros(3))

    def charge(self):
        return self._q

    @property
    def dp(self):
        return self.dipole_induced()

    @property
    def p0(self):
        return self._p0

    @property
    def p(self):
        if self._p0 is None:
            return self.dp
        else:
            return self._p0 + self.dp

    @property
    def da(self):
        return self.alpha_induced()

    @property
    def a(self):
        return self._a0 + self.da

    def alpha_induced(self):
        return dot(self.b, self.local_field)
            

    def __str__(self):
        """The output simulate the line of a potential input file"""
        #for isotropic alpha
        value_line = list(self._r) + [self._q]
        if self._p0 is not None:
            value_line += list(self._p0)
        if self._a0 is not None:
            value_line +=  [self._a0[0, 0]]
        return "1" + self.fmt*len(value_line) % tuple(value_line)

    def charge_energy(self):
        return -self._q*dot(self.local_field, self._r)

    def permanent_dipole_energy(self):
        return -dot(self._p0, self.local_field)

    def alpha_induced_dipole_energy(self):
        return -0.5*dot(self.local_field, dot(self._a0, self.local_field))

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
        return dot(self._a0, e_field) + 0.5*dot(dot(self.b, e_field), e_field)

    def monopole_field_at(self, r):
        dr = r - self._r
        dr2 = dot(dr, dr)
        return self._q*dr/dr2**1.5

    def dipole_field_at(self, r):
        dr = r - self._r
        dr2 = dot(dr, dr)
        return (3*dr*dot(dr, self.p) - dr2*self.p)/dr2**2.5

    def field_at(self, r):
        return self.monopole_field_at(r) + self.dipole_field_at(r)


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
    header_dict["hyp_pol"] = len(header_data) > 4 and header_data[3] == 1

    return header_dict

def line_to_dict(header_dict, line):
    """Transfer line data to dictonary"""


    line_data = map(float, line.split()[1:])
    line_dict = {}

    line_dict['coordinates'] = line_data[:3]
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
        #line_dict['beta'] = np.zeros(27)
        pass
    
    return line_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('potfile')
    args = parser.parse_args()
    pdl = PointDipoleList(open(args.potfile))
    print pdl.alpha()
    print pdl.alpha_iso()
