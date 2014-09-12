#!/usr/bin/env python 
#-*- coding: utf-8 -*-

import numpy as np 
from math import erf
from numpy.linalg import norm , tensorinv
from numpy import outer, dot, array, zeros, einsum, diag
import ut
from particles import PointDipole, PointDipoleList, header_to_dict, line_to_dict

I_3 = np.identity(3)
ZERO_VECTOR = np.zeros(3)
ALPHA_ZERO = np.zeros((3, 3))
BETA_ZERO = np.zeros((3, 3, 3))

class GaussianQuadrupoleList( PointDipoleList ):
    """
    A list class of ``GaussianQuadrupole`` objects

    Overrides the functions:

    __init__
    from_string
    append

    """

    def __init__(self, pf=None):
        """Class constructor 
        pf: potential file object (or iterator)
        """
        if pf is not None:
            units = pf.next()
            self.header_dict = header_to_dict( pf.next() )
            for i, line in enumerate(pf):
                if i == self.header_dict["#atoms"]: break
                line_dict = line_to_dict( self.header_dict, line)
                self.append( GaussianQuadrupole(**line_dict) )

    @staticmethod
    def from_string(potential):
        """Used to build the ``GaussianQuadrupoleList`` object when the
           potential file is given as a triple quoted string with newlines
        """
        return GaussianQuadrupoleList(iter(potential.split("\n")))

    def append(self, arg):
        """Overriding superclass list append: check if arg is GaussianQuadrupole"""
        if not isinstance(arg,  GaussianQuadrupole):
            print "GaussianQuadrupoleList.append called with object of type", type(arg)
            raise TypeError
        super(GaussianQuadrupoleList, self).append(arg)

    def solve_scf(self, max_it=100, threshold=1e-6):
        self.solve_scf_for_external(ZERO_VECTOR, max_it, threshold)

    def solve_scf_for_external(self, E, max_it=100, threshold=1e-8):
        E_p0 = np.zeros((len(self), 3))
        for i in range(max_it):
            E_at_p =  self.evaluate_field_at_atoms(external=E)
            #print i, E_at_p
            for p, Ep in zip(self, E_at_p):
                p.set_local_field(Ep)
            residual = norm(E_p0 - E_at_p)
            if residual < threshold:
                return i, residual
            E_p0[:, :] = E_at_p
        raise SCFNotConverged(residual, threshold)

    def evaluate_field_at_atoms(self, external=None):
        E_at_p =  [
            array(
                [o.field_at(p._r) for o in self if not o.in_group_of(p)]
                ).sum(axis=0) 
            for p in self
            ]
        if external is not None:
            E_at_p = [external + p for p in E_at_p]

        return array(E_at_p)

    def dipole_coupling_tensor(self):
        """Calculates the dipole coupling, tensor, describing the
        electric field strength at a given particle due to
        another electric dipole distribution:

        .. math::
            \mathbf{T}_{ij} = (3\mathbf{r}_{ij}\mathbf{r}_{ij}-r_{ij}^2\mathbf{1})/r_{ij}^5

        """
        n = len(self)
        _T = zeros((n, 3, n,  3))
        invpi = 1/ np.sqrt( np.pi )
        R = self[0]._R_p

        for i in range(n):
            ri = self[i]._r
            for j in range(i):

                if self[i].in_group_of( self[j] ):
                    continue

                rj = self[j]._r
                rij = ri - rj
                rij2 = dot(rij, rij)

                first = erf( rij2**0.5 / R)
                second = 2 * invpi * rij2**0.5 / R * np.exp( -rij2 /R**2 )
                third = 4*invpi/R**3* outer( rij , rij ) / rij2 * np.exp( -rij2 /R**2)
                Tij =  (3* outer( rij, rij ) - rij2 * I_3 )/ rij2**2.5 *(first - second) - third

                _T[i, :, j, :] = Tij
                _T[j, :, i, :] = Tij


        #for i, book in enumerate (_T):
        #    for j, page in enumerate(book):
        #        for k, row in enumerate(page):
        #            for l, col in enumerate(row):
        #                _T[i, j, k, l] = 0.0
        return _T

    def solve_Applequist_equation(self):
        # Solve the linear response equaitons
        n = len(self)
        try:
            self.solve_scf_for_external(ZERO_VECTOR)
        except SCFNotConverged as e:
            print "SCF Not converged: residual=%f, threshold=%f"% (
                float(e.residual), float(e.threshold)
                )
            raise
        dE = self.form_Applequist_rhs()
        L = self.form_Applequist_coefficient_matrix()
        dpdE = np.linalg.solve(L, dE).reshape((n, 3, 3))
        return dpdE

    def form_Applequist_rhs(self):
        n = len(self)
        alphas = [pd.a for pd in self]
        dE = array(alphas).reshape((n*3, 3))
        return dE

    def form_Applequist_coefficient_matrix(self):
        n = len(self)
        aT = self.dipole_coupling_tensor().reshape((n, 3, 3*n))
        # evaluate alphai*Tij
        alphas = [pd.a for pd in self]
        for i, a in enumerate(alphas):
            aT[i, :, :] = dot(a, aT[i, :, :])
        #matrix (1 - alpha*T)
        L = np.identity(3*n) - aT.reshape((3*n, 3*n))
        return L

    def alpha(self):
        try:
            dpdF = self.solve_Applequist_equation()
        except SCFNotConverged:
            return np.zeros((3, 3))
        return dpdF.sum(axis=0)




class GaussianQuadrupole( PointDipole ):
    """ 

    Inherits PointDipole with new attributes:

    _R_q: Charge standard deviation
    _R_p: Dipole standard deviation
    _Q0 : Permanent quadrupole moment

    Overrides the functions:
    
    field_at
    monopole_field_at
    dipole_field_at

    New functions:

    quadrupole_field_at
    quadrupole
            
    """
    def __init__(self, *args, **kwargs):

        """
        fixed quantities: 
           _r: coordinates

           _q: charge
           _R_q: charge standard deviation

           _p0: permanent dipole
           _R_p: dipole moment standard deviation

            _Q0: permanent quadrupole

            _a0: polarizability tensor
            _b0: hyperpolarizability tensor

        variable:
            _field
            _potential
        
        derived quantities 
        
           p:  total dipole moment
           a:  effective polarizability
        
        """
#Default initialization using PointDipole initiator
        super( GaussianQuadrupole , self).__init__( **kwargs )

        self._R_q = float( kwargs.get( 'charge_std' , 0.00000001   ))
        self._R_p = float( kwargs.get( 'dipole_std' , 0.00000001  ))

#Additional attribute
        if "quadrupole" in kwargs:
            upper_triangular_quadru = array( kwargs.get( 'quadrupole' , zeros( 6, ) ))
            assert upper_triangular_quadru.shape == ( 6,) 
            self._Q0 = np.zeros(( 3, 3 ))
            for ij, (i, j) in enumerate(ut.upper_triangular(2)):
                aij = upper_triangular_quadru[ij]
                self._Q0[i, j] = aij
                self._Q0[j, i] = aij
        else:
            self._Q0 = np.zeros(( 3,3, ))

#Overriding default field_at
    def field_at(self, r):

        #print "Monopole: " , self.monopole_field_at(r)
        #print "Dipole: " , self.dipole_field_at(r)
        #print "Quadrupole: " , self.quadrupole_field_at(r) , "\n-----"

        return self.monopole_field_at(r) + self.dipole_field_at(r) +\
                self.quadrupole_field_at(r)

# New version of monopole_field_at which stems from a gaussian distribution of the source charge
# Set self._R_q = 0.000001 for classical point dipole behaivor

    def monopole_field_at(self, r):
        dr = r - self._r
        dr2 = dot(dr, dr)
        if dr2 < .1: raise Exception("Nuclei too close")

        q = self._q
        R = self._R_q

        inv_sqrt = 1/np.sqrt( np.pi )
        E = q * ( erf( dr2**0.5/R ) * dr/ dr2**1.5 -  2 * inv_sqrt * dr2**0.5 * \
                np.exp( -dr2/R**2))

        return  E
# New version of dipole_field_at, which stems from a gaussian distribution of the source dipole
# Set self._R_p = 0.000001 for classical point dipole behaivor

    def dipole_field_at( self, r ):
        dr = r - self._r
        dr2 = dot(dr, dr)

        if dr2 < .1: raise Exception("Nuclei too close")
        
        R = self._R_p
        p = self.dipole_moment()

        invpi = 1/ np.sqrt( np.pi )
        
#Two scalars

        first = erf( dr2**0.5 / R)
        second = 2 * invpi * dr2**0.5 / R * np.exp( -dr2 /R**2 )
        third = 4*invpi/R**3* outer( dr , dr ) / dr2 * np.exp( -dr2 /R**2)
        E =  (3* outer( dr, dr ) - dr2 * I_3 )/ dr2**2.5 *(first - second) - third

        return dot(p,E )

#New for GaussianQuadrupole , a point-like quadrupole induced field
    def quadrupole_field_at(self, r):
        dr = r - self._r
        dr2 = dot(dr, dr)
        if dr2 < .1: raise Exception("Nuclei too close")

        tensor = zeros( (3, 3, 3,) )
        q = self.quadrupole_moment()

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    tmp = 0
                    if j == k:
                        tmp +=  dr[i]
                    if i == k:
                        tmp +=  dr[j]
                    if i == j:
                        tmp +=  dr[k]
                    tensor[i, j, k] = (15 * dr[i] * dr[j] * dr[k] - 3*tmp*dr2 ) / (dr2 ** 3.5 )
        #print einsum("ijk,jk", tensor, q )
        return  einsum("ijk,jk", tensor, q )

#For generality
    def quadrupole_moment(self):
        return self._Q0

class SCFNotConverged(Exception):
    def __init__(self, residual, threshold):
        self.residual = residual
        self.threshold = threshold

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('potfile')
    parser.add_argument('-Rq' , type = float, default = 0.000001)
    parser.add_argument('-Rp' , type = float, default = 0.000001)
    args = parser.parse_args()

    pdl = GaussianQuadrupoleList(open(args.potfile))
    for i in pdl:
        i._R_q = args.Rq
        i._R_p = args.Rp

    print pdl.beta()
