#!/usr/bin/env python 
#-*- coding: utf-8 -*-

import numpy as np 
from math import erf
from numpy.linalg import norm , tensorinv
from numpy import outer, dot, array, zeros, einsum, diag
import ut
from particles import header_to_dict, line_to_dict, PointDipoleList
from gaussian import GaussianQuadrupoleList, GaussianQuadrupole, SCFNotConverged, header_to_dict

I_3 = np.identity(3)
ZERO_VECTOR = np.zeros(3)
ALPHA_ZERO = np.zeros((3, 3))
BETA_ZERO = np.zeros((3, 3, 3))

class TholeList( GaussianQuadrupoleList ):
    """
    A list class of ``Thole`` objects

    Overrides the functions:

    __init__
    append
    dipole_coupling_tensor

    """

    def __init__(self, pf=None):
        """Class constructor 
        pf: potential file object (or iterator)
        """

        a0 = 0.52917721092
        if pf is not None:
            units = pf.next()
            self.header_dict = header_to_dict( pf.next() )
            for i, line in enumerate(pf):
                if i == self.header_dict["#atoms"]: break
                line_dict = line_to_dict( self.header_dict, line)
                self.append( Thole(**line_dict) )
            if units == 'AA':
                for p in self:
                    p._r /= a0

    def append(self, arg):
        """Overriding superclass list append: check if arg is Thole"""
        if not isinstance(arg, Thole):
            print " TholeList.append called with object of type", type(arg)
            raise TypeError
        super(TholeList, self).append(arg)

    def evaluate_field_at_atoms(self, a = 2.1304, external=None):
        E_at_p = np.zeros( (len(self), 3))
        for i, pdi in enumerate( self ):
            for j, pdj in enumerate( self ):
                if pdj.in_group_of( pdi):
                    continue
                rij = pdi.r - pdj.r
                r = np.sqrt( norm( rij ) )
                u = r / ( pdi._a0.trace() * pdj._a0.trace() / 9 )**(1/6)
                v = a * u
                fv = 1 - (( 0.5 * v + 1) * np.exp(-v))
                fe = fv - (( 0.5 * v**2 + 0.5 * v) * np.exp(-v))
                ft = fe - (v**3 * np.exp( -v ) / 6)
                #E_at_p[ i ] += pdj.field_at( pdi.r, damp_1 = fe, damp_2 = ft )
                E_at_p[ i ] += pdj.monopole_field_at( pdi.r, damp = fe ) - pdj.dipole_field_at( pdi.r, damp_1 = fe, damp_2 = ft )

        if external is not None:
            E_at_p += external
        return E_at_p

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

    def dipole_coupling_tensor(self, a = 2.1304 ):
        n = len(self)
        _T = zeros((n, 3, n,  3))
        for i in range(n):
            ri = self[i]._r
            for j in range(i):
                if self[i].in_group_of( self[j] ):
                    continue
                rj = self[j]._r
                rij = ri - rj
                rij2 = dot(rij, rij)
                r = np.sqrt( rij2 )
# For damping
                u = r / (( self[i]._a0.trace() * self[j]._a0.trace() / 9 ) ** (1.0/6))
                v = a * u
                fv = 1 - (( 0.5 * v + 1) * np.exp(-v))
                fe = fv - (( 0.5 * v**2 + 0.5 * v) * np.exp(-v))
                ft = fe - ( v**3 * np.exp( -v ) / 6 )
# end of damping
                Tij = (3*outer(rij, rij)*ft  - fe*rij2*I_3)/rij2**2.5
                _T[i, :, j, :] = Tij
                _T[j, :, i, :] = Tij
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

class Thole( GaussianQuadrupole ):
    """ 
    Inherits GaussianQuadrupole
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
        super( Thole , self).__init__( **kwargs )
    @property
    def r(self):
        return self._r

#    def field_at(self, r, damp_1 = 1 , damp_2 = 1 ):
#        return self.monopole_field_at(r, damp = damp_1) + self.dipole_field_at(r, damp_1 = damp_1, damp_2 = damp_2)
#
    def monopole_field_at(self, r, damp = 1):
        dr = r - self._r
        dr2 = dot(dr, dr)
        if dr2 < .1: raise Exception("Nuclei too close")
        return damp*self._q*dr/dr2**1.5

    def dipole_field_at(self, r, damp_1 = 1, damp_2 = 1):
        dr = r - self._r
        dr2 = dot(dr, dr)
        if dr2 < .1: raise Exception("Nuclei too close")
        p = self.dipole_moment()
        return (3*dr*dot(dr, p)*damp_2 - damp_1*dr2*p)/dr2**2.5




if __name__ == "__main__":
    pass
