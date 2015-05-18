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
                if pdi.in_group_of( pdj):
                    continue
                rij = pdi.r - pdj.r
                r = np.sqrt( norm( rij ))
                u = r * ( pdi._a0.trace() * pdj._a0.trace() / 9 )**(-1/6)
                v = a * u
                fv = 1 - ( 0.5 * v + 1) * np.exp(-v)
                fe = fv - ( 0.5 * v**2 + 0.5 * v) * np.exp(-v)
                ft = fe - v**3 * np.exp( -v ) / 6
                print pdj.field_at( pdi.r )
                E_at_p[ i ] += -fe * pdj.monopole_field_at( pdi.r ) + ft*pdj.dipole_field_at( pdi.r )
                #E_at_p[ i ] += pdj.field_at( pdi.r )

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
        """Calculates the dipole coupling, tensor using
        the exponential damping scheme by Thole.

        a is the reference adjusted parameter used in the paper


        -------------------
        REFERENCE:

        Molecular Simulation
        , Vol. 32, No. 6, 15 May 2006, 471–48
        Appendix A.
        -------------------


        """
        n = len(self)
        _T = zeros((n, 3, n,  3))
        for i, pdi in enumerate( self ):
            ri = self[i]._r
            for j, pdj in enumerate( self ):
                if self[i].in_group_of( self[j] ):
                    continue

# For constants.
                rj = self[j]._r
                _rij = ri - rj
                r = rij = np.sqrt( dot( _rij, _rij ))
                u = r * ( pdi._a0.trace() * pdj._a0.trace() / 9 )**(-1/6)
                v = a * u
# For the dyadic tensor
                fv = 1 - ( 0.5 * v + 1) * np.exp(-v)
                fe = fv - ( 0.5 * v**2 + 0.5 * v) * np.exp(-v)
                ft = fe - v**3 * np.exp( -v ) / 6


                #first_sum = 3 * r**-5 * np.outer( ri, rj ) * (1 - (a**3*r**3/6 + a**2 * r**2/2 + a*r + 1 ) * np.exp(-a* r)) 

                #second_sum = r**-3 * (1 - (0.5*a**2*r**2 + a*r + 1) * np.exp(-a*r) )

                _Tij = (3 * np.einsum('i,j->ij', pdi.r, pdj.r ) * ft - I_3 * r**2*fe)/r**5
                _T[ i, :, j, : ] = _Tij
                _T[ j, :, i, : ] = _Tij
        return _T


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


if __name__ == "__main__":
    pass
