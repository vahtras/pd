#!/usr/bin/env python 
#-*- coding: utf-8 -*-

import numpy as np 
from math import erf
from numpy.linalg import norm , tensorinv
from numpy import outer, dot, array, zeros, einsum, diag
import ut
from particles import header_to_dict, line_to_dict
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
        """Overriding superclass list append: check if arg is GaussianQuadrupole"""
        if not isinstance(arg, Thole):
            print " TholeList.append called with object of type", type(arg)
            raise TypeError
        super(TholeList, self).append(arg)

    def dipole_coupling_tensor(self, a = 2.1304 ):
        """Calculates the dipole coupling, tensor using
        the exponential damping scheme by Thole.

        a is the reference adjusted parameter used in the paper
        """
        n = len(self)
        _T = zeros((n, 3, n,  3))
        invpi = 1/ np.sqrt( np.pi )

        for i in range( n ):
            ri = self[i]._r
            for j in range( n ):
                if self[i].in_group_of( self[j] ):
                    continue
                rj = self[j]._r
                _rij = ri - rj
                r = rij = np.sqrt( dot( _rij, _rij ))

                first_sum = 3 * r**-5 * np.outer( ri, rj ) * (1 - (a**3*r**3/6 + a**2 * r**2/2 + a*r + 1 ) * np.exp(-a* r)) 

                second_sum = r**-3 * (1 - (0.5*a**2*r**2 + a*r + 1) * np.exp(-a*r) )

                _Tij = first_sum 
                _T[ i, :, j, : ] = _Tij
                _T[ i, 0, j, 0 ] -= second_sum
                _T[ i, 1, j, 1 ] -= second_sum
                _T[ i, 2, j, 2 ] -= second_sum
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

if __name__ == "__main__":
    pass
