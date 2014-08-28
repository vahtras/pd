#!/usr/bin/env python 
#-*- coding: utf-8 -*-

import numpy as np 
from math import erf
from numpy.linalg import norm 
from numpy import outer, dot, array, zeros, einsum
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

        self._R_q = float( kwargs.get( 'charge_std' , 0.5 ) )
        self._R_p = float( kwargs.get( 'dipole_std' , 0.5 ) )

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

        print self.dipole_field_at(r)

        return self.monopole_field_at(r) + self.dipole_field_at(r) +\
                self.quadrupole_field_at(r)

# New version of monopole_field_at which stems from a gaussian distribution of the source charge
    def monopole_field_at(self, r):
        dr = r - self._r
        dr2 = dot(dr, dr)
        if dr2 < .1: raise Exception("Nuclei too close")

        q = self._q
        R = self._R_q

        inv_sqrt = 1/np.sqrt( np.pi )
        E = q * ( erf( dr2**0.5/R ) * dr/ dr2**1.5 -  2 * inv_sqrt * dr2**0.5 * \
                np.exp( -dr/R**2))
        return  E
# New version of dipole_field_at, which stems from a gaussian distribution of the source dipole

    def dipole_field_at( self, r ):
        dr = r - self._r
        dr2 = dot(dr, dr)

        if dr2 < .1: raise Exception("Nuclei too close")
        
        R = self._R_p

        R = 0.5

        p = self.dipole_moment()

        invpi = 1/ np.sqrt( np.pi )
        
        first = erf( dr2**0.5 / R)
        second = 2 * dr2**0.5 * invpi / R * np.exp( -dr2 /R**2 )
        third = 4*invpi/R**3* dr * dot(dr, p ) /dr2*np.exp( -dr2**2/R**2)
        E =  (3* dr*dot( dr, p ) - dot( np.ones(3), dr2*np.diag((1,1,1))) )/ dr2**2.5 *(first - second) - third
        return E

#New for GaussianQuadrupole 
    def quadrupole_field_at(self, r):
        dr = r - self._r
        dr2 = dot(dr, dr)
        if dr2 < .1: raise Exception("Nuclei too close")

        tensor = zeros( (3, 3, 3,) )
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
        q = self.quadrupole_moment()
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
    args = parser.parse_args()

    pdl = GaussianQuadrupoleList(open(args.potfile))
    pdl.solve_scf()


