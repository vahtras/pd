#!/usr/bin/env python 
#-*- coding: utf-8 -*-

import numpy as np 
from numpy.linalg import norm 
from numpy import outer, dot, array, zeros, einsum
import ut
from particles import PointDipole, PointDipoleList, header_to_dict, line_to_dict

I_3 = np.identity(3)
ZERO_VECTOR = np.zeros(3)
ZERO_TENSOR = np.zeros((3, 3))
ALPHA_ZERO = np.zeros((3, 3))
BETA_ZERO = np.zeros((3, 3, 3))

class QuadrupoleList( PointDipoleList ):
    """
    A list class of ``Quadrupole`` objects

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
                self.append( Quadrupole(**line_dict) )

    @staticmethod
    def from_string(potential):
        """Used to build the ``QuadrupoleList`` object when the
           potential file is given as a triple quoted string with newlines
        """
        return QuadrupoleList(iter(potential.split("\n")))

    def append(self, arg):
        """Overriding superclass list append: check if arg is Quadrupole"""
        if not isinstance(arg,  Quadrupole):
            print "QuadrupoleList.append called with object of type", type(arg)
            raise TypeError
        super(QuadrupoleList, self).append(arg)

class Quadrupole( PointDipole ):
    """ 

    Inherits PointDipole with new attributes:

    _Q0 : Permanent quadrupole moment

    Overrides the functions:
    
    field_at

    New functions:

    quadrupole_field_at
    quadrupole
            
    """
    def __init__(self, *args, **kwargs):

        """
        fixed quantities: 
           _r: coordinates
           _q: charge
           _p0: permanent dipole
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
        super( Quadrupole , self).__init__( **kwargs )

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
            self._Q0 = ZERO_TENSOR

#Overriding default field_at
    def field_at(self, r):
        return self.monopole_field_at(r) + self.dipole_field_at(r) +\
                self.quadrupole_field_at(r)

#New for Quadrupole 
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

    pdl = QuadrupoleList(open(args.potfile))
    pdl.solve_scf()


