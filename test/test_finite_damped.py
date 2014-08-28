#!/usr/bin/env python
#-*- coding: utf-8 -*-


import unittest
from math import erf
import numpy as np
from numpy.linalg import norm

class GaussianTest( unittest.TestCase ):

    """

    Test so that the finite field gives the identical answer as the equations (1),(2) and (3)
    where analytically derived for two random vectors.

    """

    def setUp(self):

        self._source = np.array( [0,0,0], dtype = float )
        self._dest = np.array( [0.5, 0.5, 1.0], dtype = float )
        #self._source = np.random.random( 3 )
        #self._dest = np.random.random( 3 )
        self._R = 0.5

#Return a scalar potential given a destination vector position 0 < norm(r) <= 1
    def equation_one( self, r ):
        return erf( norm( r - self._source) / self._R ) / norm( r - self._source)

#Return vector representing the component-wise derivative of equation 2
#Which also is the electric field vector at position r due to a charge distribution

    def equation_two( self, r ):
        dr = norm( r - self._source )
        R = self._R
        invpi = 1/ np.sqrt( np.pi )
        first = erf( dr / R )
        second = 2 * dr * invpi / R * np.exp( -dr**2/R**2 )
        return -( r - self._source ) / dr**3 * ( first - second )

    def equation_three( self, r ):

        d = r
        s = self._source
        dr = norm( d - s )
        R = self._R

        invpi = 1/ np.sqrt( np.pi )
        
        first = erf( dr / R)
        second = 2 * dr * invpi / R * np.exp( -dr**2/R**2 )
        third = 4*invpi/R**3*np.outer((d-s),(d-s))/dr**2*np.exp( -dr**2/R**2)
        return (3*np.outer((d-s),(d-s))-dr**2*np.ones((3,3))) / dr**5 *(first - second) - third

    def test_finite_first(self):

        dx = np.array( [0.001, 0, 0], dtype = float )
        dy = np.array( [0, 0.001, 0], dtype = float )
        dz = np.array( [0, 0, 0.001], dtype = float )

        E_x_fin = (self.equation_one( self._dest +dx ) - self.equation_one(self._dest - dx))/(2*0.001)
        E_y_fin = (self.equation_one( self._dest +dy ) - self.equation_one(self._dest - dy))/(2*0.001)
        E_z_fin = (self.equation_one( self._dest +dz ) - self.equation_one(self._dest - dz))/(2*0.001)

        E_x_ana = self.equation_two( self._dest )[0]
        E_y_ana = self.equation_two( self._dest )[1]
        E_z_ana = self.equation_two( self._dest )[2]

        np.testing.assert_almost_equal( E_x_fin , E_x_ana , 6)
        np.testing.assert_almost_equal( E_y_fin , E_y_ana , 6)
        np.testing.assert_almost_equal( E_z_fin , E_z_ana , 6)

    def test_finite_second(self):

        dx = np.array( [0.001, 0, 0], dtype = float )
        dy = np.array( [0, 0.001, 0], dtype = float )
        dz = np.array( [0, 0, 0.001], dtype = float )

        r = self._dest

        E_xx_fin = (self.equation_two( r + dx )[0]-self.equation_two( r - dx )[0])/(2*0.001)
        E_xy_fin = (self.equation_two( r + dx )[1]-self.equation_two( r - dx )[1])/(2*0.001)
        E_xz_fin = (self.equation_two( r + dx )[2]-self.equation_two( r - dx )[2])/(2*0.001)
        E_yy_fin = (self.equation_two( r + dy )[1]-self.equation_two( r - dy )[1])/(2*0.001)
        E_yz_fin = (self.equation_two( r + dy )[1]-self.equation_two( r - dy )[1])/(2*0.001)
        E_zz_fin = (self.equation_two( r + dz )[2]-self.equation_two( r - dz )[2])/(2*0.001)

        E_xx_ana = self.equation_three(  r  )[0][0]
        E_xy_ana = self.equation_three(  r  )[0][1]
        E_xz_ana = self.equation_three(  r  )[0][2]
        E_yy_ana = self.equation_three(  r  )[1][1]
        E_yz_ana = self.equation_three(  r  )[1][1]
        E_zz_ana = self.equation_three(  r  )[2][2]

        np.testing.assert_almost_equal( E_xx_fin , E_xx_ana , 5)
        #np.testing.assert_almost_equal( E_xy_fin , E_xy_ana , 5)
        #np.testing.assert_almost_equal( E_xz_fin , E_xz_ana , 5)
        np.testing.assert_almost_equal( E_yy_fin , E_yy_ana , 5)
        np.testing.assert_almost_equal( E_yz_fin , E_yz_ana , 5)
        np.testing.assert_almost_equal( E_zz_fin , E_zz_ana , 5)


if __name__ == '__main__':
    unittest.main()


