import unittest, os
import numpy as np
import particles 
import cell

#Two water molecules, one in origo, one at r = [15, 15, 15]
POTSTRING = """AU
6 1 22 1
1      0.000000   0.000000   0.000000 -0.66229 0.00000 0.00000 0.34276 4.10574 0.00000 0.00000 4.79229 0.00000 4.01912 0.00000 0.00000 -3.33162 0.00000 0.00000 0.00000 0.00000 -0.32216 0.00000 0.79137
1      1.430429   0.000000   1.107157 0.33114 -0.16617 0.00000 -0.11629 1.53802 0.00000 1.19765 0.90661 0.00000 1.37138 -4.52137 0.00000 -5.08061 -1.35494 0.00000 -4.83365 0.00000 -0.46317 0.00000 -3.47921
1     -1.430429   0.000000   1.107157 0.33114 0.16617 0.00000 -0.11629 1.53802 0.00000 -1.19765 0.90661 0.00000 1.37138 4.52137 0.00000 -5.08061 1.35494 0.00000 4.83365 0.00000 -0.46317 0.00000 -3.47921
2     15.000000  15.000000  15.000000 -0.66229 0.00000 0.00000 0.34276 4.10574 0.00000 0.00000 4.79229 0.00000 4.01912 0.00000 0.00000 -3.33162 0.00000 0.00000 0.00000 0.00000 -0.32216 0.00000 0.79137
2     16.430429  15.000000  16.107157 0.33114 -0.16617 0.00000 -0.11629 1.53802 0.00000 1.19765 0.90661 0.00000 1.37138 -4.52137 0.00000 -5.08061 -1.35494 0.00000 -4.83365 0.00000 -0.46317 0.00000 -3.47921
2     13.569571  15.000000  16.107157 0.33114 0.16617 0.00000 -0.11629 1.53802 0.00000 -1.19765 0.90661 0.00000 1.37138 4.52137 0.00000 -5.08061 1.35494 0.00000 4.83365 0.00000 -0.46317 0.00000 -3.47921"""

class CellTest( unittest.TestCase ):
    def setUp(self):
        pass
        
    def test_init(self):
        c = cell.Cell( my_min = map(float, [0, 0, 0]),
                    my_max = map(float, [1, 1, 1] ),
                    my_cutoff = 0.4)
        assert len(c) == 3

        c = cell.Cell( my_min = map(float, [-10, 0, 0]),
                    my_max = map(float, [0, 1, 1] ),
                    my_cutoff = 12)
        assert len(c) == 1

        c = cell.Cell( my_min = map(float, [-5, 0, 0]),
                    my_max = map(float, [10, 1, 1] ),
                    my_cutoff = 4.9)
        assert len(c) == 4

    def test_from_PointDipoleList(self, ):
        _str = POTSTRING
        pdl = particles.PointDipoleList.from_string( _str )
        ce = cell.Cell.from_PointDipoleList( pdl, co = 5 )

        assert ce.shape == (2, 2, 2)
        assert isinstance( ce, cell.Cell )

        pdl = particles.PointDipoleList.cell_from_string( _str )
        assert isinstance( pdl, particles.PointDipoleList )
        assert isinstance( pdl._Cell, cell.Cell )


if __name__ == '__main__':
    unittest.main()
