import unittest
import numpy as np
from ..particles import PointDipole, PointDipoleList

class PointDipoleTest(unittest.TestCase):
    """Test basic particle properties"""

    def setUp(self):
        self.particle = PointDipole(0.,0.,0.,1.0,0.1,0.2, 0.3,0.05)

    def test_coor(self):
        np.allclose(self.particle.r, (0., 0., 0.))

    def test_charge(self):
        self.assertEqual(self.particle.q, 1.0)

    def test_dipole(self):
        np.allclose(self.particle.p, (0.1, 0.2, 0.3))

    def test_alpha(self):
        self.assertEqual(self.particle.a, 0.05)

    def test_str(self):
        self.particle.fmt = "%5.2f"
        self.assertEqual(str(self.particle),
            " 0.00 0.00 0.00 1.00 0.10 0.20 0.30 0.05"
            )
        
class PointDipoleListTest(unittest.TestCase):
    def setUp(self):
        with open('/tmp/pdltest.pot', 'w') as pf:
            pf.write("""AU
3 1 0 1
1  0.000  0.000  0.698 -0.703 -0.000 0.000 -0.284 4.230
1 -1.481  0.000 -0.349  0.352  0.153 0.000  0.127 1.089
1  1.481  0.000 -0.349  0.352 -0.153 0.000  0.127 1.089
""")
        self.pdl = PointDipoleList('/tmp/pdltest.pot')

    def test_number(self):
        self.assertEqual(len(self.pdl), 3)

    def test_charge(self):
        #round-off error in this example
        self.assertAlmostEqual(self.pdl.charge(), .001) 

    def test_alpha(self):
        self.assertAlmostEqual(self.pdl.alpha(), 4.23+2*1.089)
        
if __name__ == "__main__":
    unittest.main()
