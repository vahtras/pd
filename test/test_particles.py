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
    
    def test_dipole_tensor_zero(self):
        Tij = self.pdl.dipole_tensor()
        zeromat = np.zeros((3,3))
        self.assertTrue(np.allclose(Tij[0,0,:,:], zeromat))
        self.assertTrue(np.allclose(Tij[1,1,:,:], zeromat))
        self.assertTrue(np.allclose(Tij[2,2,:,:], zeromat))
        self.assertFalse(np.allclose(Tij[0,1,:,:], zeromat))
        self.assertFalse(np.allclose(Tij[0,2,:,:], zeromat))
        self.assertFalse(np.allclose(Tij[1,2,:,:], zeromat))

    def test_dipole_tensor_values_01(self):
        x = 1.481
        y = 0
        z = 0.698 + 0.349
        r_5 = (x*x + y*y + z*z)**2.5
        T01 = self.pdl.dipole_tensor()[0, 1]
        T01xx =  (2*x*x - y*y - z*z) / r_5
        self.assertAlmostEqual(T01xx, T01[0,0])
        T01xy =  0.0
        self.assertAlmostEqual(T01xy, T01[0,1])
        T01xz =  3*x*z  / r_5
        self.assertAlmostEqual(T01xz, T01[0,2])
        T01yy = (2*y*y - x*x - z*z) / r_5
        self.assertAlmostEqual(T01yy, T01[1,1])
        T01yz =  0.0
        self.assertAlmostEqual(T01yz, T01[1,2])
        T01zz = (2*z*z - x*x - y*y) / r_5
        self.assertAlmostEqual(T01zz, T01[2,2])

    def test_dipole_tensor_values_02(self):
        x = -1.481
        y = 0
        z = 0.698 + 0.349
        r_5 = (x*x + y*y + z*z)**2.5
        T02 = self.pdl.dipole_tensor()[0, 2]
        T02xx =  (2*x*x - y*y - z*z) / r_5
        self.assertAlmostEqual(T02xx, T02[0,0])
        T02xy =  0.0
        self.assertAlmostEqual(T02xy, T02[0,1])
        T02xz =  3*x*z  / r_5
        self.assertAlmostEqual(T02xz, T02[0,2])
        T02yy = (2*y*y - x*x - z*z) / r_5
        self.assertAlmostEqual(T02yy, T02[1,1])
        T02yz =  0.0
        self.assertAlmostEqual(T02yz, T02[1,2])
        T02zz = (2*z*z - x*x - y*y) / r_5
        self.assertAlmostEqual(T02zz, T02[2,2])

    def test_dipole_tensor_values_12(self):
        x = - 2.962
        y = 0
        z = 0
        r_5 = (x*x + y*y + z*z)**2.5
        T12 = self.pdl.dipole_tensor()[1, 2]
        T12xx =  (2*x*x - y*y - z*z) / r_5
        self.assertAlmostEqual(T12xx, T12[0,0])
        T12xy =  0.0
        self.assertAlmostEqual(T12xy, T12[0,1])
        T12xz =  3*x*z  / r_5
        self.assertAlmostEqual(T12xz, T12[0,2])
        T12yy = (2*y*y - x*x - z*z) / r_5
        self.assertAlmostEqual(T12yy, T12[1,1])
        T12yz =  0.0
        self.assertAlmostEqual(T12yz, T12[1,2])
        T12zz = (2*z*z - x*x - y*y) / r_5
        self.assertAlmostEqual(T12zz, T12[2,2])

                        
        
if __name__ == "__main__":
    unittest.main()
