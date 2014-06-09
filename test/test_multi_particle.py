import unittest
import numpy as np
from ..particles import PointDipole, PointDipoleList
from .test_particles import H2
from .util import iterize

class MultiDipoleTest(unittest.TestCase):

    def setUp(self):
        self.He2 = PointDipoleList(iter("""AA
2 2 0 1
1 0 0  0 0 6.429866513469e-01
1 0 0 52 0 6.429866513469e-01
""".split("\n")))
        pass


        self.h2o = PointDipoleList(iterize(
"""AU
3 1 0 1
1  0.000  0.000  0.698 -0.703 -0.000 0.000 -0.284 4.230
1 -1.481  0.000 -0.349  0.352  0.153 0.000  0.127 1.089
1  1.481  0.000 -0.349  0.352 -0.153 0.000  0.127 1.089
2  0.000  0.000  0.000  0.000  0.000 0.000  0.000 0.000"""
        ))

    def test_he2(self):
        alpha_he2 = self.He2.alpha()
        ref_alpha = np.diag(
            [1.285972432811e+00, 1.285972432811e+00,1.285975043401e+00]
            )
        np.allclose(alpha_he2, ref_alpha)

    def test_h2o_number(self):
        self.assertEqual(len(self.h2o), 3)

    def test_h2o_charge(self):
        #round-off error in this example
        self.assertAlmostEqual(self.h2o.charge(), 0., places=2) 

    def test_h2o_dipole_tensor_zero(self):
        Tij = self.h2o.dipole_tensor()
        zeromat = np.zeros((3,3))
        self.assertTrue (np.allclose(Tij[0, :, 0, :], zeromat))
        self.assertTrue (np.allclose(Tij[1, :, 1, :], zeromat))
        self.assertTrue (np.allclose(Tij[2, :, 2, :], zeromat))
        self.assertFalse(np.allclose(Tij[0, :, 1, :], zeromat))
        self.assertFalse(np.allclose(Tij[0, :, 2, :], zeromat))
        self.assertFalse(np.allclose(Tij[1, :, 2, :], zeromat))

    def test_dipole_tensor_values_01(self):
        x = 1.481
        y = 0
        z = 0.698 + 0.349
        r_5 = (x*x + y*y + z*z)**2.5
        T01 = self.h2o.dipole_tensor()[0, :, 1, :]
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
        T02 = self.h2o.dipole_tensor()[0, :, 2, :]
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
        T12 = self.h2o.dipole_tensor()[1, :, 2, :]
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

    def test_verify_solver(self):
        h2 = PointDipoleList(iterize(H2["POTFILE"]))
        pass

    def test_static_charge_energy(self):
        charges = PointDipoleList.from_string("""AU
2 0 0
1 0 0 0 1.0
2 0 0 1 1.0
"""
)
        E = charges.static_charge_energy()
        self.assertEqual(E, 1.)

    def test_static_dipole_energy(self):
        dipoles = PointDipoleList.from_string("""AU
2 1 0
1 0 0 0 0.0 .0 .0 .3
2 0 0 1 0.0 .0 .0 .3
"""
)
        E_ref = - (3*0.3**2 - 1*0.3**2)
        E = dipoles.static_dipole_energy()
        self.assertAlmostEqual(E, E_ref)

