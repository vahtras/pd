import unittest
import numpy as np
from ..particles import PointDipole

class PointDipoleTest(unittest.TestCase):
    """Test basic particle properties"""

    def setUp(self):
        self.particle = PointDipole(0.,0.,0.,1.0,0.1,0.2, 0.3,0.05)
        self.e_field = np.array([1., 2., 3.])

    def test_coor(self):
        np.allclose(self.particle.r, (0., 0., 0.))

    def test_charge(self):
        self.assertEqual(self.particle.q, 1.0)

    def test_dipole(self):
        np.allclose(self.particle.p, (0.1, 0.2, 0.3))

    def test_alpha(self):
        self.assertEqual(self.particle.a[0,0], 0.05)

    def test_str(self):
        self.particle.fmt = "%5.2f"
        self.assertEqual(str(self.particle),
            " 0.00 0.00 0.00 1.00 0.10 0.20 0.30 0.05"
            )

    def test_charge_energy(self):
        self.particle.r = np.array([1., 1., 1.])
        self.assertEqual(self.particle.charge_energy(self.e_field), -6.0)

    def test_dipole_energy(self):
        reference_dipole_energy = -1.4
        self.assertEqual(self.particle.dipole_energy(self.e_field), reference_dipole_energy)

    def test_induced_dipole_energy(self):
        self.assertAlmostEqual(self.particle.induced_dipole_energy(self.e_field), -0.35)

    def test_total_field_energy(self):
        self.assertEqual(self.particle.total_field_energy(self.e_field), -1.75)
