import unittest
import numpy as np
from ..particles import PointDipole

EPSILON = 0.001

def grad(f, r):
    return np.array([gradx(f, r), grady(f, r), gradz(f, r)])

def gradx(f, r):
    ex = np.array([EPSILON/2, 0, 0])
    return (f(r + ex) - f(r - ex))/EPSILON

def grady(f, r):
    ey = np.array([0, EPSILON/2, 0])
    return (f(r + ey) - f(r - ey))/EPSILON

def gradz(f, r):
    ez = np.array([0, 0, EPSILON/2])
    return (f(r + ez) - f(r - ez))/EPSILON


class PointDipoleTest(unittest.TestCase):
    """Test basic particle properties"""

    def setUp(self):
        self.particle = PointDipole(0.,0.,0.,1.0,0.1,0.2, 0.3,0.05)
        self.particle.b[0, 0, 0] = 0.01
        self.particle.b[1, 1, 1] = 0.01
        self.particle.b[2, 2, 2] = 0.01
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

    def test_permanent_dipole_energy(self):
        reference_dipole_energy = -1.4
        self.assertEqual(
            self.particle.permanent_dipole_energy(self.e_field), 
            reference_dipole_energy
            )

    def test_alpha_induced_dipole_energy(self):
        self.assertAlmostEqual(
            self.particle.alpha_induced_dipole_energy(self.e_field), -0.35
            )

    def test_beta_induced_dipole_energy(self):
        self.assertAlmostEqual(
            self.particle.beta_induced_dipole_energy(self.e_field), -0.06
            )

    def test_total_field_energy(self):
        self.assertEqual(self.particle.total_field_energy(self.e_field), -1.81)

    def test_finite_difference_energy(self):

        gradE = grad(self.particle.total_field_energy, self.e_field)
        dipole = self.particle.p + \
           self.particle.dipole_induced(self.e_field)

        np.testing.assert_almost_equal(-gradE, dipole)
