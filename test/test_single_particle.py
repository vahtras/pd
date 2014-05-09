import unittest
import numpy as np
from ..particles import PointDipole

EPSILON = 0.001

def grad(f):
    return np.array([gradx(f), grady(f), gradz(f)])

def gradx(f):
    ex = np.array([EPSILON/2, 0, 0])
    f.__self__.local_field += ex
    f1 = f()
    f.__self__.local_field -= 2*ex
    f2 = f()
    f.__self__.local_field += ex
    return (f1 - f2)/EPSILON

def grady(f):
    ey = np.array([0, EPSILON/2, 0])
    f.__self__.local_field += ey
    f1 = f()
    f.__self__.local_field -= 2*ey
    f2 = f()
    f.__self__.local_field += ey
    return (f1 - f2)/EPSILON

def gradz(f):
    ez = np.array([0, 0, EPSILON/2])
    f.__self__.local_field += ez
    f1 = f()
    f.__self__.local_field -= 2*ez
    f2 = f()
    f.__self__.local_field += ez
    return (f1 - f2)/EPSILON


class PointDipoleTest(unittest.TestCase):
    """Test basic particle properties"""

    def setUp(self):
        self.particle = PointDipole(0.,0.,0.,1.0,0.1,0.2, 0.3,0.05)
        self.particle.b[0, 0, 0] = 0.01
        self.particle.b[1, 1, 1] = 0.01
        self.particle.b[2, 2, 2] = 0.01
        self.particle.local_field = np.array([1., 2., 3.])

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
        self.assertEqual(self.particle.charge_energy(), -6.0)

    def test_permanent_dipole_energy(self):
        reference_dipole_energy = -1.4
        self.assertEqual(
            self.particle.permanent_dipole_energy(), 
            reference_dipole_energy
            )

    def test_alpha_induced_dipole_energy(self):
        self.assertAlmostEqual(
            self.particle.alpha_induced_dipole_energy(), -0.35
            )

    def test_beta_induced_dipole_energy(self):
        self.assertAlmostEqual(
            self.particle.beta_induced_dipole_energy(), -0.06
            )

    def test_total_field_energy(self):
        self.assertEqual(self.particle.total_field_energy(), -1.81)

    def test_finite_difference_energy(self):

        gradE = grad(self.particle.total_field_energy)
        dipole = self.particle.p + \
           self.particle.dipole_induced()

        np.testing.assert_almost_equal(-gradE, dipole)

    def test_set_local_field_raises_typeerror(self):
        def wrapper(particle, setvalue):
            particle.local_field = setvalue
        self.assertRaises(TypeError, wrapper, 0.0)

    def test_setget_local_field(self):
        reference_field = np.random.random((3))
        self.particle.local_field = reference_field
        local_field = self.particle.local_field
        np.testing.assert_equal(reference_field, local_field)
