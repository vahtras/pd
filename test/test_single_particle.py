import unittest
import numpy as np
from ..particles import PointDipole, line_to_dict, header_to_dict

EPSILON = 0.001
ex = np.array([EPSILON/2, 0, 0])
ey = np.array([0, EPSILON/2, 0])
ez = np.array([0, 0, EPSILON/2])

def grad(f):
    return np.array([gradx(f, ex), gradx(f, ey), gradx(f, ez)])

def gradx(f, eps):
    f.__self__.local_field += eps
    f1 = f()
    f.__self__.local_field -= 2*eps
    f2 = f()
    f.__self__.local_field += eps
    return (f1 - f2)/EPSILON


class PointDipoleTest(unittest.TestCase):
    """Test basic particle properties"""

    def setUp(self):
        self.particle = PointDipole(
            coordinates=[0.,0.,0.],
            charge=1.0,
            dipole=[0.1, 0.2, 0.3],
            iso_alpha=0.05,
            beta=np.zeros((3, 3, 3))
            )
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

    def test_header_number_of_atoms(self):
        header = "2 0 0"
        header_data = map(int, header.split())
        header_dict = header_to_dict(header)
        self.assertEqual(header_dict["#atoms"], 2)

    def test_header_max_ang_mom(self):
        header = "2 1 0"
        header_data = map(int, header.split())
        header_dict = header_to_dict(header)
        self.assertEqual(header_dict["max_angmom"], 1)

    def test_header_nopol_isotropic_polarizability_false(self):
        header = "2 1"
        header_data = map(int, header.split())
        header_dict = header_to_dict(header)
        self.assertEqual(header_dict["iso_pol"], False)

    def test_header_0_isotropic_polarizability_false(self):
        header = "2 1 0"
        header_data = map(int, header.split())
        header_dict = header_to_dict(header)
        self.assertEqual(header_dict["iso_pol"], False)

    def test_header_1_isotropic_polarizability_1_true(self):
        header = "2 1 1"
        header_data = map(int, header.split())
        header_dict = header_to_dict(header)
        self.assertEqual(header_dict["iso_pol"], True)

    def test_header_2_isotropic_polarizability_1_false(self):
        header = "2 1 2"
        header_data = map(int, header.split())
        header_dict = header_to_dict(header)
        self.assertEqual(header_dict["iso_pol"], False)

    def test_header_nopol_full_polarizability_false(self):
        header = "2 1"
        header_data = map(int, header.split())
        header_dict = header_to_dict(header)
        self.assertEqual(header_dict["full_pol"], False)

    def test_header_0_full_polarizability_false(self):
        header = "2 1 0"
        header_data = map(int, header.split())
        header_dict = header_to_dict(header)
        self.assertEqual(header_dict["full_pol"], False)

    def test_header_1_full_polarizability_1_false(self):
        header = "2 1 1"
        header_data = map(int, header.split())
        header_dict = header_to_dict(header)
        self.assertEqual(header_dict["full_pol"], False)

    def test_header_2_full_polarizability_1_true(self):
        header = "2 1 2"
        header_data = map(int, header.split())
        header_dict = header_to_dict(header)
        self.assertEqual(header_dict["full_pol"], True)


    def test_header_0_hyperpolarizability_false(self):
        header = "2 1 1 0"
        header_data = map(int, header.split())
        header_dict = header_to_dict(header)
        self.assertEqual(header_dict["hyp_pol"], False)

    def test_header_1_hyperpolarizability_true(self):
        header = "2 1 1 1"
        header_data = map(int, header.split())
        header_dict = header_to_dict(header)
        self.assertEqual(header_dict["hyp_pol"], True)

    def test_line_to_dict_charges(self):
        header_dict = {"#atoms:2": 2, "max_angmom": 0}
        pot_line = "0 0 0 1.5"
        line_dict = line_to_dict(header_dict, pot_line)
        self.assertEqual(line_dict['charge'], 1.5)

    def test_line_to_dict_dipole(self):
        header_dict = {"#atoms:2": 2, "max_angmom": 1}
        pot_line = "0 0 0 1.5 1 2 3"
        line_dict = line_to_dict(header_dict, pot_line)
        self.assertEqual(line_dict['dipole'], [1.0, 2.0, 3.0])

    def test_line_to_dict_quadrupole(self):
        header_dict = {"#atoms:2": 2, "max_angmom": 2}
        pot_line = "0 0 0 1.5 1 2 3 .1 .2 .3 .4 .5 .6"
        line_dict = line_to_dict(header_dict, pot_line)
        self.assertEqual(line_dict['quadrupole'], [.1, .2, .3, .4, .5, .6])
        
