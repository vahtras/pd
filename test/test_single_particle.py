import unittest
import numpy as np
from ..particles import PointDipole, line_to_dict, header_to_dict
from util import field_gradient

from ..particles import I_3, ORIGO, BETA_ZERO

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
        self.particle._b0[0, 0, 0] = 0.01
        self.particle._b0[1, 1, 1] = 0.01
        self.particle._b0[2, 2, 2] = 0.01
        self.particle.local_field = np.array([1., 2., 3.])
        self.particle.local_potential = 0.4
        self.beta = np.zeros((3, 3, 3))
        self.beta[0, 0, 0] = self.beta[1, 1, 1] = self.beta[2, 2, 2] = 0.01
#
# test instance attributes
#

    def test_coor(self):
        np.testing.assert_allclose(self.particle.coordinates(), (0., 0., 0.))

    def test_set_coor_tuple(self):
        self.particle.set_coordinates((1, 2, 3))
        np.testing.assert_equal(self.particle.coordinates(), (1, 2, 3))

    def test_set_coor_list(self):
        self.particle.set_coordinates([1, 2, 3])
        np.testing.assert_equal(self.particle.coordinates(), (1, 2, 3))

    def test_charge(self):
        self.assertEqual(self.particle.charge(), 1.0)

    def test_set_charge(self):
        self.particle.set_charge(1.7)
        self.assertEqual(self.particle.charge(), 1.7)

    def test_permanent_dipole(self):
        np.testing.assert_allclose(self.particle.permanent_dipole_moment(), (0.1, 0.2, 0.3))

    def test_induced_dipole(self):
        np.testing.assert_allclose(self.particle.induced_dipole_moment(), (0.055, 0.12, 0.195))

    def test_alpha(self):
        self.assertEqual(self.particle._a0[0,0], 0.05)

    def test_alpha_induced(self):
        np.testing.assert_allclose(
            self.particle.da, np.diag([0.01, 0.02, 0.03])
            )

    def test_total_alpha(self):
        np.testing.assert_allclose(
            self.particle.a,  np.diag([0.06, 0.07, 0.08])
            )

#
# Instance methods 

# energy:
#
    def test_total_field_energy(self):
        dipole = PointDipole(
            charge=1.2, 
            local_potential=0.12,
            dipole=(1, 2, 3),
            iso_alpha=0.05,
            beta=self.beta
            )
            
        self.assertEqual(self.particle.total_field_energy(), -1.41)

    def test_monopole_energy(self):
        charge_in_potential = PointDipole(charge=1.2, local_potential=0.12)
        np.testing.assert_almost_equal(
            charge_in_potential.charge_energy(),
            0.144
            )

    def test_permanent_dipole_energy(self):
        dipole = PointDipole(dipole=(.1, .2, .3), local_field=(1, 2, 3))
        self.assertEqual(
            dipole.permanent_dipole_energy(), 
            -1.4
            )

    def test_alpha_induced_dipole_energy(self):
        dipole = PointDipole(iso_alpha=0.05, local_field=(1, 2, 3))
        self.assertAlmostEqual(
            dipole.alpha_induced_dipole_energy(), -0.35
            )

    def test_beta_induced_dipole_energy(self):
        dipole = PointDipole(iso_alpha=0.05, local_field=(1, 2, 3))
        self.assertAlmostEqual(
            self.particle.beta_induced_dipole_energy(), -0.06
            )

# Potential at external point

    def test_total_potential_at(self):
        charge = PointDipole(charge=1.3, dipole=(.1, .2, .3))
        field_point = np.array([0., 3., 4.])
        np.testing.assert_almost_equal(
            charge.potential_at(field_point),
            1.3/5 + 1.8/125
            )
    def test_monopole_potential_at(self):
        charge = PointDipole(charge=1.3)
        field_point = np.array([0., 3., 4.])
        np.testing.assert_almost_equal(
            charge.monopole_potential_at(field_point),
            1.3/5
            )

    def test_dipole_potential_at(self):
        dipole = PointDipole(dipole=(.1, .2, .3))
        field_point = np.array([0., 3., 4.])
        np.testing.assert_almost_equal(dipole.dipole_potential_at(field_point), 1.8/125)

# Field at external point

    def test_monopole_field_at(self):
        charge = PointDipole(charge=1.3)
        field_point = np.array([0., 3., 4.])
        np.testing.assert_almost_equal(
            charge.monopole_field_at(field_point),
            1.3*field_point/125
            )


    def test_dipole_field_at(self):
        dipole = PointDipole(dipole=(1, 1, 1))
        field_point = np.array([0., 3., 4.])
        ref = (3*field_point*7 - 25*np.ones(3))/3125
        np.testing.assert_almost_equal(
            dipole.dipole_field_at(field_point),
            ref
            )

    def test_total_field_at(self):
        field_point = np.array([0., 3., 4.])
        self.particle.local_field = np.zeros(3)
        self.particle._p0 = np.ones(3)
        ref = (3*field_point*7 - 25*np.ones(3))/5**5 +\
            1.0*field_point/5**3
        np.testing.assert_almost_equal(
            self.particle.field_at(field_point),
            ref
            )

    def test_set_local_field_raises_typeerror(self):
        def wrapper(particle, setvalue):
            particle.local_field = setvalue
        self.assertRaises(TypeError, wrapper, 0.0)

    def test_setget_local_field(self):
        reference_field = np.random.random((3))
        self.particle.local_field = reference_field
        local_field = self.particle.local_field
        np.testing.assert_equal(reference_field, local_field)

# output methods

    def test_str(self):
        self.particle.fmt = "%5.2f"
        self.assertEqual(str(self.particle),
            "1 0.00 0.00 0.00 1.00 0.10 0.20 0.30 0.05"
            )

    def test_str_with_no_dipole(self):
        self.particle.fmt = "%5.2f"
        self.particle._p0 = None
        self.assertEqual(str(self.particle),
            "1 0.00 0.00 0.00 1.00 0.05"
            )




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
        header = "2 1 1 1 1"
        header_data = map(int, header.split())
        header_dict = header_to_dict(header)
        self.assertEqual(header_dict["hyp_pol"], True)

    def test_line_to_dict_charges(self):
        header_dict = {"#atoms:2": 2, "max_angmom": 0}
        pot_line = "1 0 0 0 1.5"
        line_dict = line_to_dict(header_dict, pot_line)
        self.assertEqual(line_dict['charge'], 1.5)

    def test_line_to_dict_dipole(self):
        header_dict = {"#atoms:2": 2, "max_angmom": 1}
        pot_line = "1 0 0 0 1.5 1 2 3"
        line_dict = line_to_dict(header_dict, pot_line)
        self.assertEqual(line_dict['dipole'], [1.0, 2.0, 3.0])

    def test_line_to_dict_quadrupole(self):
        header_dict = {"#atoms:2": 2, "max_angmom": 2}
        pot_line = "1 0 0 0 1.5 1 2 3 .1 .2 .3 .4 .5 .6"
        line_dict = line_to_dict(header_dict, pot_line)
        self.assertEqual(line_dict['quadrupole'], [.1, .2, .3, .4, .5, .6])


# More constructor tests

    def test_verify_default_origin(self):
        default_atom = PointDipole()
        np.testing.assert_equal(default_atom._r, np.zeros(3))

    def test_verify_default_dipole(self):
        default_atom = PointDipole()
        np.testing.assert_equal(default_atom._p0, ORIGO)

    def test_verify_default_isopol(self):
        default_atom = PointDipole()
        np.testing.assert_equal(default_atom._a0.diagonal(), np.zeros(3))

    def test_verify_default_hyperpol(self):
        default_atom = PointDipole()
        np.testing.assert_equal(default_atom._b0, BETA_ZERO)
            

    #def test_H2_verify_default_hyperbol
