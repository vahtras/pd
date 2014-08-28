import unittest
import numpy as np
from ..particles import PointDipole, line_to_dict, header_to_dict
from util import field_gradient

from ..constants import I_3, ZERO_VECTOR, ZERO_TENSOR, ZERO_RANK_3_TENSOR

class PointDipoleTest(unittest.TestCase):
    """Test basic particle properties"""

    def setUp(self):
        self.coordinates = np.random.random(3)
        self.particle = PointDipole(
            coordinates=self.coordinates,
            charge=1.0,
            dipole=[0.1, 0.2, 0.3],
            iso_alpha=0.05,
            beta=np.zeros((3, 3, 3))
            )
        self.particle._b0[0, 0, 0] = 0.01
        self.particle._b0[1, 1, 1] = 0.01
        self.particle._b0[2, 2, 2] = 0.01
        self.particle.set_local_field([1., 2., 3.])
        self.particle.set_local_potential(0.4)
        self.beta = np.zeros((3, 3, 3))
        self.beta[0, 0, 0] = self.beta[1, 1, 1] = self.beta[2, 2, 2] = 0.01
#
# test instance attributes
#

    def test_coor(self):
        np.testing.assert_allclose(
            self.particle.coordinates(), self.coordinates
            )

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
            self.particle.induced_polarizability(), np.diag([0.01, 0.02, 0.03])
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
            
        self.assertEqual(self.particle.total_energy(), -1.41)

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

    def test_permanent_dipole_field_at(self):
        self.particle.set_local_field((0,0,0)) # no induced dipole
        self.particle._p0 = np.ones(3)         # only permanent dipole
        dR = np.array([0., 3., 4.])
        field_point = self.coordinates + dR
        reference_field = (3*dR*7 - 25*np.ones(3))/5**5 + 1.0*dR/5**3
        np.testing.assert_almost_equal(
            self.particle.field_at(field_point),
            reference_field
            )

    def test_set_field_raises_typeerror(self):
        self.assertRaises(ValueError, self.particle.set_local_field, (0,))

    def test_setget__field(self):
        reference_field = np.random.random((3))
        self.particle.set_local_field(reference_field)
        _field = self.particle.local_field()
        np.testing.assert_equal(reference_field, _field)

# output methods

    def test_str(self):
        self.particle.fmt = "%5.2f"
        self.assertEqual(str(self.particle),
            "1%5.2f%5.2f%5.2f 1.00 0.10 0.20 0.30 0.05" % tuple(self.coordinates)
            )



# More constructor tests

    def test_verify_default_origin(self):
        default_atom = PointDipole()
        np.testing.assert_equal(default_atom._r, np.zeros(3))

    def test_verify_default_dipole(self):
        default_atom = PointDipole()
        np.testing.assert_equal(default_atom._p0, ZERO_VECTOR)

    def test_verify_default_isopol(self):
        default_atom = PointDipole()
        np.testing.assert_equal(default_atom._a0.diagonal(), np.zeros(3))

    def test_verify_default_hyperpol(self):
        default_atom = PointDipole()
        np.testing.assert_equal(default_atom._b0, ZERO_RANK_3_TENSOR)
            

    #def test_H2_verify_default_hyperbol

if __name__ == "__main__":
    unittest.main()
