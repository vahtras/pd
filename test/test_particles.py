import unittest
import numpy as np
from numpy.linalg import norm
from ..particles import PointDipole, PointDipoleList
from .util import iterize, random_scalar, random_vector
from ..constants import *
from test_applequist import H2, ANGSTROM, ANGSTROM3

# Other model systems

DIMER_TEMPLATE = """AU
2 0 0 0
1 0.00000  0.00000  0.00000 0
2 0.00000  0.00000  1.00000 0
"""

H2O_DIMER = """AU
2 1 1 0
1 0.00000  0.00000  0.48861 0.0 0.00000 -0.00000 -0.76539  6.61822
2 0.00000  0.00000  5.48861 0.0 0.00000 -0.00000 -0.76539  6.61822 
"""

H2O_DIMER_UNIT = """AU
2 1 22 1
1 0 0 -0.2249058930  0   0 0 0.814576406    7.21103625278 0 0 3.03446384360 0 5.22710462524    0 0 -18.48299798 0 0 0 0 -2.33649400 0 -11.17349291
1 0 0  0.7750941070  0   0 0 0.814576406    7.21103625278 0 0 3.03446384360 0 5.22710462524    0 0 -18.48299798 0 0 0 0 -2.33649400 0 -11.17349291
"""
        
class PointDipoleListTest(unittest.TestCase):

    def setUp(self):
        self.h2 = PointDipoleList.from_string(H2["POTFILE"])
        self.h2o_dimer = PointDipoleList.from_string(H2O_DIMER)
        self.dimer_template = PointDipoleList.from_string(DIMER_TEMPLATE)
        pass

    def test_h2o_dimer_finite_field_p(self):

        alphas = self.h2o_dimer.solve_Applequist_equation()

        self.h2o_dimer.solve_scf_for_external([0, 0, .005])
        p1 = self.h2o_dimer[0].induced_dipole_moment()
        self.h2o_dimer.solve_scf_for_external([0, 0, -.005])
        p2 = self.h2o_dimer[0].induced_dipole_moment()
        dPdE = (p1 - p2)/0.01

        np.testing.assert_allclose(alphas[0][:,2], dPdE, rtol=.001)

#
# Some refactoring tests
#
    def test_form_Applequist_rhs(self):
        h2_rhs = self.h2.form_Applequist_rhs()
        h2_rhs_ref = np.array([
            [0.168, 0, 0], 
            [0., 0.168, 0.], 
            [0., 0., 0.168],
            [0.168, 0, 0], 
            [0., 0.168, 0.], 
            [0., 0., 0.168]
            ])
        h2_rhs_ref *= ANGSTROM3
        np.testing.assert_allclose( h2_rhs, h2_rhs_ref, atol=1e-7 )

    def test_form_Applequist_coefficient_matrix(self):
        L_h2_ref = np.array([
            [1., 0., 0., 0.41240819, 0., 0.],
            [0., 1., 0., 0., 0.41240819, 0.],
            [0., 0., 1., 0., 0.,-0.82481638],
            [0.41240819, 0., 0., 1., 0., 0.],
            [0., 0.41240819, 0., 0., 1., 0.],
            [0., 0.,-0.82481638, 0., 0., 1.]
            ])
        L_h2 = self.h2.form_Applequist_coefficient_matrix()
        np.testing.assert_array_almost_equal(L_h2, L_h2_ref)

    def test_solve_Applequist_equation(self):
        alphas = self.h2.solve_Applequist_equation()
        alphas_ref =  np.array([[
            [0.11894578, 0., 0.],
            [0., 0.11894578,  0.],
            [0., 0., 0.95899377]
            ], [
            [0.11894578, 0., 0.],
            [0., 0.11894578,  0.],
            [0., 0., 0.95899377]
            ]])
        alphas_ref *= ANGSTROM3
        np.testing.assert_almost_equal(alphas, alphas_ref, 5)

    def test_induced_parallel_dipole_on_one_atom(self):
        E_external = np.array([0., 0., 1.,])
        p_ind_ref =  np.array([0., 0., 6.47160434])
        self.h2.solve_scf_for_external(E_external, max_it = 100)

        np.testing.assert_almost_equal(self.h2[0].dipole_moment(), p_ind_ref, decimal=4)

    def test_induced_orthogonal_dipole_on_one_atom(self):
        E_external = np.array([1., 0., 0.,])
        p_ind_ref =  np.array([0.11894578, 0., 0.])
        self.h2.solve_scf_for_external(E_external, max_it = 100)
        p_ind_ref *= ANGSTROM3
        np.testing.assert_almost_equal(self.h2[0].dipole_moment(), p_ind_ref, decimal=6)
        
    def test_set_groups(self):
        self.h2.set_groups((1, 2))
        self.assertEqual(self.h2.groups(), (1, 2))
        
    def test_evaluate_charge_potential_at_atoms(self):
        self.h2.set_charges((1.0, 1.0))
        self.h2.set_groups((1, 2))
        V_local = self.h2.evaluate_potential_at_atoms()
        np.testing.assert_almost_equal(V_local, [1.0*ANGSTROM/H2['R'], 1.0*ANGSTROM/H2['R']])

    def test_evaluate_dipole_potential_at_atoms(self):
        V_external = random_vector()
        p0 = random_vector()
        self.h2.set_dipoles([p0, p0])
        self.h2.set_groups((1, 2))
        V_local = self.h2.evaluate_potential_at_atoms(V_external)
        r12 = self.h2[0]._r - self.h2[1]._r
        V_ref = np.dot(p0, r12)/norm(r12)**3
        np.testing.assert_almost_equal(V_local, [V_external+V_ref, V_external-V_ref])

    def test_evaluate_field_at_atoms(self):
        E_external = np.array([0., 0., 1.,])
        E_local = self.h2.evaluate_field_at_atoms()
        np.testing.assert_almost_equal(E_local, [[0, 0, 0], [0, 0, 0]])

    def test_evaluate_monopole_field_at_atoms(self):
        dimer=PointDipoleList()
        dimer.append(PointDipole(group=2, coordinates=(0,0,0), charge=2))
        dimer.append(PointDipole(group=3, coordinates=(0,0,1)))
        E_atoms = dimer.evaluate_field_at_atoms()
        np.testing.assert_almost_equal(E_atoms, [[0,0,0],[0,0,2]])


    def test_evaluate_total_field_at_atoms(self):
        E_external = np.array([0., 0., 1.,])
        E_local = self.h2.evaluate_field_at_atoms(external=E_external)
        np.testing.assert_array_almost_equal(E_local, [[0, 0, 1], [0, 0, 1]])
        
    def test_dEi_dF_indirect(self):
        TR = self.h2._dEi_dF_indirect()
        np.testing.assert_array_almost_equal(TR, [
            [[-2.45481065*(0.11894578), 0., 0.],
             [0, -2.45481065*(0.11894578), 0.],
             [0., 0., 4.90962131*0.95899377]],
            [[-2.45481065*(0.11894578), 0., 0.],
             [0, -2.45481065*(0.11894578), 0.],
             [0., 0., 4.90962131*0.95899377]]
            ], decimal = 5)

    def test_dEi_dF(self):
        C = self.h2._dEi_dF()
        np.testing.assert_array_almost_equal(C, [
            [[1-2.45481065*(0.11894578), 0., 0.],
             [0, 1-2.45481065*(0.11894578), 0.],
             [0., 0., 1+4.90962131*0.95899377]],
            [[1-2.45481065*(0.11894578), 0., 0.],
             [0, 1-2.45481065*(0.11894578), 0.],
             [0., 0., 1+4.90962131*0.95899377]]
            ], decimal = 5)

    def test_neutral_zero_energy(self):
        E = self.dimer_template.total_energy()
        self.assertEqual(E, 0)

    def test_field_from_charges(self):
        charge_dimer = self.dimer_template
        charge_dimer.set_charges([1.0, 1.0])
        E = charge_dimer.evaluate_field_at_atoms()
        np.testing.assert_almost_equal(E, [[0,0,-1], [0,0,1]])

    def test_unconverged_solver_raises_exception(self):
        kwargs = {"max_it": 1}
        self.assertRaises(Exception, self.h2o_dimer.solve_scf, **kwargs)

    def test_alphas_as_matrix(self):
        a = np.array(range(9)).reshape((3,3))
        dimer = PointDipoleList()
        dimer.append(PointDipole(alpha=a))
        dimer.append(PointDipole(alpha=a))
        mat = dimer.alphas_as_matrix()
        np.testing.assert_almost_equal(
            mat, [
                [0, 1, 2, 0, 0, 0],
                [3, 4, 5, 0, 0, 0],
                [6, 7, 8, 0, 0, 0],
                [0, 0, 0, 0, 1, 2],
                [0, 0, 0, 3, 4, 5],
                [0, 0, 0, 6, 7, 8]
                ]
            )

    def test_read_dimer_unit(self):
        h2o_2 = PointDipoleList.from_string(H2O_DIMER_UNIT)
        self.assertEqual(len(h2o_2), 2)

    def test_unit_not_self_polarizing(self):
        h2o2 = PointDipoleList.from_string(H2O_DIMER_UNIT)
        h2o2.solve_scf()
        np.testing.assert_equal(h2o2.total_static_dipole_moment(), h2o2.total_dipole_moment())
        
if __name__ == "__main__":
    unittest.main()

