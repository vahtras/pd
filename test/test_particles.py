import unittest
import numpy as np
from ..particles import PointDipole, PointDipoleList
from .util import iterize
from const import *
from test_applequist import H2

# Other model systems

H2O_DIMER = """AU
2 1 1 0
1 0.00000  0.00000  0.48861 0.0 0.00000 -0.00000 -0.76539  6.61822
1 0.00000  0.00000  5.48861 0.0 0.00000 -0.00000 -0.76539  6.61822 
"""
        
class PointDipoleListTest(unittest.TestCase):

    def setUp(self):
        self.h2 = PointDipoleList.from_string(H2["POTFILE"])
        self.h2o_dimer = PointDipoleList.from_string(H2O_DIMER)
        pass

    def test_h2o_dimer_finite_field_p(self):

        alphas = self.h2o_dimer.solve_Applequist_equation()

        self.h2o_dimer.solve_scf_for_external([0, 0, .005])
        p1 = self.h2o_dimer[0].dp
        self.h2o_dimer.solve_scf_for_external([0, 0, -.005])
        p2 = self.h2o_dimer[0].dp
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
        np.testing.assert_array_equal(h2_rhs, h2_rhs_ref)

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
        np.testing.assert_almost_equal(alphas, alphas_ref)

    def test_induced_parallel_dipole_on_one_atom(self):
        E_external = np.array([0., 0., 1.,])
        p_ind_ref =  np.array([0., 0., 0.95899377])
        self.h2.solve_scf_for_external(E_external, max_it = 100)
        np.testing.assert_almost_equal(self.h2[0].p, p_ind_ref, decimal=6)

    def test_induced_orthogonal_dipole_on_one_atom(self):
        E_external = np.array([1., 0., 0.,])
        p_ind_ref =  np.array([0.11894578, 0., 0.])
        self.h2.solve_scf_for_external(E_external, max_it = 100)
        np.testing.assert_almost_equal(self.h2[0].p, p_ind_ref, decimal=6)
        
    def test_evaluate_local_field_at_atoms(self):
        E_external = np.array([0., 0., 1.,])
        E_local = self.h2.evaluate_field_at_atoms()
        np.testing.assert_almost_equal(E_local, [[0, 0, 0], [0, 0, 0]])

    def test_evaluate_total_field_at_atoms(self):
        E_external = np.array([0., 0., 1.,])
        E_local = self.h2.evaluate_field_at_atoms(external=E_external)
        np.testing.assert_array_almost_equal(E_local, [[0, 0, 1], [0, 0, 1]])
        

        
if __name__ == "__main__":
    unittest.main()

