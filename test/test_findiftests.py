import unittest
import numpy as np
from ..particles import *
from util import field_gradient

class PointDipoleFiniteFieldTests(unittest.TestCase):

    def setUp(self):
        self.particle = PointDipole()

    def test_finite_difference_energy(self):
#zero
        gradE = field_gradient(self.particle.total_energy)
        dipole = self.particle._p0 + \
           self.particle.induced_dipole_moment()

        np.testing.assert_almost_equal(-gradE, dipole)


    def test_finite_difference_permanent_dipole_energy(self):
        self.particle._p0 = random_vector()
        gradE = field_gradient(self.particle.permanent_dipole_energy)
        np.testing.assert_almost_equal(-gradE, self.particle.permanent_dipole_moment())

    def test_finite_difference_alpha_induced_dipole_energy(self):
        self.particle._a0 = random_tensor()
        gradE = field_gradient(self.particle.alpha_induced_dipole_energy)
        np.testing.assert_almost_equal(-gradE, self.particle.alpha_induced_dipole_moment())

    def test_finite_difference_induced_dipole_energy(self):
        self.particle._a0 = random_tensor()
        gradE = field_gradient(self.particle.alpha_induced_dipole_energy)
        np.testing.assert_almost_equal(-gradE, self.particle.induced_dipole_moment())

    def test_finite_difference_total_dipole_energy(self):
        self.particle._p0 = random_vector()
        self.particle._a0 = random_tensor()
        gradE = field_gradient(self.particle.dipole_energy)
        np.testing.assert_almost_equal(-gradE, self.particle.dipole_moment())

    def test_finite_difference_permanent_dipole_moment(self):
        self.particle._p0 = np.random.random(3)
        gradE = field_gradient(self.particle.permanent_dipole_moment)
        np.testing.assert_almost_equal(-gradE, np.zeros((3, 3)))

    def test_finite_difference_alpha_induced_dipole_moment(self):
        self.particle._a0 = random_tensor()
        gradp = field_gradient(self.particle.alpha_induced_dipole_moment)
        np.testing.assert_almost_equal(gradp, self.particle._a0)

    def test_finite_difference_beta_induced_dipole_moment(self):
        self.particle._b0 = random_tensor2()
        self.particle.set_local_field(random_vector())
        gradp = field_gradient(self.particle.beta_induced_dipole_moment)
        ref = np.dot(self.particle._b0, self.particle.local_field())
        np.testing.assert_almost_equal(gradp, ref)

    def test_finite_difference_total_dipole_moment(self):
        self.particle._p0 = random_vector()
        self.particle._a0 = random_tensor()
        gradp = field_gradient(self.particle.dipole_moment)
        np.testing.assert_almost_equal(gradp, self.particle._a0)


class PointDipoleListFiniteFieldTests(unittest.TestCase):
    
    def setUp(self):
        self.charges = PointDipoleList.from_string("""AU
2 0 0
1 0 0 0 1.0
2 0 0 1 1.0
"""
)

        self.h2o_dimer=PointDipoleList.from_string("""AU
2 1 1 0
1 0.00000  0.00000  0.48861 0.0 0.00000 -0.00000 -0.76539  6.61822
1 0.00000  0.00000  5.48861 0.0 0.00000 -0.00000 -0.76539  6.61822 
""")

    def test_finite_difference_polarizable_dimer_z(self):
        alphas = self.h2o_dimer.solve_Applequist_equation()
        eps = .001
        self.h2o_dimer.solve_scf_for_external((0,0,eps/2))
        p0a = self.h2o_dimer[0].induced_dipole_moment()
        self.h2o_dimer.solve_scf_for_external((0,0,-eps/2))
        p0b = self.h2o_dimer[0].induced_dipole_moment()
        dp0_dF = (p0a - p0b)/eps
        self.assertAlmostEqual(dp0_dF[2], alphas[0][2, 2], places=3)

    def test_finite_difference_polarizable_dimer_x(self):
        alphas = self.h2o_dimer.solve_Applequist_equation()
        eps = .001
        self.h2o_dimer.solve_scf_for_external((eps/2,0,0))
        p0a = self.h2o_dimer[0].induced_dipole_moment()
        self.h2o_dimer.solve_scf_for_external((-eps/2,0,0))
        p0b = self.h2o_dimer[0].induced_dipole_moment()
        dp0_dF = (p0a - p0b)/eps
        self.assertAlmostEqual(dp0_dF[0], alphas[0][0, 0], places=3)

    def test__field_vs_external_field(self):
        pass
        

def random_vector():
    return np.random.random(3)

def random_tensor():
    a = np.random.random((3, 3))
    a = 0.5*(a + a.T)
    return  a

def random_tensor2():
    b = np.random.random((3, 3, 3))
    b = b + b.transpose((1, 2, 0)) +  b.transpose((2, 0, 1)) +\
        b.transpose((1, 0, 2)) + b.transpose((2, 1, 0)) + b.transpose((0, 2, 1))
    return  b
