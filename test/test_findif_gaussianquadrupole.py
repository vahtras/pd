import unittest
import random
import math
import numpy as np
from ..particles import *
from ..quadrupole import *
from ..gaussian import *
from .util import *

class RandomGaussianQuadrupole(GaussianQuadrupole):
    def __init__(self):
        GaussianQuadrupole.__init__(self,
            coordinates=random_vector(),
            charge=random_scalar(),
            dipole=random_vector(),
            quadrupole=random_two_rank_triangular(),
            alpha=random_tensor(),
            beta=random_tensor2(),
            local_field=random_vector()
            )
        
class GaussianQuadrupoleFiniteFieldTests(unittest.TestCase):

    def setUp(self):
        self.particle = RandomGaussianQuadrupole()

    def test_finite_difference_permanent_dipole_moment(self):
        gradE = field_gradient(self.particle.permanent_dipole_moment)
        np.testing.assert_almost_equal(-gradE, np.zeros((3, 3)))

    def test_finite_difference_alpha_induced_dipole_moment(self):
        gradp = field_gradient(self.particle.alpha_induced_dipole_moment)
        np.testing.assert_almost_equal(gradp, self.particle._a0)

    def test_finite_difference_beta_induced_dipole_moment(self):
        self.particle.set_local_field(random_vector())
        gradp = field_gradient(self.particle.beta_induced_dipole_moment)
        ref = np.dot(self.particle._b0, self.particle.local_field())
        np.testing.assert_almost_equal(gradp, ref)

    def test_finite_difference_total_dipole_moment(self):
        gradp = field_gradient(self.particle.dipole_moment)
        np.testing.assert_almost_equal(gradp, self.particle.a)

    def test_finite_difference_hessian_dipole_moment(self):
        hess_p = field_hessian(self.particle.dipole_moment)
        np.testing.assert_almost_equal(hess_p, self.particle._b0)


