import unittest
import random
import math
import numpy as np
from ..particles import *
from ..quadrupole import *
from ..gaussian import *
from .util import *

class RandomQuadrupole(Quadrupole):
    def __init__(self):
        Quadrupole.__init__(self,
            coordinates=random_vector(),
            charge=random_scalar(),
            dipole=random_vector(),
            quadrupole=random_two_rank_triangular(),
            alpha=random_tensor(),
            beta=random_tensor2(),
            local_field=random_vector()
            )

class PointQuadrupoleFiniteFieldTests(unittest.TestCase):

    def setUp(self):
        self.particle = RandomQuadrupole()

    def test_finite_difference_permanent_dipole_energy(self):
        gradE = field_gradient(self.particle.permanent_dipole_energy)
        np.testing.assert_almost_equal(-gradE, self.particle.permanent_dipole_moment())
    def test_finite_difference_permanent_dipole_energy(self):
        gradE = field_gradient(self.particle.permanent_dipole_energy)
        np.testing.assert_almost_equal(-gradE, self.particle.permanent_dipole_moment())

    def test_finite_difference_alpha_induced_dipole_energy(self):
        gradE = field_gradient(self.particle.alpha_induced_dipole_energy)
        np.testing.assert_almost_equal(-gradE, self.particle.alpha_induced_dipole_moment())

    def test_finite_difference_beta_induced_dipole_energy(self):
        gradE = field_gradient(self.particle.beta_induced_dipole_energy)
        np.testing.assert_almost_equal(-gradE, self.particle.beta_induced_dipole_moment())

    def test_finite_difference_induced_dipole_energy(self):
        gradE = field_gradient(self.particle.induced_dipole_energy)
        p_ind = self.particle.induced_dipole_moment()
        np.testing.assert_almost_equal(-gradE, p_ind)

    def test_finite_difference_total_dipole_energy(self):
        gradE = field_gradient(self.particle.dipole_energy)
        np.testing.assert_almost_equal(-gradE, self.particle.dipole_moment())

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

class PointQuadrupoleListFiniteFieldTests(unittest.TestCase):
    
    def setUp(self):
        self.charges = QuadrupoleList.from_string("""AU
2 0 0
1 0 0 0 1.0
2 0 0 1 1.0
"""
)

        self.h2o_monomer=QuadrupoleList.from_string("""AU
1 1 1 0
1 0.00000  0.00000  5.48861 0.0 0.00000 -0.00000 -0.76539  6.61822 
""")

        self.h2o_dimer=QuadrupoleList.from_string("""AU
2 1 1 0
1 0.00000  0.00000  0.48861 0.0 0.00000 -0.00000 -0.76539  6.61822
2 0.00000  0.00000  5.48861 0.0 0.00000 -0.00000 -0.76539  6.61822 
""")


        self.h2o_monomer_hyp=QuadrupoleList()

        self.h2o_monomer_hyp.append(
            PointDipole(
                coordinates=(0, 0, 0.48861),
                charge=0,
                dipole=(0, 0, -0.81457755),
                alpha=(
                    (7.21103625278, 0, 0), 
                    (0, 3.03446384360, 0),
                    (0, 0, 5.22710462524)
                ),
                beta=(
                    (
                        (0, 0, -18.48299918),
                        (0, 0, 0),
                        (-18.48299918,0,0)
                    ),
                    (
                        (0, 0, 0),
                        (0, 0, -2.33649395),
                        (0, -2.33649395, 0)
                    ),
                    (
                        (-18.48299798, 0, 0),
                        (0, -2.33649400, 0),
                        (0, 0, -11.17349291)
                    )
                )
            )
        )

        self.h2o_dimer_hyp=PointDipoleList()

        self.h2o_dimer_hyp.append(
            PointDipole(
                group=1,
                coordinates=(0, 0, 0),
                charge=0,
                dipole=(0, 0, -0.81457755),
                alpha=(
                    (7.21103625278, 0, 0), 
                    (0, 3.03446384360, 0),
                    (0, 0, 5.22710462524)
                ),
                beta=(
                    (
                        (0, 0, -18.48299918),
                        (0, 0, 0),
                        (-18.48299918,0,0)
                    ),
                    (
                        (0, 0, 0),
                        (0, 0, -2.33649395),
                        (0, -2.33649395, 0)
                    ),
                    (
                        (-18.48299798, 0, 0),
                        (0, -2.33649400, 0),
                        (0, 0, -11.17349291)
                    )
                )
            )
        )

        self.h2o_dimer_hyp.append(
            PointDipole(
                group=2,
                coordinates=(0, 0, 10),
                charge=0,
                dipole=(0, 0, -0.81457755),
                alpha=(
                    (7.21103625278, 0, 0), 
                    (0, 3.03446384360, 0),
                    (0, 0, 5.22710462524)
                ),
                beta=(
                    (
                        (0, 0, -18.48299918),
                        (0, 0, 0),
                        (-18.48299918,0,0)
                    ),
                    (
                        (0, 0, 0),
                        (0, 0, -2.33649395),
                        (0, -2.33649395, 0)
                    ),
                    (
                        (-18.48299798, 0, 0),
                        (0, -2.33649400, 0),
                        (0, 0, -11.17349291)
                    )
                )
            )
        )
