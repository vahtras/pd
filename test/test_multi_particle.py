import unittest
import numpy as np
from ..particles import PointDipole, PointDipoleList
from .test_particles import H2
from .util import iterize

class MultiDipoleTest(unittest.TestCase):

    def setUp(self):
        self.He2 = PointDipoleList(iter("""AA
2 2 0 1
1 0 0  0 0 6.429866513469e-01
1 0 0 52 0 6.429866513469e-01
""".split("\n")))
        pass

        self

    def test_he2(self):
        alpha_he2 = self.He2.alpha()
        ref_alpha = np.diag(
            [1.285972432811e+00, 1.285972432811e+00,1.285975043401e+00]
            )
        np.allclose(alpha_he2, ref_alpha)

    def test_verify_solver(self):
        h2 = PointDipoleList(iterize(H2["POTFILE"]))
        pass

