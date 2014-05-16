import unittest
import numpy as np
from ..particles import PointDipole, PointDipoleList

class MultiDipoleTest(unittest.TestCase):

    def setUp(self):
        self.He2 = PointDipoleList(iter("""AA
2 2 0 1
1 0 0  0 0 6.429866513469e-01
1 0 0 52 0 6.429866513469e-01
""".split("\n")))
        pass

    def test_he2(self):
       alpha_he2 = self.He2.alpha()
       alpha_he = self.He2[0].a
       np.allclose(2*alpha_he, alpha_he2)

