import unittest
import numpy as np
from ..particles import PointDipole

class MultiDipoleTest(unittest.TestCase):

    def setUp(self):
        self.He2 = iter("""AA
2 2 0 1
1 0 0  0 0 6.429866513469e-01
1 0 0 52 0 6.429866513469e-01
""").split("\n")
        pass

