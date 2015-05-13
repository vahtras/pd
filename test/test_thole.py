import unittest
import numpy as np

from ..thole import TholeList
from data_thole import *



class TholeTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_H2_iso(self):

        h2 = TholeList.from_string(H2["POTFILE"])
        self.assertAlmostEqual(h2.alpha_iso(), H2["ALPHA_ISO"], places=DECIMALS)



