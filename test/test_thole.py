import unittest
import numpy as np

from ..thole import TholeList
from data_thole import *


DECIMALS=2

class TholeTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_H2_iso(self):

        h2 = TholeList.from_string(H2["POTFILE"])
        print h2
        print 'additive:', h2[0]._a0.trace()/3 * 2
        print 'ref:', H2["ALPHA_ISO"]
        print 'got alpha iso:', h2.alpha_iso()
        self.assertAlmostEqual( h2.alpha_iso(), H2["ALPHA_ISO"], places=DECIMALS )


