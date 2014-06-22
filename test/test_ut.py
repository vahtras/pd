from unittest import TestCase
from ..ut import *

class TestTriangular(TestCase):
    def test_setUp(self):
        pass

    def test_twodim(self):
        two_seq = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
        self.assertEqual(tuple(upper_triangular(2)), two_seq)

    def test_threedim(self):
        three_seq = (
            (0, 0, 0), (0, 0, 1), (0, 0, 2),
            (0, 1, 1), (0, 1, 2),
            (0, 2, 2),
                       (1, 1, 1), (1, 1, 2),
                       (1, 2, 2),
                                  (2, 2, 2)
            )
        self.assertEqual(tuple(upper_triangular(3)), three_seq)


