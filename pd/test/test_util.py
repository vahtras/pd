import unittest
from util import iterize
from nose.plugins.attrib import attr

@attr(speed='fast')
class TestUtil(unittest.TestCase):
    def setUp(self):
        pass

    def test_iterize(self):
        ref = """\
abc
def
"""
        this = "\n".join(line for line in iterize(ref))
        self.assertEqual(this, ref)
