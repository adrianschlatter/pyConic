# -*- coding: utf-8 -*-
"""
Unit tests for ConicSection

:author: Adrian Schlatter
"""

import unittest
from pyConic import ConicSection
import numpy as np


class Test_UnitCircle(unittest.TestCase):
    """Test unit-circle"""

    def setUp(self):
        self.A33 = [[1, 0], [0, 1]]
        self.v = [[0], [0]]
        self.F = -1
        self.cs = ConicSection(self.A33, self.v, self.F)

    def test_proper(self):
        """A circle is proper (non-degenerate)"""

        self.assertTrue(self.cs.isProper)
        self.assertFalse(self.cs.isDegenerate)

    def test_center(self):
        """This circle is centered at (0, 0)"""

        self.assertTrue((self.cs.center == np.array([[0], [0]])).all())

    def test_coefficients(self):
        """Test correctnes of unit-circle coefficients"""

        coefs = self.cs.coefficients

        self.assertEqual(coefs, (1, 0, 1, 0, 0, -1))

    def test_A33(self):
        """Test A33"""
        self.assertTrue((np.matrix(self.A33) == self.cs.A33).all())

    def test_v(self):
        """Test v"""
        self.assertTrue((np.matrix(self.v) == self.cs.v).all())

    def test_F(self):
        """Test F"""
        self.assertEqual(np.matrix(self.F), self.cs.F)

    def test_conicType(self):
        """Test that circle has conic-type 'circle'"""
        self.assertEqual('circle', self.cs.conicType)

    def test_central(self):
        """Unit circle is centered"""
        self.assertTrue(self.cs.isCentral)

    def test_standardForm(self):
        """Test correct standard form of unit circle"""

        a, b, t, M = self.cs.standardForm
        self.assertEqual((a, b, t[0, 0], t[1, 0]), (1., 1., 0., 0.))

    def test_centeredEquation(self):
        """Test centered equation of unit circle"""

        A33, K = self.cs.centeredEquation
        self.assertTrue((self.A33 == A33).all())
        self.assertEqual(K, 1.)

if __name__ == '__main__':
    unittest.main()
