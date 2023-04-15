# -*- coding: utf-8 -*-
"""
Class representing a conic section.

:author: Adrian Schlatter
"""

import numpy as np


class ConicSection(object):
    """
    Represents a conic section given by the equation

        x.T * A33 * x + v.T * x + F = 0.

    It has a wealth of properties to classify and transform the conic section.
    """

    def __init__(self, A33, v, F):
        A33, v, F = [np.array(x, ndmin=2) for x in [A33, v, F]]
        v.shape = (-1, 1)

        if A33.shape != (2, 2):
            raise ValueError("A33 has to be a 2x2 matrix")
        if v.shape != (2, 1):
            raise ValueError("v has to be a 2x1 matrix")
        if F.shape != (1, 1):
            raise ValueError("F has to be convertible to a 1x1 matrix")
        if (A33 != A33.T).all():
            raise ValueError("A33 must be symmetric")

        self.AQ = np.block([[A33, v], [v.T, F]])

    def __repr__(self):
        s = '{0} * x**2 + {1} * x * y + {2} * y**2 + {3} * x + ' \
            '{4} * y + {5} = 0'
        return(s.format(*self.coefficients))

    @property
    def coefficients(self):
        """
        Coefficients of the polynomial equation of the conic section:

        A * x**2 + B * x * y + C * y**2 + D * x + E * y + F = 0
        """

        A = self.AQ[0, 0]
        B = 2 * self.AQ[0, 1]
        C = self.AQ[1, 1]
        D = 2 * self.AQ[0, 2]
        E = 2 * self.AQ[1, 2]
        F = self.AQ[2, 2]
        return(A, B, C, D, E, F)

    @property
    def A33(self):
        """A33 of AQ = np.bmat([[A33, v], [v.T, F]])"""
        return(self.AQ[:2, :2])

    @property
    def v(self):
        """v of AQ = np.bmat([[A33, v], [v.T, F]])"""
        return(self.AQ[:2, 2])

    @property
    def F(self):
        """F of AQ = np.bmat([[A33, v], [v.T, F]])"""
        return(self.AQ[2:, 2:])

    @property
    def isProper(self):
        """Returns true if conic section is proper (not degenerate)"""
        return(np.linalg.det(self.AQ) != 0)

    @property
    def isDegenerate(self):
        """Returns true if conic section is degenerate (not proper)"""
        return(not self.isProper)

    @property
    def conicType(self):
        """Returns the type of the conic section"""

        detA33 = np.linalg.det(self.A33)
        detAQ = np.linalg.det(self.AQ)
        A, B, C, D, E, F = self.coefficients

        if self.isProper:
            if detA33 < 0:
                return('hyperbola')
            elif detA33 > 0:   # ellipse
                if A == C and B == 0:
                    return('circle')
                else:
                    if (A + C) * detAQ < 0:
                        return('real ellipse')
                    elif (A + C) * detAQ > 0:
                        return('imaginary ellipse')
            else:
                return('parabola')
        else:
            if detA33 < 0:
                return('intersecting lines')
            elif detA33 > 0:
                return('point')
            else:
                if D**2 + E**2 > 4 * (A + C) * F:
                    return('distinct real parallel lines')
                elif D**2 + E**2 < 4 * (A + C) * F:
                    return('distinct imaginary parallel lines')
                else:
                    return('coincident lines')

    @property
    def isCentral(self):
        """Returns true if conic section is central (i.e., not a parabola)"""

        return(np.linalg.det(self.A33) != 0)

    @property
    def center(self):
        """The center of the conic section"""

        if not self.isCentral:
            raise TypeError('Only central conics have a center')

        return(-np.linalg.inv(self.A33) * self.v)

    @property
    def centeredEquation(self):
        """
        Shifts conic so that its center is at (0, 0).T and returns A33 and K
        of its equation:

            x.T * A33 * x = K
        """
        K = - np.linalg.det(self.AQ) / np.linalg.det(self.A33)
        return(self.A33, K)

    @property
    def standardForm(self):
        """
        Returns coefficients of standard form (i.e., coordinate origin
        shifted to center of conic, axes rotated to coincide with conic axes)
        as well as shift-vector t and coordinate transformation M.

        The standard form is:
            a * x'**2 + b * y'**2 = 1

        where
            x' = M.I * (x - t)

        Returns:

            a, b:   Standard-form coefficients
            t:      Shift vector
            M:      Rotation matrix
        """

        w, v = np.linalg.eigh(self.A33)
        K = self.centeredEquation[1]

        return(w[0] / K, w[1] / K, self.center, v)
