##########################################
# File: bezier.py                    #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import sympy as sy
import argparse
import numpy as np
from numpy import linalg as la
import math
import warnings

from subdivision import loop
from common import example_extraordinary_patch

# main
def main():

    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    # The bspline-to-bezier conversion matrix is given for N = 6
    # in Richard Stebbing's thesis (A.18).  Here we check that
    # the calculated matrix is correct.

    N = 6

    calculatedBezierWeights = loop.bspline_to_bezier_conversion(N)

    knownBezierWeights =[[ 2,12, 2, 0, 0, 0, 2, 2, 2, 2, 0, 0],
                         [ 2, 2, 0, 0, 0, 2,12, 2, 0, 0, 2, 2],
                         [12, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0],
                         [ 4,12, 3, 0, 0, 0, 3, 1, 0, 1, 0, 0],
                         [ 3,12, 1, 0, 0, 0, 4, 3, 1, 0, 0, 0],
                         [ 3, 4, 0, 0, 0, 1,12, 3, 0, 0, 1, 0],
                         [ 4, 3, 0, 0, 0, 3,12, 1, 0, 0, 0, 1],
                         [12, 3, 1, 0, 1, 3, 4, 0, 0, 0, 0, 0],
                         [12, 4, 3, 1, 0, 1, 3, 0, 0, 0, 0, 0],
                         [ 4, 8, 0, 0, 0, 0, 8, 4, 0, 0, 0, 0],
                         [ 8, 4, 0, 0, 0, 4, 8, 0, 0, 0, 0, 0],
                         [ 8, 8, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0],
                         [ 6,10, 1, 0, 0, 0, 6, 1, 0, 0, 0, 0],
                         [ 6, 6, 0, 0, 0, 1,10, 1, 0, 0, 0, 0],
                         [10, 6, 1, 0, 0, 1, 6, 0, 0, 0, 0, 0]]
    knownBezierWeights = np.asarray(knownBezierWeights, dtype=np.float64)
    knownBezierWeights /= 24.

    difference = knownBezierWeights - calculatedBezierWeights
    assert(np.allclose(difference, np.zeros(difference.shape, dtype = np.float64)))
    print "Success!  The Bezier conversion matrix was correctly calculated for N = 6."

if __name__ == '__main__':
    main()
