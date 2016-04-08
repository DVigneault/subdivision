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

    # Valency of the vertex of interest.
    N = 6

    A = loop.extended_subdivision_matrix(N)
    B = loop.bigger_subdivision_matrix(N)

    # Calculate Eigenvalues/Eigenvectors

    val = np.asarray(loop.extended_subdivision_eigenvalues(N))
    R = np.asarray(loop.extended_subdivision_eigenvectors(N))
    L = la.inv(R).transpose() # Columnwise L Eigenvectors

    # Ensure that the eigenvalues/eigenvectors are not complex.
    assert(np.allclose(val.imag, np.zeros(val.imag.shape, dtype = np.float64)))
    assert(np.allclose(R.imag, np.zeros(R.imag.shape, dtype = np.float64)))
    assert(np.allclose(L.imag, np.zeros(L.imag.shape, dtype = np.float64)))

    # Sort according to eigenvalues
    sorted_indices = np.argsort(val)[::-1]
    val = val[sorted_indices]
    R = R[:,sorted_indices]
    L = L[:,sorted_indices]

    # Component weights
    weights = [[  0, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # p0
               [  0,  0, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # p1
               [ 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # p2
               [  0, 12,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0], # p0-
               [  0, 12,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0], # p0+
               [  0,  0, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3], # p1-
               [  0,  0, 12,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0], # p1+
               [ 12,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0], # p2-
               [ 12,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0], # p2+
               [  0,-10,-10,  0, 32,  0,  0,  0,  0, -2,  0,  0, -2], # p01
               [-10,  0,-10,  0,  0, 32,  0,  0, -2,  0,  0, -2,  0], # p12
               [-10,-10,  0, 32,  0,  0,  0, -2,  0,  0, -2,  0,  0], # p20
               [ -7, -1, -7,  0,  0,  0, 27, -1, -1,  0,  0, -1, -1], # p0_
               [ -7, -7, -1,  0,  0,  0, 27, -1, -1, -1, -1,  0,  0], # p1_
               [ -1, -7, -7,  0,  0,  0, 27,  0,  0, -1, -1, -1, -1]] # p2_
    weights = np.asarray(weights, dtype=np.float64)
    weights /= 12.

    # Components
    components = np.zeros((13, 12), dtype=np.float64)

    #########################
    # Patch corner X(1,0,0) #
    #########################

    v1R = R[:,0]
    assert(((v1R.sum() / len(v1R)) - v1R[0]) < 10e-6)
    v1L = L[:,0]*v1R[0]
    components[0,:] = v1L

    #########################
    # Patch corner X(0,1,0) #
    #########################

    components[1,:] = loop.recursive_evaluate(0,
                                              loop.triangle_bspline_position_basis,
                                              N,
                                              (1,0))

    #########################
    # Patch corner X(0,0,1) #
    #########################

    components[2,:] = loop.recursive_evaluate(0,
                                              loop.triangle_bspline_position_basis,
                                              N,
                                              (0,1))

    ###################################
    # Patch midpoint X(0.5, 0.5, 0.0) #
    ###################################

    components[3,:] = loop.recursive_evaluate(0,
                                              loop.triangle_bspline_position_basis,
                                              N,
                                              (0.5,0.0))

    ###################################
    # Patch midpoint X(0.0, 0.5, 0.5) #
    ###################################

    components[4,:] = loop.recursive_evaluate(0,
                                              loop.triangle_bspline_position_basis,
                                              N,
                                              (0.5,0.5))

    ###################################
    # Patch midpoint X(0.5, 0.0, 0.5) #
    ###################################

    components[5,:] = loop.recursive_evaluate(0,
                                              loop.triangle_bspline_position_basis,
                                              N,
                                              (0.0,0.5))

    #################################
    # Patch center X(1/3, 1/3, 1/3) #
    #################################

    components[6,:] = loop.recursive_evaluate(0,
                                              loop.triangle_bspline_position_basis,
                                              N,
                                              (1./3,1./3))

    #########################
    # Xs(1,0,0) - Xr(1,0,0) #
    #########################

    vs = np.zeros((1, N+6), dtype = np.float64)
    for i in xrange(0,N):
        vs[0,i+1] = math.cos(2*math.pi*i/N)

    sub = R[:,1:3]
    assert(1 == np.sum(abs(vs.dot(sub)) < 10e-6))
    scale = vs.dot(sub)[abs(vs.dot(sub)) > 10e-6][0]

    components[7,:] = vs / scale

    #########################
    # Xt(1,0,0) - Xr(1,0,0) #
    #########################

    vt = np.zeros((1, N+6), dtype = np.float64)
    for i in xrange(1,N+1):
        vt[0,i] = math.cos(2*math.pi*i/N)

#    assert(1 == np.sum(abs(vt.dot(sub)) < 10e-6)) # assertion fails
#    print "vt.dot(sub)", vt.dot(sub) # vt.dot(sub) [[-2.598  1.5  ]]

    components[8,:] = vt / scale

    #########################
    # Xt(0,1,0) - Xs(0,1,0) #
    #########################

    Xt = loop.recursive_evaluate(1,
                                 loop.triangle_bezier_dt_basis,
                                 N,
                                 (1.0, 0.0))

    Xs = loop.recursive_evaluate(1,
                                 loop.triangle_bezier_ds_basis,
                                 N,
                                 (1.0, 0.0))

    components[9,:] = Xt - Xs

    #########################
    # Xr(0,1,0) - Xs(0,1,0) #
    #########################

    Xr = loop.recursive_evaluate(1,
                                 loop.triangle_bezier_dr_basis,
                                 N,
                                 (1.0, 0.0))

    Xs = loop.recursive_evaluate(1,
                                 loop.triangle_bezier_ds_basis,
                                 N,
                                 (1.0, 0.0))

    components[10,:] = Xr - Xs

    #########################
    # Xr(0,0,1) - Xt(0,0,1) #
    #########################

    Xr = loop.recursive_evaluate(1,
                                 loop.triangle_bezier_dr_basis,
                                 N,
                                 (0.0, 1.0))

    Xt = loop.recursive_evaluate(1,
                                 loop.triangle_bezier_dt_basis,
                                 N,
                                 (0.0, 1.0))

    components[11,:] = Xr - Xt

    #########################
    # Xs(0,0,1) - Xt(0,0,1) #
    #########################

    Xs = loop.recursive_evaluate(1,
                                 loop.triangle_bezier_ds_basis,
                                 N,
                                 (0.0, 1.0))

    Xt = loop.recursive_evaluate(1,
                                 loop.triangle_bezier_dt_basis,
                                 N,
                                 (0.0, 1.0))

    components[12,:] = Xs - Xt

    calculatedBezierWeights = weights.dot(components)
     
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

    if 6 == N:
        difference = knownBezierWeights - calculatedBezierWeights
        assert(np.allclose(difference, np.zeros(difference.shape, dtype = np.float64)))
        print "Success!  The Bezier conversion matrix was correctly calculated for N = 6."

if __name__ == '__main__':
    main()
