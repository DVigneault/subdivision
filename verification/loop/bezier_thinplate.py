import sympy as sp
import numpy as np
import math
import warnings
import copy

from subdivision import loop
from common import example_extraordinary_patch

x, y = sp.symbols('x y')
r, s, t = sp.symbols('r s t')
r1, r2, r3, r4 = sp.symbols('r1 r2 r3 r4')
s1, s2, s3, s4 = sp.symbols('s1 s2 s3 s4')
t1, t2, t3, t4 = sp.symbols('t1 t2 t3 t4')

def expand_powers(expr):
  rep_r = [(r1,r**1),(r2,r**2),(r3,r**3),(r4,r**4)]
  rep_s = [(s1,s**1),(s2,s**2),(s3,s**3),(s4,s**4)]
  rep_t = [(t1,t**1),(t2,t**2),(t3,t**3),(t4,t**4)]

  expr = expr.subs(rep_r)
  expr = expr.subs(rep_s)
  expr = expr.subs(rep_t)

  return expr

def bezier_position_function():
  position_basis = sp.Matrix([s4,           # p0
                              t4,           # p1
                              r4,           # p2
                              4*r1*s3,      # p0-
                              4*t1*s3,      # p0+
                              4*s1*t3,      # p1-
                              4*r1*t3,      # p1+
                              4*t1*r3,      # p2-
                              4*s1*r3,      # p2+
                              6*s2*t2,      # p01
                              6*t2*r2,      # p12
                              6*r2*s2,      # p20
                              12*s2*t1*r1,  # p0m
                              12*s1*t2*r1,  # p0m
                              12*s1*t1*r2]) # p0m

  position_basis = expand_powers(position_basis)
  return position_basis

def main():

  # Global options
  np.set_printoptions(precision=3, suppress=True, linewidth=100)

  ###########################
  # Generate Ordinary Patch #
  ###########################

  X = example_extraordinary_patch(6) # BSpline
  B = loop.bspline_to_bezier_conversion(6).dot(X)

  #################
  # Check corners #
  #################

  assert(np.allclose(B[0] - X[1], np.zeros(X.shape)))
  assert(np.allclose(B[1] - X[6], np.zeros(X.shape)))
  assert(np.allclose(B[2] - X[0], np.zeros(X.shape)))

  ##############################
  # Check points in the middle #
  ##############################

  bezier_position = bezier_position_function()

  for a in np.arange(0.1, 1.0, 0.1):
    for b in np.arange(0.1, 1 - a, 0.1):
      bezier = B.transpose() * bezier_position.subs( { r : 1 - a - b, s : a, t : b } )
      bezier = np.array(bezier.tolist()).astype(np.float64).transpose()[0,:]
      bspline = loop.recursive_evaluate(0,
                                        loop.triangle_bspline_position_basis,
                                        6,
                                        (a,b),
                                        X)
      assert(np.allclose(bezier - bspline,np.zeros(bezier.shape)))

  #####################
  # Thin plate energy #
  #####################

  bezier_position = bezier_position.subs( { r : 1 - s - t } )
  bezier_position = bezier_position.subs( { s : x - t/2, t : y * sp.sqrt(3) } )

  bezier_basis_xx = copy.deepcopy(bezier_position).diff(x).diff(x)
  bezier_basis_xy = copy.deepcopy(bezier_position).diff(x).diff(y)
  bezier_basis_yy = copy.deepcopy(bezier_position).diff(y).diff(y)

  xx = bezier_basis_xx * bezier_basis_xx.transpose()
  xy = bezier_basis_xy * bezier_basis_xy.transpose()
  yy = bezier_basis_yy * bezier_basis_yy.transpose()

  i_x = lambda expression: expression.integrate((x, y/sp.sqrt(3), (1-y/sp.sqrt(3))))
  i_y = lambda expression: expression.integrate((y, 0, sp.sqrt(3)/2))

  M = (2/sp.sqrt(3))*i_y(i_x(xx + 2*xy + yy))
  M = np.array(M.tolist()).astype(np.float64)

#  sp.pprint(M)

  B = B.flatten('F')
  M_Block = np.zeros((M.shape[0]*2, M.shape[1]*2))
  M_Block[0:M.shape[0], 0:M.shape[1]] = M
  M_Block[M.shape[0]:, M.shape[1]:] = M

  print B.dot(M_Block).dot(B)

if __name__ == '__main__':
    main()

