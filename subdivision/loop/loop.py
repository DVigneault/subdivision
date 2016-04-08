##########################################
# File: loop.py                          #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import numpy as np
import re
import sympy as sp
from functools import partial
from operator import add, mul
from scipy import linalg as la

# Requires `rscommon`.
from rscommon.sympy_ import sympy_polynomial_to_function

# __all__.
__all__ = ['g',
           'extended_subdivision_eigenvalues',
           'extended_subdivision_eigenvectors',
           'subdivision_matrix',
           'extended_subdivision_matrix',
           'bigger_subdivision_matrix',
           'picker_matrix',
           'transform_u_to_subdivided_patch',
           'recursive_evaluate',
           'triangle_bspline_position_basis',
           'triangle_bspline_du_basis',
           'triangle_bspline_dv_basis',
           'triangle_bspline_du_du_basis',
           'triangle_bspline_du_dv_basis',
           'triangle_bspline_dv_dv_basis',
           'triangle_bezier_dr_basis',
           'triangle_bezier_ds_basis',
           'triangle_bezier_dt_basis']

# Loop Subdivision Matrices

# `g` is a namespace which provides `cos`, `pi` and `Rational`.
# Setting `g` at the module level allows manipulation of the underlying numeric
# type used for the Doo-Sabin weights, matrices, and basis functions.
# e.g. `g` can be replaced with the sympy module.
class g(object):
    cos = np.cos
    pi = np.pi
    @staticmethod
    def Rational(a, b):
        return float(a) / b

# calculate_alpha
def calculate_alpha(N):
    return g.Rational(5, 8) - (3 + 2 * g.cos(2 * g.pi / N))**2 / 64

# calculate_f
def calculate_f(k, N):
    return g.Rational(3, 8) + g.Rational(2, 8) * g.cos(2 * g.pi * k / N)

# The subdivision matrix, extended subdivision matrix, and "bigger" subdivision
# matrix are all terms from: Stam, "Evaluation of Loop Subdivision Surfaces".

# Eigenvalues

# subdivision_eigenvalues
def subdivision_eigenvalues(N): # Sigma in Stam's paper
    s = []
    s.append(1)
    s.append( g.Rational(5, 8) - calculate_alpha(N) )
    for i in xrange(3, N+2):
        s.append( calculate_f(i-2,N) )
    return s


# s12_eigenvalues
def s12_eigenvalues(): # Delta in Stam's paper
    a = g.Rational(1, 8)
    b = g.Rational(1, 16)
    return [a, a, a, b, b]

# extended_subdivision_eigenvalues
def extended_subdivision_eigenvalues(N): # Lambda in Stam's paper
    s = subdivision_eigenvalues(N)
    s12 = s12_eigenvalues()
    return s + s12

# Eigenvectors

# subdivision_eigenvectors
def subdivision_eigenvectors(N): # U0 in Stam's paper

    # Calculate *complex* eigenvectors.
    s = np.ones((N+1, N+1), dtype=complex)
    s[0,:] = 0
    s[0,0] = 1
    s[0,1] = g.Rational(-8, 3) * calculate_alpha(N)
    indices = [i+1 + 0.j for i in xrange(0,N-1)]
    col_indices = np.zeros((N-1,1), dtype=complex)
    row_indices = np.zeros((1,N-1), dtype=complex)
    col_indices[:,0] = indices
    row_indices[0,:] = indices
    indices = col_indices.dot(row_indices)
    s[2:,2:] = np.exp(2.j * g.pi * indices / N)

    # Calculate *real* eigenvectors.
    if (1 == N % 2):
        print "odd"
    else:
        for u in xrange(3, (N+1-2-1)/2+3):
            l_c = s[:,u-1]
            r_c = s[:,N-(u-3)]
            l_r = (l_c + r_c) / 2
            r_r = (l_c - r_c) / 2j
            s[:,u-1] = l_r
            s[:,N-(u-3)] = r_r

    assert(np.allclose(np.imag(s), np.zeros(np.shape(s))))
    s = np.real(s).tolist()

    return s

# s12_eigenvectors
def s12_eigenvectors(): # W1 in Stam's paper
    return [ [0,-1, 1, 0, 0],
             [1,-1, 1, 0, 1],
             [1, 0, 0, 0, 0],
             [0, 0, 1, 1, 0],
             [0, 1, 0, 0, 0] ]

# subdivision_eigenvectors_lower_block
def subdivision_eigenvectors_lower_block(N): # U1 in Stam's paper
    # Sylvester Equation:
    # A * X + X * B = Q

    # From Stam:
    # u1 * sigma - s12 * u1 = s11 * u0

    # Fit to Sylvester:
    # (u1 * sigma - s12 * u1)^T = (s11 * u0)^T
    # (u1 * sigma)^T + (-1 * s12 * u1)^T = (s11 * u0)^T
    # Transpose distributes over multiplication: (A * B)^T = B^T * A^T
    # sigma is diagonal : (sigma^T = sigma)
    # sigma * u1^T + u1^T * (-1 * s12)^T = (s11 * u0)^T
    # |___|   |__|   |__|   |__________|   |__________|
    #   A       X      X          B              Q

    sigma = np.diag(np.asarray(subdivision_eigenvalues(N)))
    s12 = np.asarray(s12_matrix())
    s11 = np.asarray(s11_matrix(N))
    u0 = np.asarray(subdivision_eigenvectors(N))

    A = sigma                     # (M, M)
    B = (-1 * s12).transpose()    # (N, N)
    Q = (s11.dot(u0)).transpose() # (M, N)
    X = la.solve_sylvester(A, B, Q).transpose() # (M, N)

    return X.tolist()

# extended_subdivision_eigenvectors
def extended_subdivision_eigenvectors(N): # V in Stam's paper

    # V = | u0  0 |
    #     | u1 w1 |

    u0 = subdivision_eigenvectors(N)
    u1 = subdivision_eigenvectors_lower_block(N)
    w1 = s12_eigenvectors()

    u0 = [r + [0] * 5 for r in u0]
    lower_block = []
    for i in xrange(0,len(u1)):
      lower_block.append(u1[i] + w1[i])
    for r in lower_block:
      u0.append(r)
    return u0

# Core Matrices

# subdivision_matrix
def subdivision_matrix(N):
    """
    Parameters
    ----------
    N : int
        Valency of vertex of interest.

    Returns
    -------
    P : array_like of shape (N+1, N+1)
        The upper left block of the extended and bigger subdivision matrices.
    """
    alpha = calculate_alpha(N)
    a = 1 - alpha
    b = alpha / N
    c = g.Rational(3, 8)
    d = g.Rational(1, 8)

    S = []
    S.append([a] + [b] * N)
    for i in range(N):
        S.append([c] + [0] * N)
        S[i + 1][i + 1] = c
        S[i + 1][(i + 1) % N + 1] = d
        S[i + 1][(i - 1) % N + 1] = d
    return S

# extended_subdivision_matrix
def extended_subdivision_matrix(N):
    """
    Parameters
    ----------
    N : int
        Valency of vertex of interest.

    Returns
    -------
    P : array_like of shape (N+6, N+6)
        This matrix subdivides the extraordinary component of a matrix of control
        points, yielding a new matrix of identical topology.
    """
    # A = |   s   0 |
    #     | s11 s12 |

    S = subdivision_matrix(N)
    s11 = s11_matrix(N)
    s12 = s12_matrix()

    A = [s + [0] * 5 for s in S]
    lower_block = []
    for i in xrange(0,len(s11)):
      lower_block.append(s11[i] + s12[i])
    for r in lower_block:
      A.append(r)
    return A

def s11_matrix(N):
    a, b, c, d = [g.Rational(i, 16) for i in [1, 2, 6, 10]]
    s11 = [[b, c] + [0] * (N - 2) + [c]]
    s11.append([a, d, a] + [0] * (N - 3) + [a])
    s11.append([b, c, c] + [0] * (N - 2))
    s11.append([a, a] + [0] * (N - 3) + [a, d])
    s11.append([b] + [0] * (N - 2) + [c, c])
    return s11

def s12_matrix():
    a, b = [g.Rational(i, 16) for i in [1, 2]]
    s12 = [[b, 0, 0, 0, 0],
           [a, a, a, 0, 0],
           [0, 0, b, 0, 0],
           [a, 0, 0, a, a],
           [0, 0, 0, 0, b]]
    return s12
    

# bigger_subdivision_matrix
def bigger_subdivision_matrix(N):
    """
    Parameters
    ----------
    N : int
        Valency of vertex of interest.

    Returns
    -------
    P : array_like of shape (N+12, N+6)
        This matrix contains the logic to refine a list of control vertices to
        four patches: 3 with ordinary topology, and one with topology identical
        to the original matrix.
    """
    A_ = extended_subdivision_matrix(N)

    a, b = [g.Rational(i, 8) for i in [1, 3]]
    A_.append([0, b] + [0] * (N - 2) + [a, b, a, 0, 0, 0])
    A_.append([0, b] + [0] * (N - 1) + [a, b, a, 0, 0])
    A_.append([0, b, a] + [0] * (N - 1) + [a, b, 0, 0])
    A_.append([0, a] + [0] * (N - 2) + [b, b, 0, 0, a, 0])
    A_.append([0] * N + [b, a, 0, 0, b, a])
    A_.append([0] * (N - 1) + [a, b, 0, 0, 0, a, b])
    return A_

# picker_matrix
PICKING_INDICES = [
    lambda N: [1, N + 2, N + 3, 2, 0, N, N + 1, N + 6, N + 7, N + 8, N + 9,
               N + 4],
    lambda N: [N + 1, N, N + 4, N + 9, N + 6, N + 2, 1, 0, N-1, N + 5, 2,
               N + 3],
    lambda N: [N, N + 1, 1, 0, N-1, N + 5, N + 4, N + 9, N + 6, N + 2, N + 10,
               N + 11],
    lambda N: range(N + 6),
]
def picker_matrix(N, k):
    """
    Parameters
    ----------
    N : int
        Valency of vertex of interest.

    k : int
        Child index of the patch (0, 1, or 2).

    Returns
    -------
    P : array_like of shape (12, N+12)
        A matrix which "picks" the twelve control points corresponding to the
        ordinary patch at index k.
    """
    M = N + 12
    j = PICKING_INDICES[k](N)
    n = len(j)
    P = [[0] * M for _ in range(n)]
    for i in range(n):
        P[i][j[i]] = 1
    return P

# Basis Functions

# From Stam, 'Evaluation of Loop Subdivision Surfaces'
# http://www.dgp.toronto.edu/~stam/reality/Research/pub.html

# SOURCE_POLYNOMIALS
SOURCE_POLYNOMIALS = '''
u4 + 2u3v,
u4 + 2u3w,
u4 + 2u3w + 6u3v + 6u2vw + 12u2v2 + 6uv2w + 6uv3 + 2v3w + v4,
6u4 + 24u3w + 24u2w2 + 8uw3 + w4 + 24u3v + 60u2vw + 36uvw2 + 6vw3 + 24u2v2 + 36uv2w + 12v2w2 + 8uv3 + 6v3w + v4,
u4 + 6u3w + 12u2w2 + 6uw3 + w4 + 2u3v + 6u2vw + 6uvw2 + 2vw3,
2uv3 + v4,
u4 + 6u3w + 12u2w2 + 6uw3 + w4 + 8u3v + 36u2vw + 36uvw2 + 8vw3 + 24u2v2 + 60uv2w + 24v2w2 + 24uv3 + 24v3w + 6v4,
u4 + 8u3w + 24u2w2 + 24uw3 + 6w4 + 6u3v + 36u2vw + 60uvw2 + 24vw3 + 12u2v2 + 36uv2w + 24v2w2 + 6uv3 + 8v3w + v4,
2uw3 + w4,
2v3w + v4,
2uw3 + w4 + 6uvw2 + 6vw3 + 6uv2w + 12v2w2 + 2uv3 + 6v3w + v4,
w4 + 2vw3'''

# SOURCE_DIVISOR
SOURCE_DIVISOR = 12

# REQUIRED_PERMUTATION
# NOTE `SOURCE_POLYNOMIALS` is intended for labeling from "Figure 1",
# but the labeling in "Figure 2" onwards is what is actually used for
# subdivision
REQUIRED_PERMUTATION = [3, 6, 2, 0, 1, 4, 7, 10, 9, 5, 11, 8]

# parse_source_polynomials_to_sympy
PARSED_SOURCE_POLYNOMIALS = None
def parse_source_polynomials_to_sympy():
    global PARSED_SOURCE_POLYNOMIALS
    if PARSED_SOURCE_POLYNOMIALS is not None:
        return PARSED_SOURCE_POLYNOMIALS

    b_strs = SOURCE_POLYNOMIALS.replace('\n', ' ').split(',')
    assert len(b_strs) == 12

    b_strs = [b_strs[i] for i in REQUIRED_PERMUTATION]

    # NOTE No substitution for `u`, `v` and `w`.
    req_symbs = 'uvw'
    symb_list = map(sp.Symbol, req_symbs)
    symbs = dict(zip(req_symbs, symb_list))
    b = map(partial(parse_source_polynomial_to_sympy, symbs),
            b_strs)

    return symb_list, sp.Matrix(b)

# parse_source_polynomial_to_sympy
def parse_source_polynomial_to_sympy(symbs, b_str):
    term_strs = map(str.strip, b_str.split('+'))
    terms = map(partial(parse_term, symbs), term_strs)
    return reduce(add, terms) / SOURCE_DIVISOR

# parse_term
PARSE_TERM_RE = re.compile(r'([0-9]*)([u]?)([0-9]*)'
                           r'([v]?)([0-9]*)'
                           r'([w]?)([0-9]*)')
def parse_term(symbs, term_str):
    g = PARSE_TERM_RE.search(term_str).groups()
    l = [int(g[0]) if g[0] else 1]

    for i in 1, 3, 5:
        s = symbs.get(g[i], 1)
        p = int(g[i + 1]) if g[i + 1] else 1
        l.append(s ** p)

    return reduce(mul, l)

# `triangle_bspline_basis_uvw`
((u, v, w),
 triangle_bspline_basis_uvw) = parse_source_polynomials_to_sympy()

r, s, t = sp.symbols('r s t')
triangle_bezier_basis_rst = triangle_bspline_basis_uvw.subs(
    {u:r, v:s, w:t}, simultaneous=True)

# Use `u` for the first coordinate and `v` for the second.
# (Stam uses `v` and `w` respectively.)
triangle_bspline_basis_uvw = triangle_bspline_basis_uvw.subs(
    {v : u, w : v, u : w}, simultaneous=True)

# Eliminate `w = 1 - u - v`.
# `triangle_bspline_basis_uv`.
triangle_bspline_basis_uv = triangle_bspline_basis_uvw.subs(
    {w : 1 - u - v}).expand()

# Evaluation Functions

# transform_u_to_subdivided_patch
def transform_u_to_subdivided_patch(u):
    """
    Parameters
    ----------
    u : array_like of shape = (2,)
        The patch coordinate.

    Returns
    -------
    n : int
        Number of required subdivisions.

    k : int
        Child index of the patch (0, 1, or 2).

    u : array_like of shape = (2,)
        The transformed patch coordinate.
    """
    u = np.copy(u)
    n = int(np.floor(1.0 - np.log2(np.sum(u))))
    u *= 2**(n - 1)
    if u[0] > 0.5:
        k = 0
        u[0] = 2 * u[0] - 1
        u[1] = 2 * u[1]
    elif u[1] > 0.5:
        k = 2
        u[0] = 2 * u[0]
        u[1] = 2 * u[1] - 1
    else:
        k = 1
        u[0] = 1 - 2 * u[0]
        u[1] = 1 - 2 * u[1]
    return n, k, u

# recursive_evaluate
def recursive_evaluate(p, b, N, u, X=None):
    """Evaluate the basis vector (or point) at a given patch coordinate.

    Parameters
    ----------
    p : int
        The derivative order of the supplied basis functions.
        `0` should be supplied for position functions,
        `1` for first partial derivatives, `2` for second
        partial derivatives, etc.  This is used to calculate
        an appropriate scaling factor.

    b : function
        The basis function (e.g. `triangle_bspline_position_basis`).

    N : int
        The valency of the extraordinary vertex.

    u : array_like of shape = (2,)
        The patch coordinate.

    X : optional, array_like of shape = (N + 6, dim)
        The (optional) matrix of control vertices which define the geometry
        of the patch.

    Returns
    -------
    r : np.ndarray
        The weight vector if `X = None` else the evaluated point.
    """
    n, k, u = transform_u_to_subdivided_patch(u)
    if N != 6:
        assert n >= 1, 'n < 1 (= %d)' % n

    A_ = bigger_subdivision_matrix(N)
    P3 = picker_matrix(N, 3)
    m = 2.0 ** (p * n) * (-1 if k == 1 else 1) ** p
    x = m * np.dot(b(u).ravel(), picker_matrix(N, k))
    for i in range(n - 1):
        x = np.dot(x, np.dot(A_, P3))
    x = np.dot(x, A_)
    return x if X is None else np.dot(x, X)

# exprs_to_uv_basis
def exprs_to_uv_basis(exprs, func_name=None):
    bs = [sympy_polynomial_to_function(e, (u, v)) for e in exprs]
    def basis_function(U):
        u, v = np.atleast_2d(U).T
        B = np.empty((len(u), len(bs)), dtype=np.float64)
        for i, b in enumerate(bs):
            B[:, i] = b(u, v)
        return B

    if func_name is not None:
        basis_function.func_name = func_name
    return basis_function

def exprs_to_st_basis(exprs, func_name=None):
    bs = [sympy_polynomial_to_function(e, (s, t)) for e in exprs]
    def basis_function(U):
        s, t = np.atleast_2d(U).T
        B = np.empty((len(s), len(bs)), dtype=np.float64)
        for i, b in enumerate(bs):
            B[:, i] = b(s, t)
        return B

    if func_name is not None:
        basis_function.func_name = func_name
    return basis_function

# triangle_bspline_position_basis
triangle_bspline_position_basis = exprs_to_uv_basis(
    triangle_bspline_basis_uv,
    'triangle_bspline_position_basis')

def Du(b): return [sp.diff(f, u) for f in b]
def Dv(b): return [sp.diff(f, v) for f in b]
def Dw(b): return [sp.diff(f, w) for f in b]

def Dr(b): return [sp.diff(f, r) for f in b]
def Ds(b): return [sp.diff(f, s) for f in b]
def Dt(b): return [sp.diff(f, t) for f in b]

# triangle_bspline_du_basis
triangle_bspline_du_basis = exprs_to_uv_basis(
    Du(triangle_bspline_basis_uv),
    'triangle_bspline_du_basis')

# triangle_bspline_dv_basis
triangle_bspline_dv_basis = exprs_to_uv_basis(
    Dv(triangle_bspline_basis_uv),
    'triangle_bspline_dv_basis')

# triangle_bspline_du_du_basis
triangle_bspline_du_du_basis = exprs_to_uv_basis(
    Du(Du(triangle_bspline_basis_uv)),
    'triangle_bspline_du_du_basis')

# triangle_bspline_du_dv_basis
triangle_bspline_du_dv_basis = exprs_to_uv_basis(
    Dv(Du(triangle_bspline_basis_uv)),
    'triangle_bspline_du_dv_basis')

# triangle_bspline_dv_dv_basis
triangle_bspline_dv_dv_basis = exprs_to_uv_basis(
    Dv(Dv(triangle_bspline_basis_uv)),
    'triangle_bspline_dv_dv_basis')

# triangle_bezier_dr_basis
triangle_bezier_dr_basis = exprs_to_st_basis(
    [b.subs({r: 1 - s - t}) for b in Dr(triangle_bezier_basis_rst)],
    'triangle_bezier_dr_basis')

# triangle_bezier_ds_basis
triangle_bezier_ds_basis = exprs_to_st_basis(
    [b.subs({r: 1 - s - t}) for b in Ds(triangle_bezier_basis_rst)],
    'triangle_bezier_ds_basis')

# triangle_bezier_dt_basis
triangle_bezier_dt_basis = exprs_to_st_basis(
    [b.subs({r: 1 - s - t}) for b in Dt(triangle_bezier_basis_rst)],
    'triangle_bezier_dt_basis')

