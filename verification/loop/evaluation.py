##########################################
# File: evaluation.py                    #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from subdivision import loop
from common import example_extraordinary_patch

# main
def main():
    parser = argparse.ArgumentParser()
    # Valency of the vertex of interest.
    parser.add_argument('N', nargs='?', type=int, default=5)
    # Number of levels of subdivision.
    parser.add_argument('n', nargs='?', type=int, default=2)
    # Sampling density along each parametric axis.
    parser.add_argument('m', nargs='?', type=int, default=21)
    args = parser.parse_args()

    # Generate example extraordinary patch with an extraordinary vertex of valency `N`.
    X = example_extraordinary_patch(args.N)

    # Visualise `n` "levels" of subdivision.
    A = loop.extended_subdivision_matrix(args.N)
    A_ = loop.bigger_subdivision_matrix(args.N)
    Xs, Xs_ = [X], []
    for i in range(args.n):
        Xi = Xs[-1]
        Xs.append(np.dot(A, Xi))
        Xs_.append(np.dot(A_, Xi))

    colours = cm.Set1(np.linspace(0.0, 1.0, 9, endpoint=True))[:, :3]
    f, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for i, Xi in enumerate(Xs):
        x, y = Xi.T
        c = colours[i % colours.shape[0]]
        ax.plot(x, y, 'o-', c=c)
        if i >= 1:
            x1, y1 = Xs_[i - 1][-6:].T
            ax.plot(x1, y1, 'o-', c=c)
            ax.plot([x[-1], x1[0]], [y[-1], y1[0]], '--', c=c)

    # Evaluate points on the patch using recursive subdivision and the position
    # basis functions.
    t, step = np.linspace(0.0, 1.0, args.m, endpoint=False, retstep=True)
    t += 0.5 * step
    U = np.dstack(np.broadcast_arrays(t[:, np.newaxis], t)).reshape(-1, 2)
    U = U[np.sum(U, axis=1) <= 1.0 - 1e-8]
    P = np.array([loop.recursive_evaluate(
                    0, loop.triangle_bspline_position_basis,
                    args.N, u, X)
                  for u in U])
    x, y = P.T
    ax.plot(x, y, 'k.')
    ax.set_title('N = %d, n = %d, # = %d' % (args.N, args.n, len(P)))

    plt.show()

if __name__ == '__main__':
    main()
