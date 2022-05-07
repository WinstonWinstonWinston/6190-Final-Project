from numpy import empty, interp, searchsorted
from numpy.random import rand

# from functools import reduce
# from operator import mul
# def prod(iterable):
#     return reduce(mul, iterable, 1)
# # In python 3.8 (future), just use from math import prod


def inverse_transform(pdf, x_grid, U=None, ns=100, fast=False):
    n_dim = pdf.ndim
    if U is None:
        U = rand(ns, n_dim)
    else:
        ns = U.shape[0]
    # Calculate the marginal for the 1st dimension:
    marg_x0 = pdf.copy()
    for i in range(1, n_dim):
        shape = (1, -1,) + (1,) * (n_dim - (i + 1))
        Δxi = (x_grid[i][1:] - x_grid[i][:-1]).reshape(shape)
        # trapezoid rule...
        marg_x0 = 0.5 * (Δxi*marg_x0[:,:-1] + Δxi*marg_x0[:,1:]).sum(axis=1)
    # Calculate the cumulative across the 1st dimension:
    Δx0 = x_grid[0][+1:] - x_grid[0][:-1]
    cum_x0 = empty(pdf.shape[0])
    cum_x0[0] = 0
    cum_x0[1:] = 0.5 * (Δx0 * marg_x0[:-1] + Δx0 * marg_x0[1:]).cumsum()
    cum_x0 /= cum_x0[-1]
    # Perform inverse transform sampling on the marginal:
    X = empty((ns, n_dim))
    X[:, 0] = interp(U[:, 0], cum_x0, x_grid[0])
    if n_dim > 1:
        # TODO: Optionally parallelize this loop in dask.
        for i in range(ns):
            # Condition on sample:
            ind = searchsorted(x_grid[0], X[i, 0])
            α = ((X[i, 0]        - x_grid[0][ind-1]) /
                 (x_grid[0][ind] - x_grid[0][ind-1]))  # incorrect when ind==0
            if fast or ind == 0:
                # Nearest neighbor interpolation:
                if α <= 0.5 and ind > 0:
                    cond_pdf = pdf[ind-1]
                else:
                    cond_pdf = pdf[ind]
            else:
                # Linear interpolation:
                cond_pdf = (1 - α) * pdf[ind - 1] + α * pdf[ind]
                # This is the bottleneck for high-dims. with many samples.
            # Recurse:
            X[i, 1:] = inverse_transform(cond_pdf, x_grid[1:], U[i:i+1, 1:])
    return X


if __name__ == '__main__':
    from numpy import (array, linspace, meshgrid, where, nan_to_num,
                       exp, log, sin, cos, pi as π)
    from numpy.linalg import inv, det
    import matplotlib.pyplot as plt

    problem = 2
    n_grid = 200

    if problem == 1:
        def nln_target(y):
            """-ln(p(x)). Provide an interesting two-dimensional distribution."""
            θ = π / 3
            A = array([[cos(θ), -sin(θ)], [sin(θ), cos(θ)]])
            Ainv = inv(A)
            detA = det(A)
            y1_prime = Ainv[0] @ y.T
            x1 = where(y1_prime > 1e-10, log(y1_prime), -23.0)
            x2 = Ainv[1] @ y.T - 8e-3 * y1_prime**2
            arg = 2 * π * 0.6 * 2.0 * detA * y1_prime
            nln_dist = (0.5 * (
                        ((x1 - 3) / 0.6)**2 + ((x2 - 3) / 2)**2) + where(
                arg > 1e-10, log(arg), -23.0))
            return nln_dist

        x1, Δx1 = linspace(-5, 12, n_grid, retstep=True)
        x2, Δx2 = linspace( 0, 65, n_grid+1, retstep=True)

    else:  # problem == 2:
        def nln_target(y_in):
            y = y_in.reshape((-1, 2))
            σ2 = 1.0  # 1.5
            x1 = where(y[:, 0] > 1e-5, log(y[:, 0]), -11)
            x2 = y[:, 1] - (y[:, 0] - 2)**3 - 6
            dist = (0.5 * (((x1 - 0.5) / 0.5)**2 + ((x2 - 0) / σ2)**2) + log(
                2 * π * 0.5 * σ2) + x1)
            return dist

        x1, Δx1 = linspace(1e-4,  4, n_grid, retstep=True)
        x2, Δx2 = linspace(1e-4, 12, n_grid+1, retstep=True)

    # Evaluate the function on a grid:
    X1, X2 = meshgrid(x1, x2, indexing='ij')
    x = array([X1.reshape(-1), X2.reshape(-1)]).T
    nln_p = nln_target(x).reshape((n_grid, n_grid+1))
    p = nan_to_num(exp(-nln_p))
    p /= Δx1 * Δx2 * p.sum()

    plt.figure(figsize=(8, 7))
    col = plt.pcolor(X1, X2, p, cmap='GnBu')
    plt.xlim([x1[0], x1[-1]])
    plt.ylim([x2[1], x2[-1]])
    plt.xlabel('$X_1$', fontsize=18)
    plt.ylabel('$X_2$', fontsize=18)
    cb = plt.colorbar(col)
    cb.set_label('PDF', fontsize=16)

    Xs = inverse_transform(p, (x1, x2), ns=1000)
    plt.scatter(Xs[:, 0], Xs[:, 1], marker='o', color='r', s=2)

    plt.show()
