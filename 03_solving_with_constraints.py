"""
Here, we incorporate constraints in the optimization process.

Examples of constraints are nonnegative occupation numbers or fixed norm solutions, ...
"""

import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import scipy.optimize as so
from tools import Timer


# construct some site positions
n1 = 5  # the number of sites along one side of a square grid
n = n1 * n1  # the resulting total number of sites

# construct the coordinates of a square grid
x0 = np.linspace(0, 10., n1, dtype=np.float32)
X, Y = np.meshgrid(x0, x0)

# turn the coordinate arrays into n 2-vectors
positions = np.zeros((n, 2), dtype=np.float32)
positions[:, 0] = X.flatten()
positions[:, 1] = Y.flatten()

# construct A as the matrix of pairwise electrostatic potentials
A = np.zeros((n, n), dtype=np.float32)
all_indices = np.arange(n, dtype=int)

for i in range(n):
	others = np.where(all_indices != i)
	diffs = positions[others] - positions[i]  # the difference vectors between site i and all other sites
	A[i, others] = 0.5 / np.sqrt(np.sum(diffs * diffs, axis=1))  # the electrostatic potential

# define the vector of site energies 
s = -1. * np.ones(n, dtype=np.float32)
# add some Gaussian disorder to the site energies
# s += np.random.normal(scale=0.1, size=n)
# add a well in the middle
# well_site = n // 2
# others = np.where(all_indices != well_site)
# diffs = positions[others] - positions[well_site]  # the difference vectors between site i and all other sites
# s[others] -= 2. / np.sqrt(np.sum(diffs * diffs, axis=1))  # the electrostatic potential
# s[well_site] = s[well_site-1] - 0.1

# find x that solves Ax = -s, and thus minimizes x.T A x + x.T s
# we use the low-level LAPACK solver, since that was found to be the fastest in step 2
# lapack_gesv = sl.get_lapack_funcs('gesv', (A, -s))
# _, _, x, success = lapack_gesv(A, -s)
# if success != 0:
# 	print(f"Optimization failed, LAPACK success info {success}.", flush=True)

# a possible way to involve constraints is to use an iterative minimization algorithm
def energy(x):
	return x.T @ A @ x + x.T @ s
def jac(x):
	return A @ x + s
def hess(x):
	return A

m = 4

def discreteness_constraint(x):
	return (x - 0.5)*(x - 0.5) - 0.25
def constr_jac(x):
	return 2 * np.identity(n) * x - 1.
def constr_hess(x, v):
	return 2 * np.identity(n) * v

# linear constraint matrix
C = np.ones((1, n), dtype=np.float32)
# C[1:] = np.identity(n, dtype=np.float32)
# lb = np.zeros(n+1, dtype=np.float32)
# lb[0] = m
# ub = np.ones(n+1, dtype=np.float32)
# ub[0] = m

with Timer('minimizer'):
	res = so.minimize(energy, x0=np.ones_like(s), jac=jac, hess=hess, method='trust-constr',
		              constraints=[so.LinearConstraint(C, m, m),
		                           so.NonlinearConstraint(discreteness_constraint, 0, 1e-8)])
	x = res.x
	# x = np.round(res.x)
print(f"Final energy: {energy(x)}")
print(x)
print(np.sum(x))

# plot the sites, colored with their value of x
plt.figure(figsize=(8, 8))
sc = plt.scatter(positions[:, 0], positions[:, 1], c=x, cmap='viridis')
plt.gca().set_aspect('equal')
plt.colorbar(sc, fraction=0.046, pad=0.035)
plt.show()