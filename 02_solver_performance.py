"""
Here, we compare the performance of a few different implementations of solving the matrix equation introduced in step 1.

Note that A defined as it is here is symmetric, but not (necessarily) positive definite, so Cholesky decomposition can not be used. 
The fastest methods so far were LU decomposition and the high-level functions scipy.linalg.solve and numpy.linalg.solve,
with all three showing rather similar runtimes.

In more detail, numpy.linalg.solve and explicit LU decomposition have very similar runtimes for all sizes of the matrix.
Interestingly, scipy.linalg.solve is about 2 times slower than both of them for small matrices (up to about n=36) and 
a factor of about 1.5 for large arrays (n=400 and more). For sizes in between, scipy.linalg.solve is actually the fastest, 
up to about a factor of 2 for n=100.

For very large matrices (n=5000 and upwards), all three implementations perform very similarly again.

Comparison with directly calling the low-level LAPACK dgesv solver shows that the latter is even faster than all previous methods,
although not by a large margin.
"""

import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
from tools import Timer


# construct some site positions
n1 = 11  # the number of sites along one side of a square grid
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
s = -1 * np.ones(n, dtype=np.float32)
# add some Gaussian disorder to the site energies
# s += np.random.normal(scale=0.4, size=n)

# find x that solves Ax = -s, and thus minimizes x.T A x + x.T s
# with Timer('np.linalg.solve'):
# 	x = np.linalg.solve(A, -s)

# explicit matrix inversion turns out to be slower, and we don't need A_inv at the moment:
# with Timer('Explicit inversion'):
# 	A_inv = np.linalg.inv(A)
# 	x = np.dot(A_inv, -s)  # fun fact: this can be also written as A_inv @ (-s)

# scipy's LU factorization
# with Timer('LU solver'):
# 	LU, piv = sl.lu_factor(A, overwrite_a=False, check_finite=False)
# 	x = sl.lu_solve((LU, piv), -s)

# scipy version of linalg.solve
# with Timer('scipy.linalg.solve'):
# 	x = sl.solve(A, -s, assume_a='sym', overwrite_a=False, overwrite_b=True, check_finite=False)

# low-level LAPACK solver
lapack_gesv = sl.get_lapack_funcs('gesv', (A, -s))
with Timer('Low-level LAPACK solver'):
	_, _, x, _ = lapack_gesv(A, -s)

# LDL decomposition - rather slow and pretty numerically unstable
# with Timer('LDL decomposition'):
# 	lu, d, perm = sl.ldl(A, overwrite_a=True, check_finite=False)
# 	L = lu[perm]
# 	z = sl.solve(L, -s, overwrite_a=True, check_finite=False)
# 	y = sl.solve(d, z, overwrite_a=True, check_finite=False)
# 	x = sl.solve(d @ L.T, y, overwrite_a=True, check_finite=False)

# plot the sites, colored with their value of x
plt.figure(figsize=(8, 8))
sc = plt.scatter(positions[:, 0], positions[:, 1], c=x, cmap='viridis')
plt.gca().set_aspect('equal')
plt.colorbar(sc, fraction=0.046, pad=0.035)
plt.show()