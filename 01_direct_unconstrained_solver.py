"""
The purpose of this script is to get to know some linear algebra involved in optimizing quadratic forms.

The example idea is like this: Consider n sites which can host electrical charges. If charges are placed on these sites,
they affect each other via the electrostatic potential V that is proportional to the inverse of their distance d: V = 1/d.
Also each site has a certain intrinsic site energy. Thus, the total energy of the system is the
sum of all charge-charge potential energies and all occupied site energies.

The question is now: What is the configuration of m charges that minimizes the total energy, i.e. the system's ground state?

This question can be reformulated into graph language: Suppose the n sites are the vertices of an undirected graph. We add an edge between
each pair of sites, making the graph _complete_. Each edge has as a weight the potential energy between the two sites it connects.
Also, each site gets assigned its site energy s. We can then define a n-vector f of occupation numbers, whose ith component is 1 if site i
is occupied and 0 otherwise. The total energy can then be written in matrix formulation:

E_tot = f.T A f + f.T s,

where A is the symmetric adjacency matrix whose component a_ij is the weight of the edge that connects sites i and j. This is a general
quadratic form, which is minimized by the f for which the gradient of E_tot vanishes, i.e. 

A f + s = 0.

Below, we implement a program that solves this linear system for f, not posing any contrains on the solution just yet.
This especially means that f is not yet constrained to be 0 or 1, but can be any real number.
"""

import numpy as np
import matplotlib.pyplot as plt


# construct some site positions
n1 = 70  # the number of sites along one side of a square grid
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
x = np.linalg.solve(A, -s)
# this does the same as linalg.solve(), but turns out to be slower, and we don't need A_inv at the moment:
# A_inv = np.linalg.inv(A)
# x = np.dot(A_inv, -s)

# plot the sites, colored with their value of x
plt.figure(figsize=(8, 8))
sc = plt.scatter(positions[:, 0], positions[:, 1], c=x, cmap='viridis', s=10)
plt.gca().set_aspect('equal')
plt.colorbar(sc, fraction=0.046, pad=0.035)
plt.show()