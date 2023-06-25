import numpy as np

import numpy as np
from lotusvis.spod import spod
import h5py
import os


import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.animation as animation
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"

u = np.load("data/stationary/10k/v.npy")
u = np.einsum("ijk -> kji", u)
v = np.load("data/stationary/10k/v.npy")
v = np.einsum("ijk -> kji", v)
xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nt, ny, nx = v.shape
T = 8  # number of cycles
dt = T / nt
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)

# Reynolds decomposition
u_flucs = np.empty_like(u)
v_flucs = np.empty_like(v)
u_mean = u.mean(axis=2)
v_mean = v.mean(axis=2)
for t in range(nt):
    u_flucs[:,:,t] = u[:, :, t] - u_mean
    v_flucs[:,:,t] = v[:, :, t] - v_mean

# temporal gradients
dudt = np.gradient(u_flucs, dt, axis=2)
dvdt = np.gradient(v_flucs, dt, axis=2)

# spatial gradients
dudx = np.gradient(u_flucs, 4096*np.diff(pxs).mean(), axis=0, edge_order=2)
dudy = np.gradient(u_flucs, 4096*np.diff(pys).mean(), axis=1, edge_order=2)
dvdx = np.gradient(v_flucs, 4096*np.diff(pxs).mean(), axis=0, edge_order=2)
dvdy = np.gradient(v_flucs, 4096*np.diff(pys).mean(), axis=1, edge_order=2)
d2udx2 = np.gradient(dudx, 4096*np.diff(pxs).mean(), axis=0, edge_order=2)
d2vdy2 = np.gradient(dvdy, 4096*np.diff(pys).mean(), axis=1, edge_order=2)

convect = 

# Combine terms to obtain linearized equations −u⋅∇(u)−(u⋅∇)u+Re−1∇u2
L = -u_mean * dudx + dvdt + v_mean * dvdy


import numpy as np

# Example input
u = np.random.rand(2, 1024)  # Example vector u with shape (2, 1024)

# Matrix C
C = np.matrix([[1], [0]])

# Matrix B
B = np.matrix([[1, 0], [0, 0]])

# Matrix L
u_bar = np.mean(u, axis=1)  # Mean of vector u along axis 1
u_bar_bar = np.mean(u_bar)  # Mean of u_bar
grad_squared = np.gradient(u_bar_bar)**2

L = np.array([[-u_bar.dot(grad_squared) - grad_squared.dot(u_bar_bar), -grad_squared.dot(u_bar_bar)],
              [grad_squared.dot(u_bar), 0]])

# Print the matrices
print("Matrix C:")
print(C)
print("Matrix B:")
print(B)
print("Matrix L:")
print(L)
