import os
from tqdm import tqdm

import numpy as np
from lotusvis.spod import spod
from scipy.fft import fft, ifft, fftfreq
import scipy.sparse as sp
from src.fft_wrapper import fft_wrapper
import h5py
from fenics import *

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.animation as animation
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"


# u = np.load("data/stationary/10k/u.npy")
# u = np.einsum("ijk -> kji", u)
# u = u[::8, ::8, :250]
u = np.random.rand(200,400,50)
# v = np.load("data/stationary/10k/v.npy")
# v = np.einsum("ijk -> kji", v)
# v = v[::8, ::8, :250]
v = np.random.rand(200,400,50)
# u = np.random.rand(9, 10, 11)
# v = np.random.rand(9, 10, 11)
xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nx, ny, nt = v.shape
T = 28  # number of cycles
T = 4
dt = T / nt
pxs = np.linspace(*xlims, nx)
dx = np.diff(pxs).mean()
pys = np.linspace(*ylims, ny)
dy = np.diff(pys).mean()

domain = Rectangle(Point(xlims[0], ylims[0]), Point(xlims[1], xlims[1]))
mesh = RectangleMesh(domain, nx, ny)
# Define function spaces
P1 = FiniteElement('P', triangle, 1)
element = VectorElement(P1, dim=2)
V = FunctionSpace(mesh, element)

# Construct the linear operator
u_mean = u.mean(axis=2)
v_mean = v.mean(axis=2)
flat_v_mean = v_mean.reshape(v_mean.shape[0]*v_mean.shape[1])
n = flat_v_mean.size

print("Loaded fields done")

# Reynolds decomposition
u_flucs = np.empty_like(u)
v_flucs = np.empty_like(v)
for t in range(nt):
    u_flucs[:,:,t] = u[:, :, t] - u_mean
    v_flucs[:,:,t] = v[:, :, t] - v_mean

print("Flucs done")

means = np.array([u_mean, v_mean])
flucs_field = np.array([u_flucs, v_flucs])
flat_flucs = flucs_field.reshape(2, nx*ny, nt)
# flatfucku = fft_wrapper(flat_flucs[0].T, dt, nDFT=nt/2).mean(axis=2)
# flatfuckv = fft_wrapper(flat_flucs[1].T, dt, nDFT=nt/2).mean(axis=2)
# _, fNt = flatfucku.shape
# fft_flucs = np.array([flatfucku, flatfuckv]).reshape(2, nx, ny, fNt)
mean_field = np.repeat(means.reshape(2, nx, ny, 1), nt, axis=3)

print("FFT done")


def create_laplacian_operator(nx, ny):
    size = nx * ny
    # Diagonal values
    main_diag = np.ones(size) * -4
    off_diag = np.ones(size - 1)
    off_diag[nx-1::nx] = 0  # Set off-diagonal values to 0 for last column in each row
    off_diag2 = np.ones(size - nx)
    diagonals = [main_diag, off_diag, off_diag, off_diag2, off_diag2]
    # Offsets for diagonals
    offsets = [0, -1, 1, -nx, nx]
    laplacian_operator = sp.diags(diagonals, offsets, shape=(size, size), format='csr')
    return laplacian_operator

def create_grad_operator(nx, ny):
    size = nx * ny
    # Diagonal values
    main_diag = np.ones(size) * 0
    off_diag = np.ones(size)*1
    # off_diag[nx-1::nx] = 0  # Set off-diagonal values to 0 for last column in each row
    off_diag2 = np.ones(size) * -1
    diagonals = [main_diag, off_diag, off_diag2]
    # Offsets for diagonals
    offsets = [0, -1, 1]
    grad_operator = sp.diags(diagonals, offsets, shape=(size, size), format='csr')
    # Set corners appropriately
    grad_operator[0, 0] = -1.5
    grad_operator[0, 1] = 2
    grad_operator[0, 2] = -0.5
    grad_operator[size-1, size-1] = 1.5
    grad_operator[size-1, size-2] = -2
    grad_operator[size-1, size-3] = 0.5
    return grad_operator

# Example usage
nx = 5
ny = 4
utest = np.linspace(0,nx-1,nx).reshape(nx,1) @ np.ones(ny).reshape(1,ny)
utest
create_grad_operator(nx, ny).toarray()
op = create_grad_operator(nx, ny).toarray()
op.T@utest.reshape(nx*ny,1)
utest.T.reshape(1,nx*ny)@np.gradient(np.eye(nx*ny), axis=0, edge_order=2)
np.gradient()

np.gradient(utest, axis=0, edge_order=2)
print(laplacian.toarray())


def compute_laplacian(q, dx, dy):
    # Extracting the dimensions
    dim = 2
    
    # Computing the Laplacian
    laplacian = np.empty_like(q)
    
    for d in range(dim):
        laplacian[d] = np.gradient(
            np.gradient(q[d], dx, axis=0, edge_order=2),
            dx, axis=0, edge_order=2
        ) + np.gradient(
            np.gradient(q[d], dy, axis=1, edge_order=2),
            dy, axis=1, edge_order=2
        )
    
    return laplacian

def discrete_laplacian(q, dx, dy):
    # Extracting the dimensions
    dim = 2
    
    # Computing the Laplacian
    laplacian = np.empty_like(q)
    
    for d in range(dim):
        laplacian[d] = np.gradient(
            np.gradient(q[d], dx, axis=0, edge_order=2),
            dx, axis=0, edge_order=2
        ) + np.gradient(
            np.gradient(q[d], dy, axis=1, edge_order=2),
            dy, axis=1, edge_order=2
        )
    
    return laplacian

def grad(q, dx, dy):
    # Extracting the dimensions
    dim = 2
    dd = [dx, dy]
    # Computing the Laplacian
    grad = np.empty_like(q)
    
    for d in range(dim):
        grad[d] = np.gradient(q[d], dd[d], axis=d, edge_order=2)
    
    return grad


def lns_operator(ubar, sketch, Re):
    # Extracting the dimensions
    dim, nt, ny, nx = ubar.shape

    # Compute the dot products
    dot_product1 = -(ubar * grad(sketch, dx, dy)).sum(axis=0)
    dot_product2 = -(sketch * grad(ubar, dx, dy)).sum(axis=0)

    # Compute the Laplacian of ubar as the second derivatives
    laplacian = compute_laplacian(sketch, dx, dy)

    # Combine the terms
    operator = np.empty_like(ubar)
    for d in range(dim):
        operator[d] = dot_product1 + dot_product2 + 1 / Re * laplacian[d]
    return operator

L = lns_operator(mean_field, flucs_field, 10250)


np.save("data/stationary/10k/L.npy", L)
# L = np.load("data/stationary/10k/L.npy")
dim, nx, ny, fNt = L.shape
Lflat = L.reshape(2, nx*ny, fNt)


# np.save("data/stationary/10k/fft_velocity.npy", fft_flucs)
# Lflatfftu = fft(Lflat[0].T)
# Lflatfftv = fft(Lflat[1].T)

omegaspan = np.linspace(1,20*np.pi)
Sigmau, Sigmav = [], []
for omega in tqdm(omegaspan):
    H = 1/(1j*omega- Lflat[0])
    S = np.linalg.svd(H, compute_uv=False)
    Sigmau.append(S)
    H = 1/(1j*omega- Lflat[1])
    S = np.linalg.svd(H, compute_uv=False)
    Sigmav.append(S)
    # Phi.append(Ph)
    # Psi.append(Ps)

np.save("data/stationary/10k/Sigmau.npy", np.array(Sigmau))
np.save("data/stationary/10k/Sigmav.npy", np.array(Sigmav))
omegaspan = np.linspace(0,20*np.pi)
Sigmau = np.load("data/stationary/10k/Sigmau.npy")
Sigmav = np.load("data/stationary/10k/Sigmav.npy")

fig, ax = plt.subplots(figsize=(3, 3))
# ax.set_yscale("log")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\sigma_u$")
ax.loglog(omegaspan, np.array(Sigmav)[:,0])
ax.loglog(omegaspan, np.array(Sigmav)[:,1])
ax.loglog(omegaspan, np.array(Sigmav)[:,2])

plt.savefig(f"./stationary/figures/gainv.pdf", dpi=600)
plt.close()

# # Test plot
# fig, ax = plt.subplots(figsize=(5, 3))
# lim = [-0.5, 0.5]
# levels = np.linspace(lim[0], lim[1], 44)
# _cmap = sns.color_palette("seismic", as_cmap=True)
# cs = ax.contourf(
#     pxs,
#     pys,
#     flatflucs[0, :].reshape(ny, nx),
#     levels=levels,
#     vmin=lim[0],
#     vmax=lim[1],
#     # norm=norm,
#     cmap=_cmap,
#     extend="both",
# )
# ax.set_aspect(1)
# # ax.set_title(f"$\omega={frequencies_bsort[oms]:.2f},St={frequencies_bsort[oms]/(2*np.pi):.2f}$")
# plt.savefig(f"./stationary/figures/testv100k.pdf", dpi=600)
# plt.close()


