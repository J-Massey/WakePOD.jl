import numpy as np
from lotusvis.spod import spod
from scipy.fft import fft, ifft, fftfreq
from src.fft_wrapper import fft_wrapper
import h5py
import os
from tqdm import tqdm


import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.animation as animation
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"


u = np.load("data/stationary/10k/u.npy")
u = np.einsum("ijk -> kji", u)
u = u[::2, ::2, :]
v = np.load("data/stationary/10k/v.npy")
v = np.einsum("ijk -> kji", v)
v = v[::2, ::2, :]
# u = np.random.rand(9, 10, 11)
# v = np.random.rand(9, 10, 11)
xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nx, ny, nt = v.shape
T = 16/1.5  # number of cycles
dt = T / nt
pxs = np.linspace(*xlims, nx)
dx = np.diff(pxs).mean()
pys = np.linspace(*ylims, ny)
dy = np.diff(pys).mean()

print("Loaded fields done")

# Reynolds decomposition
u_flucs = np.empty_like(u)
v_flucs = np.empty_like(v)
u_mean = u.mean(axis=2)
v_mean = v.mean(axis=2)
for t in range(nt):
    u_flucs[:,:,t] = u[:, :, t] - u_mean
    v_flucs[:,:,t] = v[:, :, t] - v_mean

print("Flucs done")

means = np.array([u_mean, v_mean])
flucs_field = np.array([u_flucs, v_flucs])
flat_flucs = flucs_field.reshape(2, nx*ny, nt)
mean_field = np.repeat(means.reshape(2, nx, ny, 1), nt, axis=3)

print("FFT done")

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

def grad(q, dx, dy):
    # Extracting the dimensions
    dim = 2
    dd = [dx, dy]
    # Computing the Laplacian
    grad = np.empty_like(q)
    
    for d in range(dim):
        grad[d] = np.gradient(q[d], dd[d], axis=d, edge_order=2)
    
    return grad


def lns_operator(ubar, q, Re):
    # Extracting the dimensions
    dim, nt, ny, nx = q.shape

    # Compute the dot products
    dot_product1 = -(ubar * grad(q, dx, dy)).sum(axis=0)
    dot_product2 = -(q * grad(ubar, dx, dy)).sum(axis=0)
        
    # Compute the Laplacian of u_bar as the second derivatives
    laplacian = compute_laplacian(q, dx, dy)

    # Combine the terms
    operator = np.empty_like(q)
    for d in range(dim):
        operator[d] = dot_product1 + dot_product2\
        + 1/Re * laplacian[d]
    return operator

L = lns_operator(mean_field, flucs_field, 10250)

Laplacian = compute_laplacian(mean_field, dx, dy)
Laplacian_operator = np.linalg.solve(Laplacian, mean_field)

N=20
Adj = np.zeros((N * N, N * N))
Deg = np.diag(sum(Adj, 2))
L = Deg - Adj

np.save("data/stationary/10k/L.npy", L)
# L = np.load("data/stationary/10k/L.npy")

dim, nx, ny, fNt = L.shape
Lflat = L.reshape(2, nx*ny, fNt)

flatfucku = fft_wrapper(flat_flucs[0].T, dt, nDFT=nt/2).mean(axis=2)
flatfuckv = fft_wrapper(flat_flucs[1].T, dt, nDFT=nt/2).mean(axis=2)
_, fNt = flatfucku.shape
fft_flucs = np.array([flatfucku, flatfuckv]).reshape(2, nx, ny, fNt)

np.save("data/stationary/10k/fft_velocity.npy", fft_flucs)
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


