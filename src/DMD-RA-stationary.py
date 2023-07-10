import numpy as np
from lotusvis.spod import spod
from scipy.fft import fft, ifft, fftfreq
import scipy.sparse as sp
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
u = u[::2, ::2, ::2]
v = np.load("data/stationary/10k/v.npy")
v = np.einsum("ijk -> kji", v)
v = v[::2, ::2, ::2]
xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nx, ny, nt = v.shape
T = 28  # number of cycles
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
# flat_flucs = flucs_field.reshape(2, nx*ny, nt)
# flatfucku = fft_wrapper(flat_flucs[0].T, dt, nDFT=nt/2).mean(axis=2)
# flatfuckv = fft_wrapper(flat_flucs[1].T, dt, nDFT=nt/2).mean(axis=2)
# _, fNt = flatfucku.shape
# fft_flucs = np.array([flatfucku, flatfuckv]).reshape(2, nx, ny, fNt)
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
    
    dot_product1 = -(np.flip(ubar, axis=0) * grad(q, dx, dy)).sum(axis=0)
    dot_product2 = -(np.flip(q, axis=0) * grad(ubar, dx, dy)).sum(axis=0)
        
    # Compute the Laplacian of u_bar as the second derivatives
    laplacian = compute_laplacian(q, dx, dy)
    operator = np.empty_like(q)
    for d in range(dim):
        operator[d] = dot_product1 + dot_product2\
        + 1/Re * laplacian[d]
    return operator

L = lns_operator(mean_field, flucs_field, 10250)

# Define inputs for DMD on the vertical velocity
flat_flucs = L[1].reshape(nx*ny, nt)
fluc1 = flat_flucs[:, :-1]
fluc2 = flat_flucs[:, 1:]

k = 100
# def fbDMD(fluc1,fluc2,k):
# backwards
U,Sigma,VT = np.linalg.svd(fluc2,full_matrices=False)
# Sigma_plot(Sigma)
Ur = U[:,:k]
Sigmar = np.diag(Sigma[:k])
VTr = VT[:k,:]
Atildeb = np.linalg.solve(Sigmar.T,(Ur.T @ fluc1 @ VTr.T).T).T
#forwards
U,Sigma,VT = np.linalg.svd(fluc1,full_matrices=False) # Step 1 - SVD, init
# Sigma_plot(Sigma)
Ur = U[:,:k]
Sigmar = np.diag(Sigma[:k])
VTr = VT[:k,:]
Atildef = np.linalg.solve(Sigmar.T,(Ur.T @ fluc2 @ VTr.T).T).T # Step 2 - Find the linear operator using psuedo inverse
Atilde = 1/2*(Atildef + np.linalg.inv(Atildeb))
# I think we're good up to here...

rho, W = np.linalg.eig(Atilde) # Step 3 - Eigenvalues
# Wadj = np.conjugate(W).T

Lambda = np.log(rho)/dt  # Spectral expansion

# Phi = fluc2 @ np.linalg.solve(Sigmar.T,VTr).T @ W # Step 4 - Modes
# alpha1 = Sigmar @ VTr[:,0]  # First mode POD
# b = np.linalg.solve(W @ rho,alpha1)  # The mode amplitudes

# # Let's trim some fat using the mode aplitudes
# tol = 1e-10
# large = abs(b)>tol*np.max(abs(b))
# Phi = Phi[:,large]
# Lambda =  Lambda[large]

# define the resolvant operator
omegaSpan = np.linspace(0, 2, 100)
gain = np.empty((omegaSpan.size, k))
for idx, omega in enumerate(omegaSpan):
    R = np.linalg.svd(np.linalg.inv(-1j*omega*np.eye(k)-(Atilde)),
                      compute_uv=False)
    gain[idx] = R**2


fig, ax = plt.subplots(figsize = (3,3))
ax.set_yscale('log')
ax.set_xlabel(r"$f^*$")
ax.set_ylabel(r"$\sigma_j$")
# ax.set_xlim(0, 10)
for i in range(0,4):
    ax.plot(omegaSpan, np.sqrt(gain[:, i]))

plt.savefig("figures/opt_gain_DMD.pdf")
plt.close()