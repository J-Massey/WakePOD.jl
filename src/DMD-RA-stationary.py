import numpy as np
from scipy.linalg import svd, cholesky
from scipy.fft import fft, ifft, fftfreq
import scipy.sparse as sp
# from src.fft_wrapper import fft_wrapper
import h5py
import os
from tqdm import tqdm


import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.animation as animation
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rcParams["image.cmap"] = "gist_earth"


def load_and_process_data(filepath):
    data = np.load(filepath)
    data = np.einsum("ijk -> kji", data)
    return data[::2, ::2, :]

# Define the slicing pattern as a variable for clarity
slice_pattern = (slice(None, None, 2), slice(None, None, 2), slice(None, None, 4))

u = load_and_process_data("data/stationary/10k/u.npy")
v = load_and_process_data("data/stationary/10k/v.npy")
p = load_and_process_data("data/stationary/10k/p.npy")

xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nx, ny, nt = v.shape

T = 7  # number of cycles
dt = T / nt

pxs = np.linspace(*xlims, nx)
dx = np.diff(pxs).mean()

pys = np.linspace(*ylims, ny)
dy = np.diff(pys).mean()

# Reynolds decomposition
u_flucs = np.empty_like(u)
v_flucs = np.empty_like(v)
p_flucs = np.empty_like(p)
u_mean = u.mean(axis=2)
v_mean = v.mean(axis=2)
p_mean = p.mean(axis=2)
for t in range(nt):
    u_flucs[:, :, t] = u[:, :, t] - u_mean
    v_flucs[:, :, t] = v[:, :, t] - v_mean
    p_flucs[:, :, t] = p[:, :, t] - p_mean

print("Flucs done")

# means = np.array([u_mean, v_mean, p_mean])
flucs_field = np.array([u_flucs, v_flucs, p_flucs])
# flat_mean_field = means.reshape(3, nx * ny)
flat_flucs = flucs_field.reshape(3, nx * ny, nt)

print("FFT done")

# Define inputs for DMD on the vertical velocity
flat_flucs.resize(3*nx*ny, nt)
fluc1 = flat_flucs[:, :-1]
fluc2 = flat_flucs[:, 1:]

# def fbDMD(fluc1,fluc2,k):
# backwards
# U,Sigma,VT = np.linalg.svd(fluc2,full_matrices=False)
# # Sigma_plot(Sigma)
# Ur = U[:,:k]
# Sigmar = np.diag(Sigma[:k])
# VTr = VT[:k,:]
# Atildeb = np.linalg.solve(Sigmar.T,(Ur.T @ fluc1 @ VTr.T).T).T
#forwards
# U,Sigma,VT = svd(fluc1,full_matrices=False) # Step 1 - SVD, init
# Sigma_plot(Sigma)
# Ur = U[:,:k]
# Sigmar = np.diag(Sigma[:k])
# VTr = VT[:k,:]
# Atilde = 1/2*(Atildef + np.linalg.inv(Atildeb))
# I think we're good up to here...

U, Sigma, VT = svd(fluc1, full_matrices=False)
fig, ax = plt.subplots(figsize = (3,3))
ax.set_ylabel(r"$\sigma_r/\Sigma \sigma$")
ax.set_xlabel(r"$r$")
ax.scatter(range(Sigma.size), Sigma/np.sum(Sigma), s=2)
plt.savefig("stationary/figures/sigmas.png", dpi=700)
plt.close()
r = 2 # input("Enter the number of DMD modes you'd like to retain (e.g., 2): ")
U_r = U[:, :r]
S_r = np.diag(Sigma[:r])
VT_r = VT[:r, :]

A_tilde = np.linalg.solve(S_r.T,(U_r.T @ fluc2 @ VT_r.T).T).T # Step 2 - Find the linear operator using psuedo inverse
# A_tilde = np.dot(np.dot(np.dot(U_r.T, fluc2), VT_r.T), np.linalg.inv(S_r))
eigvals, W = np.linalg.eig(A_tilde)

Phi = np.dot(np.dot(fluc2, VT_r.T), np.dot(np.linalg.inv(S_r), W))
# Phi = fluc2 @ np.linalg.solve(S_r.T,VT_r).T @ W # Step 4 - Modes

# Q = sp.eye(3 * nx * ny)  # Identity matrix for simplicity, no need for now

# print("The memory size of Q is:", Q.itemsize * Q.size / 1e9, "GB")

V_r_star_Q = Phi.conj().T
V_r_star_Q_V_r = np.dot(V_r_star_Q, Phi)

# Cholesky factorization
F_tilde = cholesky(V_r_star_Q_V_r)

rho, W = np.linalg.eig(A_tilde) # Step 3 - Eigenvalues
# Wadj = np.conjugate(W).T

Lambda = np.log(eigvals)/dt  # Spectral expansion

fig, ax = plt.subplots(figsize = (3,3))
ax.set_xlabel(r"$\Im \lambda_i$")
ax.set_ylabel(r"$\Re \lambda_i$")
ax.scatter(Lambda.imag, Lambda.real, s=2)
plt.savefig("stationary/figures/Lambda.pdf")
plt.close()

omegaSpan = np.linspace(1, 100, 1000)
gain = np.empty((omegaSpan.size, Lambda.size))
for idx, omega in tqdm(enumerate(omegaSpan)):
    R = np.linalg.svd(F_tilde@np.linalg.inv((-1j*omega)*np.eye(Lambda.shape[0])-np.diag(Lambda)@np.linalg.inv(F_tilde)),
                      compute_uv=False)
    gain[idx] = R**2

fig, ax = plt.subplots(figsize = (3,3))
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\sigma_i$")
# ax.set_xlim(0, 10)
for i in range(0,4):
    ax.loglog(omegaSpan, np.sqrt(gain[:, i]))
plt.savefig("stationary/figures/opt_gain_DMD.png", dpi=700)
plt.close()


alpha1 = Sigmar @ VTr[:,0]  # First mode POD
# b = np.linalg.solve(W @ rho,alpha1)  # The mode amplitudes
b = alpha1/(W @ rho)

# Let's trim some fat using the mode aplitudes
tol = 1e-6
large = abs(b)>tol*np.max(abs(b))
Phi = Phi[:,large]
Lambda =  Lambda[large]

# define the resolvant operator

