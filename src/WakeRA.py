import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.animation as animation
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"

def Sigma_plot(Sigma):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(Sigma / Sigma.sum(), "o", ms=2)
    ax.set_xlabel("Mode number")
    ax.set_ylabel(r"$ \Sigma $")
    ax.set_yscale("log")
    plt.savefig("figures/Sigma.pdf", bbox_inches="tight")
    plt.close()


def lambdaPlot(Lambda):
    # Plot the eigenvalues
    fig, ax = plt.subplots(figsize=(3, 3))
    # theta = np.linspace(0,1,200)*2*np.pi
    # ax.plot(np.cos(theta)/2, np.sin(theta)/2, c='grey')
    ax.plot((Lambda).real, (Lambda).imag, "P", markerfacecolor="none", ms=4, c='k', alpha=0.8)
    ax.set_xlabel(r"$\Re(\lambda)$")
    ax.set_ylabel(r"$\Im(\lambda)$")
    plt.savefig("figures/lambda.pdf", bbox_inches="tight")
    plt.close()


def LambdaPlot(omega):
    # Plot the eigenvalues
    fig, ax = plt.subplots(figsize=(3, 3))
    # theta = np.linspace(0,1,200)*2*np.pi
    # ax.plot(np.cos(theta)/2, np.sin(theta)/2, c='grey')
    ax.plot(np.diag(omega).imag, np.diag(omega).real, "o", markerfacecolor="none", ms=1, c='k', alpha=0.8)
    ax.set_ylabel(r"$\Re(\omega)$")
    ax.set_xlabel(r"$\Im(\omega)$")
    plt.savefig("figures/omega.pdf", bbox_inches="tight")
    plt.close()


def bPlot(b):
    # Plot the eigenvalues
    fig, ax = plt.subplots(figsize=(3, 3))
    # theta = np.linspace(0,1,200)*2*np.pi
    # ax.plot(np.cos(theta)/2, np.sin(theta)/2, c='grey')
    ax.plot(b, "X", markerfacecolor="none", ms=4, c='k', alpha=0.8)
    # ax.set_xlabel(r"$\Re(\lambda)$")
    # ax.set_ylabel(r"$\Im(\lambda)$")
    plt.savefig("figures/ModeAmplitudes.pdf", bbox_inches="tight")
    plt.close()


flow = np.load("data/wake_vort.npy")
xlims, ylims = (1, 2), (-0.35, 0.35)
nx, ny, nt = flow.shape
T = 8  # number of cycles
dt = T / nt
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)

# Split the data into phases
phases = np.split(flow, T, axis=2)
phase_average = np.array(phases).mean(axis=0)
phase_fluctuations = np.empty((nx, ny, nt))
for t in range(T):
    phase_fluctuations[:, :, t*int(nt/T):(t+1)*int(nt/T)] = phases[t] - phase_average

np.save("data/phase_fluctuations.npy", phase_fluctuations)

# time_flucs = flow - flow.mean(axis=2)[:, :, None]
# np.save("data/time_flucs.npy", time_flucs)
del flow
# time_flucs = np.load("data/time_flucs.npy")


# Define inputs for DMD
flat_flucs = phase_fluctuations.reshape(nx*ny, nt)
fluc1 = flat_flucs
fluc2 = np.roll(flat_flucs, 1, axis=1)


k = 200
# def fbDMD(fluc1,fluc2,k):
# backwards
# U,Sigma,VT = np.linalg.svd(fluc2,full_matrices=False)
# Sigma_plot(Sigma)
# Ur = U[:,:k]
# Sigmar = np.diag(Sigma[:k])
# VTr = VT[:k,:]
# Atildeb = np.linalg.solve(Sigmar.T,(Ur.T @ fluc1 @ VTr.T).T).T
#forwards
U,Sigma,VT = np.linalg.svd(fluc1,full_matrices=False) # Step 1 - SVD, init
Sigma_plot(Sigma)
Ur = U[:,:k]
Sigmar = np.diag(Sigma[:k])
VTr = VT[:k,:]
Atilde = np.linalg.solve(Sigmar.T,(Ur.T @ fluc2 @ VTr.T).T).T # Step 2 - Find the linear operator using psuedo inverse
# Atilde = 1/2*(Atildef + np.linalg.inv(Atildeb))
# I think we're good up to here...

rho, W = np.linalg.eig(Atilde) # Step 3 - Eigenvalues
Wadj = np.conjugate(W).T
rho = np.diag(rho)

Lambda = np.diag(np.log(rho))/dt  # Spectral expansion

Phi = fluc2 @ np.linalg.solve(Sigmar.T,VTr).T @ W # Step 4 - Modes
alpha1 = Sigmar @ VTr[:,0]  # First mode POD
b = np.linalg.solve(W @ rho,alpha1)  # The mode amplitudes

# Let's trim some fat using the mode aplitudes
tol = 1e-10
large = abs(b)>tol*np.max(abs(b))
Phi = Phi[:,large]
Lambda =  Lambda[large]

# Cut off frequency to avoid unresolved modes
# freqs = np.diag(Lambda).imag
# mask = (abs(freqs) > 1) & (abs(freqs)<2*np.pi*95)
# filtered_b = b[mask].real
# filtered_Phi = Phi[:,mask]
# filtered_frequencies = freqs[mask]
# Lambda = np.diag(Lambda)[mask]

# define the resolvant operator
omegaSpan = np.linspace(-2*np.pi*100, 2*np.pi*100, 800)
gain = np.empty((omegaSpan.size, k))
for idx, omega in enumerate(omegaSpan):
    R = np.linalg.svd(np.linalg.inv(-1j*omega*np.eye(Lambda.size)-Lambda),
                      compute_uv=False)
    gain[idx] = R**2

print(gain)
fig, ax = plt.subplots(figsize = (3,3))
ax.set_yscale('log')
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\sigma_j$")
for i in range(0,10):
    ax.plot(omegaSpan, np.sqrt(gain[:, i]))
plt.savefig("figures/opt_gain.pdf")
plt.close()


