import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"

def Sigma_plot(Sigma):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(Sigma / Sigma.sum(), "o", markerfacecolor="none", ms=1, c='k', alpha=0.8)
    ax.set_xlabel("Mode number")
    ax.set_ylabel(r"$ \Sigma $")
    ax.set_yscale("log")
    plt.savefig("figures/stationary/Sigma.pdf", bbox_inches="tight")
    plt.close()


def lambdaPlot(Lambda):
    # Plot the eigenvalues
    fig, ax = plt.subplots(figsize=(3, 3))
    # theta = np.linspace(0,1,200)*2*np.pi
    # ax.plot(np.cos(theta)/2, np.sin(theta)/2, c='grey')
    ax.plot(np.diag(Lambda).real, np.diag(Lambda).imag, "o", markerfacecolor="none", ms=1, c='k', alpha=0.8)
    ax.set_xlabel(r"$\Re(\lambda)$")
    ax.set_ylabel(r"$\Im(\lambda)$")
    plt.savefig("figures/stationary/lambda.pdf", bbox_inches="tight")
    plt.close()


def omegaPlot(omega):
    # Plot the eigenvalues
    fig, ax = plt.subplots(figsize=(3, 3))
    # theta = np.linspace(0,1,200)*2*np.pi
    # ax.plot(np.cos(theta)/2, np.sin(theta)/2, c='grey')
    ax.plot(np.diag(omega).imag, np.diag(omega).real, "o", markerfacecolor="none", ms=1, c='k', alpha=0.8)
    ax.set_ylabel(r"$\Re(\omega)$")
    ax.set_xlabel(r"$\Im(\omega)$")
    plt.savefig("figures/stationary/omega.pdf", bbox_inches="tight")
    plt.close()


def bPlot(b):
    # Plot the eigenvalues
    fig, ax = plt.subplots(figsize=(3, 3))
    # theta = np.linspace(0,1,200)*2*np.pi
    # ax.plot(np.cos(theta)/2, np.sin(theta)/2, c='grey')
    ax.plot(b, "o", markerfacecolor="none", ms=1, c='k', alpha=0.8)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$b$")
    plt.savefig("figures/stationary/ModeAmplitudes.pdf", bbox_inches="tight")
    plt.close()



xlims, ylims = (1, 2), (-0.35, 0.35)

vort = np.load("data/stationary/100k/data/vort.npy")
vort2 = np.roll(vort, 1, axis=2)

nx, ny, nt = vort.shape
print("Loaded snaps")
# Just the wake
dt = 8 / nt
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)


# Flatten snapshots along first two dimensions
# flucs = np.load("data/fluctuations.npy")
vort1 = vort.reshape(nx*ny,nt)
vort2 = vort2.reshape(nx*ny,nt)

k = 200
# def DMD(X,Xprime,r):
U,Sigma,VT = np.linalg.svd(vort1,full_matrices=False) # Step 1 - Heaviest
Sigma_plot(Sigma)
Ur = U[:,:k]
Sigmar = np.diag(Sigma[:k])
VTr = VT[:k,:]
Atilde = np.linalg.solve(Sigmar.T,(Ur.T @ vort2 @ VTr.T).T).T # Step 2 - Find the linear operator using psuedo inverse
# Now backwards
U,Sigma,VT = np.linalg.svd(vort2,full_matrices=False)
Sigma_plot(Sigma)
Ur = U[:,:k]
Sigmar = np.diag(Sigma[:k])
VTr = VT[:k,:]
Atildeb = np.linalg.solve(Sigmar.T,(Ur.T @ vort1 @ VTr.T).T).T

Atilde = 1/2*(Atilde + np.linalg.inv(Atildeb))

Lambda, W = np.linalg.eig(Atilde) # Step 3 - Eigenvalues
Lambda = np.diag(Lambda)
lambdaPlot(Lambda)
omega = np.log(Lambda)/dt  # Spectral expansion
omegaPlot(omega)


Phi = vort2 @ np.linalg.solve(Sigmar.T,VTr).T @ W # Step 4 - Modes
alpha1 = Sigmar @ VTr[:,0]  # First mode POD
b = np.linalg.solve(W @ Lambda,alpha1)  # The mode amplitudes
bPlot(b)

# Cut off frequency to avoid unresolved modes
freqs = np.diag(omega).imag
mask = (abs(freqs) > -0.1) & (abs(freqs)<2*np.pi*101)
filtered_b = b[mask].real
filtered_Phi = Phi[:,mask]
filtered_frequencies = freqs[mask]
Omega_filtered = np.diag(omega)[mask]

# Consider the system sorted by the real eigenvalues in descending order
omsort = np.flip(np.argsort(Lambda.real))
Phi_omsort = filtered_Phi[:,omsort]
frequencies_omsort = filtered_frequencies[omsort]
Omega_omsort = Omega_filtered[omsort]
modea_omsort = filtered_b[omsort]

# for oms in range(0,20):
#     fig, ax = plt.subplots(figsize=(5, 3))
#     lim = [-.01, .01]
#     levels = np.linspace(lim[0], lim[1], 44)
#     _cmap = sns.color_palette("seismic", as_cmap=True)
#     cs = ax.contourf(
#         pxs,
#         pys,
#         Phi[:,oms].reshape(nx, ny).T,
#         levels=levels,
#         vmin=lim[0],
#         vmax=lim[1],
#         # norm=norm,
#         cmap=_cmap,
#         extend="both",
#     )
#     ax.set_aspect(1)
#     ax.set_title(f"$\omega={frequencies_omsort[oms]:.2f}, f={frequencies_omsort[oms]/(2*np.pi):.2f}$")
#     plt.savefig(f"./figures/stationary/omsort/{oms}.png", dpi=600)
#     plt.close()


# for oms in range(180,200):
#     fig, ax = plt.subplots(figsize=(5, 3))
#     lim = [-.01, .01]
#     levels = np.linspace(lim[0], lim[1], 44)
#     _cmap = sns.color_palette("seismic", as_cmap=True)
#     cs = ax.contourf(
#         pxs,
#         pys,
#         Phi[:,oms].reshape(nx, ny).T,
#         levels=levels,
#         vmin=lim[0],
#         vmax=lim[1],
#         # norm=norm,
#         cmap=_cmap,
#         extend="both",
#     )
#     ax.set_aspect(1)
#     ax.set_title(f"$\omega={frequencies_omsort[oms]:.2f}, f={frequencies_omsort[oms]/(2*np.pi):.2f}$")
#     plt.savefig(f"./figures/stationary/omsort/{oms}.png", dpi=600)
#     plt.close()


# # Sort based on mode amplitude
# bsort = np.flip(np.argsort(abs(filtered_b)))
# Phi_bsort = filtered_Phi[:,bsort]
# frequencies_bsort = filtered_frequencies[bsort]
# Omega_bsort = Omega_filtered[bsort]
# modea_bsort = filtered_b[bsort]


# for oms in range(0,50):
#     fig, ax = plt.subplots(figsize=(5, 3))
#     lim = [-.01, .01]
#     levels = np.linspace(lim[0], lim[1], 44)
#     _cmap = sns.color_palette("seismic", as_cmap=True)
#     cs = ax.contourf(
#         pxs,
#         pys,
#         Phi[:,oms].reshape(nx, ny).T,
#         levels=levels,
#         vmin=lim[0],
#         vmax=lim[1],
#         # norm=norm,
#         cmap=_cmap,
#         extend="both",
#     )
#     ax.set_aspect(1)
#     ax.set_title(f"$\omega={frequencies_bsort[oms]:.2f},b={filtered_b[bsort][oms]:.2f}$")
#     plt.savefig(f"./figures/bsort/{oms}.png", dpi=600)
#     plt.close()










# # Now reconstruct based on these modes

# ts = np.linspace(0.01,0.99,30)
# left = Phi_bsort@np.diag(modea_bsort)
# T_om = np.matrix(np.exp(Omega_bsort)).T @ np.matrix(ts)
# Xt = left@T_om

# for idx,t in enumerate(ts):
#     fig, ax = plt.subplots(figsize=(5, 3))
#     lim = [-.01, .01]
#     levels = np.linspace(lim[0], lim[1], 44)
#     _cmap = sns.color_palette("seismic", as_cmap=True)
#     cs = ax.contourf(
#         pxs,
#         pys,
#         Xt[:,idx].reshape(nx, ny).T,
#         levels=levels,
#         vmin=lim[0],
#         vmax=lim[1],
#         # norm=norm,
#         cmap=_cmap,
#         extend="both",
#     )
#     ax.set_aspect(1)
#     # ax.set_title(f"$\omega={frequencies_bsort[bs]:.2f},b={filtered_b[bsort][bs]:.2f}$")
#     plt.savefig(f"./figures/reconst/{t:.2f}.png", dpi=600)
#     plt.close()