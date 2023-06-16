import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
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
    ax.plot(np.diag(Lambda).real, np.diag(Lambda).imag, "P", markerfacecolor="none", ms=4, c='k', alpha=0.8)
    ax.set_xlabel(r"$\Re(\lambda)$")
    ax.set_ylabel(r"$\Im(\lambda)$")
    plt.savefig("figures/lambda.pdf", bbox_inches="tight")
    plt.close()


def omegaPlot(omega):
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



xlims, ylims = (1, 2), (-0.35, 0.35)

vort = np.load("data/wake_vort.npy")
vort2 = np.roll(vort, 1, axis=2)

nx, ny, nt = vort.shape
print("Loaded snaps")
# Just the wake
dt = 8 / nt
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)


# Flatten snapshots along first two dimensions
vort1 = vort.reshape(nx * ny, nt)
vort2 = vort2.reshape(nx * ny, nt)

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

# Now sort everything based on the real part of the spectral expansion
p = np.argsort(omega.imag)
Atildes = Atilde[p]
Sigmars = Sigmar[p]
Ws = W[p]
VTrs = VTr[p]
Lambdas = Lambda[p]


Phi = vort2 @ np.linalg.solve(Sigmar.T,VTr).T @ W # Step 4 - Modes
alpha1 = Sigmar @ VTr[:,0]  # Time?
b = np.linalg.solve(W @ Lambda,alpha1)  # The mode amplitudes
    # return Phi, Lambda, b
bPlot(b)
print(VTr.shape)


fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(alpha1, "X", markerfacecolor="none", ms=4, c='k', alpha=0.8)
# ax.set_xlabel(r"$\Re(\lambda)$")
# ax.set_ylabel(r"$\Im(\lambda)$")
plt.savefig("figures/freqs.pdf", bbox_inches="tight")
plt.close()


for m in range(0,k):
    fig, ax = plt.subplots(figsize=(3, 3))
    # theta = np.linspace(0,1,200)*2*np.pi
    # ax.plot(np.cos(theta)/2, np.sin(theta)/2, c='grey')
    ax.plot(np.linspace(0,8,nt-1), VTr[m, :], c='k', alpha=0.8)
    ax.set_xlabel(r"$t$")
    # ax.set_ylabel(r"$\Im(\omega)$")
    plt.savefig(f"figures/TimeDynamics/V_{m}.png", bbox_inches="tight")
    plt.close()


print(Lambda.shape)


for m in range(0,k):
    fig, ax = plt.subplots(figsize=(5, 3))
    lim = [-.0075, .0075]
    levels = np.linspace(lim[0], lim[1], 44)
    _cmap = sns.color_palette("seismic", as_cmap=True)
    cs = ax.contourf(
        pxs,
        pys,
        Phi[:,m].reshape(nx, ny).T,
        levels=levels,
        vmin=lim[0],
        vmax=lim[1],
        # norm=norm,
        cmap=_cmap,
        extend="both",
    )
    ax.set_aspect(1)
    # ax.set_title(f"$f^*={frequencies[m]/dt:.2f}$")
    plt.savefig(f"./figures/modes/Phi_{m}.png", dpi=600)
    plt.close()


# print(Vt)
# for m in range(0,10):
#     fig, ax = plt.subplots(figsize=(5, 3))
#     lim = [-.0075, .0075]
#     levels = np.linspace(lim[0], lim[1], 44)
#     _cmap = sns.color_palette("seismic", as_cmap=True)
#     cs = ax.contourf(
#         pxs,
#         pys,
#         U[:,m].reshape(nx, ny).T,
#         levels=levels,
#         vmin=lim[0],
#         vmax=lim[1],
#         # norm=norm,
#         cmap=_cmap,
#         extend="both",
#     )
#     ax.set_aspect(1)
#     plt.savefig(f"./figures/modes/U_{m}.png", dpi=600)
#     plt.close()
