import numpy as np
from scipy.linalg import qr, svd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"

xlims, ylims = (1, 2), (-0.35, 0.35)

vort = np.load("data/wake_vort_quarter.npy")

nx, ny, nt = vort.shape
print("Loaded snaps")
# Just the wake
dt = 8 / 4 / nt
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)


# Flatten snapshots along first two dimensions
flat_vort = vort.reshape(nx * ny, nt)

def rSVD(A, k):
    m, n = A.shape
    Φ = np.random.rand(n, k)
    Ar = A @ Φ
    Q, _ = qr(Ar, mode='economic')
    B = Q.T @ A
    U, S, Vt = svd(B, full_matrices=False)
    max_idx = np.argsort(S)[::-1][:k]
    Sk = S[max_idx]
    Uk = U[:, max_idx]
    Vk = Vt[max_idx, :]
    U = Q @ Uk
    return U, Sk, Vk


k = 20
U, Σ, Vh = svd(flat_vort, full_matrices=False)

for m in range(k):
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 1, 1)
    ϕ = U[:, m].reshape(nx, ny)
    print(np.min(ϕ), np.max(ϕ))
    co = ax.contourf(
        pxs, pys, ϕ,
        levels=np.linspace(-0.01, 0.01, num=44),
        extend='both'
    )
    ax.set(xlabel=r"$x$", ylabel=r"$y$", title=r"$\phi_{}$".format(m))
    fig.colorbar(co)
    plt.tight_layout()
    fig.savefig("figures/modes/U_{}.png".format(m))

A = np.diag(Σ[:k]) @ Vh[:k, :]
M1 = U[:, :1] @ A[:1, :]

# Animate the modes
for m in range(k):
    Mk = U[:, m:m+1] @ A[m:m+1, :]
    for t in range(nt):
        fig = plt.figure(figsize=(16, 4))
        ax = fig.add_subplot(1, 1, 1)
        co = ax.contourf(
            pxs, pys, Mk[:, t].reshape(nx, ny),
            levels=np.linspace(-0.01, 0.01, num=44),
            cmap='seismic',
            extend='both'
        )
        ax.set(xlabel=r"$x$", ylabel=r"$y$", title=r"$\phi_1(t)$")
        fig.colorbar(co)
        plt.tight_layout()
        fig.savefig("figures/time/M{}_{}.png".format(m, t))

# # Now let's get the DMD
Phi = U @ np.sqrt(np.diag(Σ)) @ Vh  # DMD modes
Lambda = np.sqrt(np.diag(Σ)) @ Vh  # Temporal dynamics
