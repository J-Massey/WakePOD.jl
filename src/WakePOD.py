import numpy as np
from scipy.linalg import qr, svd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"

xlims, ylims = (1, 2), (-0.35, 0.35)

vort = np.load("data/wake_vort.npy")

nx, ny, nt = vort.shape
print("Loaded snaps")
# Just the wake
dt = 8 / nt
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)


# Flatten snapshots along first two dimensions
flat_vort = vort.reshape(nx * ny, nt)

U, S, Vt = svd(flat_vort, full_matrices=False)

fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(S / S.sum(), "o", ms=2)
ax.set_xlabel("Mode number")
ax.set_ylabel("Energy fraction")
ax.set_yscale("log")
plt.savefig("figures/wakePOD_energy.pdf", bbox_inches="tight")



for m in range(0,10):
    fig, ax = plt.subplots(figsize=(5, 3))
    lim = [-.0075, .0075]
    levels = np.linspace(lim[0], lim[1], 44)
    _cmap = sns.color_palette("seismic", as_cmap=True)
    cs = ax.contourf(
        pxs,
        pys,
        U[:,m].reshape(nx, ny).T,
        levels=levels,
        vmin=lim[0],
        vmax=lim[1],
        # norm=norm,
        cmap=_cmap,
        extend="both",
    )
    ax.set_aspect(1)
    plt.savefig(f"./figures/modes/U_{m}.png", dpi=600)
    plt.close()


print(Vt)
for m in range(0,10):
    fig, ax = plt.subplots(figsize=(5, 3))
    lim = [-.0075, .0075]
    levels = np.linspace(lim[0], lim[1], 44)
    _cmap = sns.color_palette("seismic", as_cmap=True)
    cs = ax.contourf(
        pxs,
        pys,
        U[:,m].reshape(nx, ny).T,
        levels=levels,
        vmin=lim[0],
        vmax=lim[1],
        # norm=norm,
        cmap=_cmap,
        extend="both",
    )
    ax.set_aspect(1)
    plt.savefig(f"./figures/modes/U_{m}.png", dpi=600)
    plt.close()
