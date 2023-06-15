import numpy as np
from scipy.linalg import qr, svd
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.animation as animation

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"

k=20
xlims, ylims = (1, 2), (-0.35, 0.35)

vort = np.load("data/wake_vort.npy")

nx, ny, nt = vort.shape
print("Loaded snaps")
# Just the wake
dt = 8 / nt
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)

flat_vort = vort.reshape(nx * ny, nt)
## Perform PCA on a data set:
pca = PCA(n_components=k)
pca.fit(np.transpose(flat_vort))
scores = pca.transform(np.transpose(flat_vort))
u_R_PCA = np.transpose(pca.inverse_transform(scores))

print(u_R_PCA.shape)


sec = flat_vort[:,:200]
# Create a figure and axes for the contour plot
fig, ax = plt.subplots(figsize=(5, 4))

lim = [-.05, .05]
levels = np.linspace(lim[0], lim[1], 44)
_cmap = sns.color_palette("seismic", as_cmap=True)

# Create a contour plot for the exact solution
# exact_contour = ax.contourf(pxs, pys, flat_vort.reshape(nx,ny,nt)[:,:,0].T, cmap='viridis')

# Create a contour plot for the first PC approximation
cont = ax.contourf(pxs, pys, sec.reshape(nx,ny,200)[:,:,50].T,
                             levels=levels,
                             vmin=lim[0],
                             vmax=lim[1],
                             # norm=norm,
                             cmap=_cmap,
                             extend="both",
                            )

# Add a colorbar
# cbar = plt.colorbar(approx_contour)
ax.set_aspect(1)
ax.set(xlabel=r"$x$", ylabel=r"$y$")#, title=r"$\phi_{" + str(m) + r"}$")
# fig.colorbar(co)
# fig.savefig()


def animate(i):
    global cont
    z = sec.reshape(nx,ny,200)[:,:,i].T
    for c in cont.collections:
        c.remove()
    cont = plt.contourf(pxs, pys, z,
                             levels=levels,
                             vmin=lim[0],
                             vmax=lim[1],
                             # norm=norm,
                             cmap=_cmap,
                             extend="both",
                            )
    return cont.collections

anim = animation.FuncAnimation(fig, animate, frames=200, interval=200, blit=True, repeat=False)

anim.save("./figures/wake_anim.gif", fps=100, bitrate=-1)


sec = u_R_PCA[:,:200]
# Create a figure and axes for the contour plot
fig, ax = plt.subplots(figsize=(5, 4))

lim = [-.05, .05]
levels = np.linspace(lim[0], lim[1], 44)
_cmap = sns.color_palette("seismic", as_cmap=True)

cont = ax.contourf(pxs, pys, sec.reshape(nx,ny,200)[:,:,50].T,
                             levels=levels,
                             vmin=lim[0],
                             vmax=lim[1],
                             # norm=norm,
                             cmap=_cmap,
                             extend="both",
                            )

ax.set_aspect(1)
ax.set(xlabel=r"$x$", ylabel=r"$y$")#, title=r"$\phi_{" + str(m) + r"}$")


def animate(i):
    global cont
    z = sec.reshape(nx,ny,200)[:,:,i].T
    for c in cont.collections:
        c.remove()
    cont = plt.contourf(pxs, pys, z,
                             levels=levels,
                             vmin=lim[0],
                             vmax=lim[1],
                             # norm=norm,
                             cmap=_cmap,
                             extend="both",
                            )
    return cont.collections

anim = animation.FuncAnimation(fig, animate, frames=200, interval=200, blit=True, repeat=False)

anim.save("./figures/first_mode_pca.gif", fps=100, bitrate=-1)
