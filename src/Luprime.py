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
u = u[::4, ::4, ::4]
v = np.load("data/stationary/10k/v.npy")
v = np.einsum("ijk -> kji", v)
v = v[::4, ::4, ::4]
p = np.load("data/stationary/10k/p.npy")
p = np.einsum("ijk -> kji", p)
p = p[::4, ::4, ::4]
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
mean_field = np.repeat(means.reshape(2, nx, ny, 1), nt, axis=3)

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

Luprime = lns_operator(mean_field, flucs_field, 10250)

np.save("stationary/Luprime.npy", Luprime)

print("Luprime done")
Luprime[0, :,:,0].max()

# Now visualise the two fields


fig, ax = plt.subplots(figsize = (3,3))
lim=[-0.4, 0.4]

fig, ax = plt.subplots(figsize=(5, 4))
levels = np.linspace(lim[0], lim[1], 44)
_cmap = sns.color_palette("seismic", as_cmap=True)

cont = ax.contourf(pxs, pys, p[:, :, -1].T,
                            levels=levels,
                            vmin=lim[0],
                            vmax=lim[1],
                            # norm=norm,
                            cmap=_cmap,
                            extend="both",
                        )

ax.set_aspect(1)
ax.set(xlabel=r"$x$", ylabel=r"$y$")

plt.savefig("stationary/figures/p.pdf")
plt.close()

# Now animate
sec = u[:, :, :]
fig, ax = plt.subplots(figsize=(5, 4))
levels = np.linspace(lim[0], lim[1], 44)

cont = ax.contourf(pxs, pys, sec[:, :, 0].T,
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
    for c in cont.collections:
        c.remove()
    cont = plt.contourf(pxs, pys, sec[:,:,i].T,
                            levels=levels,
                            vmin=lim[0],
                            vmax=lim[1],
                            # norm=norm,
                            cmap=_cmap,
                            extend="both",
                        )
    return cont.collections

anim = animation.FuncAnimation(fig, animate, frames=nt, interval=30, blit=True, repeat=False)

anim.save(f"stationary/figures/u.gif", fps=50, bitrate=-1, dpi=400)