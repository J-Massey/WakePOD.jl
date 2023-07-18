import os
from tqdm import tqdm

import numpy as np
from lotusvis.spod import spod
from scipy.fft import fft, ifft, fftfreq
import scipy.sparse as sp
from src.fft_wrapper import fft_wrapper
import h5py

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.animation as animation

plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"


u = np.load("data/stationary/10k/u.npy")
u = np.einsum("ijk -> kji", u)
u = u[::16, ::16, ::32]
v = np.load("data/stationary/10k/v.npy")
v = np.einsum("ijk -> kji", v)
v = v[::16, ::16, ::32]
p = np.load("data/stationary/10k/p.npy")
p = np.einsum("ijk -> kji", p)
p = p[::16, ::16, ::32]

xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nx, ny, nt = v.shape
T = 7  # number of cycles
# T = 4

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

means = np.array([u_mean, v_mean, p_mean])
flucs_field = np.array([u_flucs, v_flucs, p_flucs])
flat_mean_field = means.reshape(3, nx * ny)
flat_flucs = flucs_field.reshape(3, nx * ny, nt)


def create_grad_operator_y(nx, ny, dy):
    size = nx * ny
    # Diagonal values
    main_diag = np.ones(size) * 0
    off_diag = np.ones(size) * -0.5
    off_diag2 = np.ones(size) * 0.5
    diagonals = [main_diag, off_diag, off_diag2]
    # Offsets for diagonals
    offsets = [0, -1, 1]
    grad_operator = sp.diags(diagonals, offsets, shape=(size, size), format="csr")
    for idx in range(nx):
        grad_operator[idx * (ny), idx * (ny) - 1] = 0
        grad_operator[idx * (ny), idx * (ny)] = -1.5
        grad_operator[idx * (ny), idx * (ny) + 1] = 2
        grad_operator[idx * (ny), idx * (ny) + 2] = -0.5
        grad_operator[-idx * (ny) - 1, -idx * (ny)] = 0
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 1] = 1.5
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 2] = -2
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 3] = 0.5
    return grad_operator / dy


def test_grad_operator_y(nx, ny):
    utestrand = np.random.rand(nx, ny, nt)
    durand = np.gradient(utestrand, 0.1, axis=1, edge_order=2)
    oprand = create_grad_operator_y(nx, ny, 0.1)
    return np.isclose(
        (oprand @ utestrand.reshape(nx * ny, nt)).reshape(nx, ny, nt), durand
    )


# test_grad_operator_y(nx, ny)


def create_grad_operator_x(nx, ny, dx):
    size = nx * ny
    # Diagonal values
    main_diag = np.ones(size) * 0
    off_diag = np.ones(size) * -0.5
    off_diag2 = np.ones(size) * 0.5
    diagonals = [main_diag, off_diag, off_diag2]
    # Offsets for diagonals
    offsets = [0, -ny, ny]
    grad_operator = sp.diags(diagonals, offsets, shape=(size, size), format="csr")
    for idx in range(ny):
        grad_operator[idx] = 0
        grad_operator[idx, idx] = -1.5
        grad_operator[idx, ny + idx] = 2
        grad_operator[idx, 2 * ny + idx] = -0.5
        bc = size - 1
        grad_operator[bc - idx] = 0
        grad_operator[bc - idx, bc - idx] = 1.5
        grad_operator[bc - idx, bc - idx - ny] = -2
        grad_operator[bc - idx, bc - idx - 2 * ny] = 0.5
    return grad_operator / dx


def test_grad_operator_x(nx, ny):
    utestrand = np.random.rand(nx, ny)
    durand = np.gradient(utestrand, axis=0, edge_order=2)
    oprand = create_grad_operator_x(nx, ny, 1)
    return np.allclose((oprand @ utestrand.reshape(nx * ny, 1)).reshape(nx, ny), durand)


# test_grad_operator_x(nx, ny)


def create_laplacian_operator_x(nx, ny, dx=1):
    size = nx * ny
    # Diagonal values
    main_diag = np.ones(size) * -2
    off_diag = np.ones(size) * 1
    off_diag2 = np.ones(size) * 1
    diagonals = [main_diag, off_diag, off_diag2]
    # Offsets for diagonals
    offsets = [0, -ny, ny]
    laplacian_operator = sp.diags(diagonals, offsets, shape=(size, size), format="csr")
    for idx in range(ny):
        laplacian_operator[idx] = 0
        laplacian_operator[idx, idx] = 2
        laplacian_operator[idx, ny + idx] = -5
        laplacian_operator[idx, 2 * ny + idx] = 4
        laplacian_operator[idx, 3 * ny + idx] = -1
        bc = size - 1
        laplacian_operator[bc - idx] = 0
        laplacian_operator[bc - idx, bc - idx] = 2
        laplacian_operator[bc - idx, bc - idx - ny] = -5
        laplacian_operator[bc - idx, bc - idx - 2 * ny] = 4
        laplacian_operator[bc - idx, bc - idx - 3 * ny] = -1
    return laplacian_operator / (dx**2)


def create_laplacian_operator_y(nx, ny, dy):
    size = nx * ny
    # Diagonal values
    main_diag = np.ones(size) * -2
    off_diag = np.ones(size) * 1
    off_diag2 = np.ones(size) * 1
    diagonals = [main_diag, off_diag, off_diag2]
    # Offsets for diagonals
    offsets = [0, -1, 1]
    grad_operator = sp.diags(diagonals, offsets, shape=(size, size), format="csr")
    for idx in range(nx):
        grad_operator[idx * (ny), idx * (ny) - 1] = 0
        grad_operator[idx * (ny), idx * (ny)] = 2
        grad_operator[idx * (ny), idx * (ny) + 1] = -5
        grad_operator[idx * (ny), idx * (ny) + 2] = 4
        grad_operator[idx * (ny), idx * (ny) + 3] = -1
        grad_operator[-idx * (ny) - 1, -idx * (ny)] = 0
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 1] = 2
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 2] = -5
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 3] = 4
        grad_operator[-idx * (ny) - 1, -idx * (ny) - 4] = -1
    return grad_operator / (dy**2)


def test_laplacian_operator_y(nx, ny):
    x = np.linspace(0, 1, nx) ** 3
    dy = np.diff(x).mean()
    x1 = 3 * np.linspace(0, 1, nx) ** 2
    x2 = 6 * np.linspace(0, 1, nx)
    y = np.ones(ny)
    utest = np.outer(x, y)
    du = np.outer(x1, y)
    du2 = np.outer(x2, y)
    op = create_laplacian_operator_y(nx, ny, dy)
    return np.isclose((op @ utest.reshape((nx) * (ny), 1)).reshape(nx, ny), du2)


D1x = create_grad_operator_x(nx, ny, dx)
D1y = create_grad_operator_y(nx, ny, dy)
D2x = create_laplacian_operator_x(nx, ny, dx)
D2y = create_laplacian_operator_y(nx, ny, dy)
LAP = D2x + D2y
I = sp.eye(nx * ny)
Z = np.zeros((nx * ny, nx * ny))
Re = 10250

L1_big = np.concatenate(
    (
        (np.diag(-flat_mean_field[1] @ D1x) + LAP / Re),
        np.diag(-D1x @ flat_mean_field[0]),
        -D1x.toarray(),
    ),
    axis=1,
)

L2_big = np.concatenate(
    (
        np.diag(-D1y @ flat_mean_field[1]),
        (np.diag(-flat_mean_field[0] @ D1y) + LAP / Re),
        -D1y.toarray(),
    ),
    axis=1,
)

L3_big = np.concatenate((D1x.toarray(), D1y.toarray(), Z), axis=1)

L4_big = np.concatenate((D1y.toarray(), D1y.toarray(), Z), axis=1)

L_big = np.concatenate((L1_big, L2_big, L3_big), axis=0)
print("The memory size of numpy array arr is:", L_big.itemsize * L_big.size / 1e9, "GB")

big_flucs = flat_flucs.reshape(3 * nx * ny, nt)#.repeat(3, axis=0)
L_big.shape, big_flucs.shape

Lq = (L_big @ big_flucs)
# Test the product with the discrete operator
# Reshape Lq
Lq.resize(3, nx*ny, nt)
Lq.resize(3, nx, ny, nt)

lim = [-1, 1]
fig, ax = plt.subplots(figsize=(5, 4))
levels = np.linspace(lim[0], lim[1], 44)
_cmap = sns.color_palette("seismic", as_cmap=True)

te_field = Lq[0, :, :, :]

cont = ax.contourf(
    pxs,
    pys,
    te_field[:, :, 6].T,
    levels=levels,
    vmin=lim[0],
    vmax=lim[1],
    # norm=norm,
    cmap=_cmap,
    extend="both",
)

ax.set_aspect(1)
ax.set(xlabel=r"$x$", ylabel=r"$y$")

plt.savefig("stationary/figures/test_u.pdf")
plt.close()

sec = te_field
fig, ax = plt.subplots(figsize=(5, 4))
levels = np.linspace(lim[0], lim[1], 44)

cont = ax.contourf(
    pxs,
    pys,
    sec[:, :, 0].T,
    levels=levels,
    vmin=lim[0],
    vmax=lim[1],
    # norm=norm,
    cmap=_cmap,
    extend="both",
)

ax.set_aspect(1)
ax.set(xlabel=r"$x$", ylabel=r"$y$")  # , title=r"$\phi_{" + str(m) + r"}$")


def animate(i):
    global cont
    for c in cont.collections:
        c.remove()
    cont = plt.contourf(
        pxs,
        pys,
        sec[:, :, i].T,
        levels=levels,
        vmin=lim[0],
        vmax=lim[1],
        # norm=norm,
        cmap=_cmap,
        extend="both",
    )
    return cont.collections


anim = animation.FuncAnimation(
    fig, animate, frames=nt, interval=30, blit=True, repeat=False
)

anim.save(f"stationary/figures/test_u.gif", fps=5, bitrate=-1, dpi=400)


omegaspan = np.linspace(1,20*np.pi)
Sigma =[]
for omega in tqdm(omegaspan):
    H = (1j*omega*np.eye(L_big.shape[0])- L_big)
    S = np.linalg.svd(H, compute_uv=False)
    Sigma.append(S)
    # Phi.append(Ph)
    # Psi.append(Ps)

np.save("data/stationary/10k/Sigmau.npy", np.array(Sigmau))
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
