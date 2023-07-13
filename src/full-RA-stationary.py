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


# u = np.load("data/stationary/10k/u.npy")
# u = np.einsum("ijk -> kji", u)
# u = u[::8, ::8, :250]
u = np.random.rand(20,30,50)
# v = np.load("data/stationary/10k/v.npy")
# v = np.einsum("ijk -> kji", v)
# v = v[::8, ::8, :250]
v = np.random.rand(20,30,50)

xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nx, ny, nt = v.shape
T = 28  # number of cycles
T = 4
dt = T / nt
pxs = np.linspace(*xlims, nx)
dx = np.diff(pxs).mean()
pys = np.linspace(*ylims, ny)
dy = np.diff(pys).mean()

# Construct the linear operator
u_mean = u.mean(axis=2)
v_mean = v.mean(axis=2)
# Flatten the mean fields
flat_u_mean = u_mean.reshape(u_mean.shape[0]*u_mean.shape[1])
flat_v_mean = v_mean.reshape(v_mean.shape[0]*v_mean.shape[1])
N = flat_v_mean.size

print("Loaded fields done")


def create_grad_operator_y(nx, ny, dy):
    size = nx * ny
    # Diagonal values
    main_diag = np.ones(size) * 0
    off_diag = np.ones(size) * -0.5
    off_diag2 = np.ones(size) * 0.5
    diagonals = [main_diag, off_diag, off_diag2]
    # Offsets for diagonals
    offsets = [0, -1, 1]
    grad_operator = sp.diags(diagonals, offsets, shape=(size, size), format='csr')
    for idx in range(nx):
        grad_operator[idx*(ny),idx*(ny)-1] = 0
        grad_operator[idx*(ny),idx*(ny)] = -1.5
        grad_operator[idx*(ny),idx*(ny)+1] = 2
        grad_operator[idx*(ny),idx*(ny)+2] = -0.5
        grad_operator[-idx*(ny)-1, -idx*(ny)] = 0
        grad_operator[-idx*(ny)-1, -idx*(ny)-1] = 1.5
        grad_operator[-idx*(ny)-1, -idx*(ny)-2] = -2
        grad_operator[-idx*(ny)-1, -idx*(ny)-3] = 0.5
    return grad_operator/dy


def test_grad_operator_y(nx, ny):
    y = np.linspace(0, 1, ny)**2
    dy = np.diff(y).mean()
    y1 = 2*np.linspace(0, 1, ny)
    x = np.ones(nx)
    utest = np.outer(x, y).ravel().reshape(nx*ny, 1)
    du = np.outer(x, y1).ravel().reshape(nx*ny, 1)
    op = create_grad_operator_y(nx, ny, dy)

    utestrand = np.random.rand(nx, ny)
    durand = np.gradient(utestrand, axis=1, edge_order=2)
    oprand = create_grad_operator_y(nx, ny, 1)
    return np.allclose((op @ utest), du), np.allclose((oprand @ utestrand.reshape(nx*ny, 1)).reshape(nx, ny), durand)

test_grad_operator_y(nx, ny)


def create_grad_operator_x(nx, ny, dx):
    size = nx * ny
    # Diagonal values
    main_diag = np.ones(size) * 0
    off_diag = np.ones(size) * -0.5
    off_diag2 = np.ones(size) * 0.5
    diagonals = [main_diag, off_diag, off_diag2]
    # Offsets for diagonals
    offsets = [0, -ny, ny]
    grad_operator = sp.diags(diagonals, offsets, shape=(size, size), format='csr')
    for idx in range(ny):
        grad_operator[idx] = 0
        grad_operator[idx,idx] = -1.5
        grad_operator[idx,ny+idx] = 2
        grad_operator[idx,2*ny+idx] = -0.5
        bc = size-1
        grad_operator[bc - idx] = 0
        grad_operator[bc - idx, bc - idx] = 1.5
        grad_operator[bc - idx, bc - idx - ny] = -2
        grad_operator[bc - idx, bc - idx - 2 * ny] = 0.5
    return grad_operator/dx


def test_grad_operator_x(nx, ny):
    utestrand = np.random.rand(nx, ny)
    durand = np.gradient(utestrand, axis=0, edge_order=2)
    oprand = create_grad_operator_x(nx, ny, 1)
    return np.allclose((oprand @ utestrand.reshape(nx*ny, 1)).reshape(nx, ny), durand)

test_grad_operator_x(nx, ny)

def create_laplacian_operator_y(nx, ny, dy=1):
    size = nx * ny
    # Diagonal values
    main_diag = np.ones(size) * -2
    off_diag = np.ones(size) * 1
    off_diag2 = np.ones(size) * 1
    diagonals = [main_diag, off_diag, off_diag2]
    # Offsets for diagonals
    offsets = [0, -1, 1]
    laplacian = sp.diags(diagonals, offsets, shape=(size, size), format='csr')
    for idx in range(nx):
        laplacian[idx,idx-1] = 0
        laplacian[idx,idx] = 2
        laplacian[idx,idx+1] = -5
        laplacian[idx,idx+2] = 4
        laplacian[idx,idx+3] = -1
    
        laplacian[idx-1,idx-4] = -1
        laplacian[idx-1,idx-3] = 4
        laplacian[idx-1,idx-2] = -5
        laplacian[idx-1,idx-1] = 2
        laplacian[idx-1,idx] = 0
    return laplacian/dy**2


def test_laplacian_operator_y(nx, ny):
    x = np.linspace(0, 1, nx)**3
    dy = np.diff(x).mean()
    x1 = 3*np.linspace(0, 1, nx)**2
    x2 = 6*np.linspace(0, 1, nx)
    y = np.ones(ny)
    utest = np.outer(x, y)
    du = np.outer(x1, y)
    du2 = np.outer(x2, y)
    op = create_laplacian_operator_y(nx, ny, dy)
    return np.isclose((op @ utest.reshape((nx)*(ny),1)).reshape(nx,ny), du2)

test_laplacian_operator_y(nx, ny)


def create_laplacian_operator_x(nx, ny, dx=1):
    size = nx * ny
    # Diagonal values
    main_diag = np.ones(size) * -2
    off_diag = np.ones(size) * 1
    off_diag2 = np.ones(size) * 1
    diagonals = [main_diag, off_diag, off_diag2]
    # Offsets for diagonals
    offsets = [0, -ny, ny]
    laplacian_operator = sp.diags(diagonals, offsets, shape=(size, size), format='csr')
    for idx in range(ny):
        laplacian_operator[idx] = 0
        laplacian_operator[idx, idx] = 2
        laplacian_operator[idx, ny+idx] = -5
        laplacian_operator[idx, 2*ny+idx] = 4
        laplacian_operator[idx, 3*ny+idx] = -1
        bc = size-1
        laplacian_operator[bc - idx] = 0
        laplacian_operator[bc - idx, bc - idx] = 2
        laplacian_operator[bc - idx, bc - idx - ny] = -5
        laplacian_operator[bc - idx, bc - idx - 2 * ny] = 4
        laplacian_operator[bc - idx, bc - idx - 3 * ny] = -1
    return laplacian_operator/dx**2

def test_laplacian_operator_x(nx, ny):
    x = np.linspace(0, 1, nx)**3
    dy = np.diff(x).mean()
    x1 = 3*np.linspace(0, 1, nx)**2
    x2 = 6*np.linspace(0, 1, nx)
    y = np.ones(ny)
    utest = np.outer(x, y)
    du = np.outer(x1, y)
    du2 = np.outer(x2, y)
    op = create_laplacian_operator_x(nx, ny, dy)
    return np.isclose((op @ utest.reshape((nx)*(ny),1)).reshape(nx,ny), du2)

test_laplacian_operator_x(nx, ny)
test_laplacian_operator_y(nx, ny)    # This needs testing properly

D1x = create_grad_operator_x(nx, ny, dx)
D1y = create_grad_operator_y(nx, ny, dy)
D2x = create_laplacian_operator_x(nx, ny, dx)
D2y = create_laplacian_operator_y(nx, ny, dy)
I = sp.eye(nx*ny)
Z = np.zeros((nx*ny, nx*ny))
Re = 10250

# Block matrix L representing linearized NS equations
# The last column is the pressure gradient, last row is continuity
L1 = np.array([(np.diag(-flat_u_mean)+D2x/Re), np.diag(-D1x@flat_u_mean), Z])  # u (ax.)
L2 = np.array([Z, (np.diag(-flat_v_mean)+D2y/Re), Z])  # v (rad.)
L4 = np.array([-D1x.toarray(), -D1y.toarray(), Z])  #contin.
L  = np.array([L1, L2, L4])
u = np.linalg.svd(L, compute_uv=False)
print("The memory size of numpy array arr is:",L.itemsize*L.size/1e9,"GB")
L1@[:,:,0]


np.save("data/stationary/10k/L.npy", L)
# L = np.load("data/stationary/10k/L.npy")
dim, nx, ny, fNt = L.shape
Lflat = L.reshape(2, nx*ny, fNt)


# np.save("data/stationary/10k/fft_velocity.npy", fft_flucs)
# Lflatfftu = fft(Lflat[0].T)
# Lflatfftv = fft(Lflat[1].T)

omegaspan = np.linspace(1,20*np.pi)
Sigmau, Sigmav = [], []
for omega in tqdm(omegaspan):
    H = 1/(1j*omega- Lflat[0])
    S = np.linalg.svd(H, compute_uv=False)
    Sigmau.append(S)
    H = 1/(1j*omega- Lflat[1])
    S = np.linalg.svd(H, compute_uv=False)
    Sigmav.append(S)
    # Phi.append(Ph)
    # Psi.append(Ps)

np.save("data/stationary/10k/Sigmau.npy", np.array(Sigmau))
np.save("data/stationary/10k/Sigmav.npy", np.array(Sigmav))
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



