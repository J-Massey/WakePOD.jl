import os
from tqdm import tqdm

import numpy as np
from LNSO import *
import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg
import scipy.linalg as linalg

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"

def load_and_process_data(filepath):
    data = np.load(filepath)
    data = np.einsum("ijk -> kji", data)
    return data[::4, ::4, :]

# Define the slicing pattern as a variable for clarity
slice_pattern = (slice(None, None, 2), slice(None, None, 2), slice(None, None, 4))

u = load_and_process_data("data/stationary/10k/u.npy")
v = load_and_process_data("data/stationary/10k/v.npy")
p = load_and_process_data("data/stationary/10k/p.npy")

xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nx, ny, nt = v.shape
T = 15  # number of cycles

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
flat_mean_field = means.reshape(3, nx * ny)
flucs_field = np.array([u_flucs, v_flucs, p_flucs])
flatflucs = flucs_field.reshape(3, nx * ny, nt)

def ensure_csc_format(matrix):
    """Ensure that the input matrix is in the csc format.
    
    Args:
        matrix (sp.spmatrix): The input sparse matrix.
    
    Returns:
        sp.csc_matrix: Matrix in csc format.
    """
    return matrix.tocsc()

# Create differential operators
D1x = ensure_csc_format(create_grad_operator_x(nx, ny, dx))
D1y = ensure_csc_format(create_grad_operator_y(nx, ny, dy))
D2x = ensure_csc_format(create_laplacian_operator_x(nx, ny, dx))
D2y = ensure_csc_format(create_laplacian_operator_y(nx, ny, dy))
LAP = ensure_csc_format(D2x + D2y)

# Identity and zero matrices
I = ensure_csc_format(sp.eye(nx * ny))
Z = ensure_csc_format(0 * I)

Re = 10250

# Convert other dense matrices or vectors to sparse and ensure csc format
flat_mean_field_0_sparse = ensure_csc_format(sp.diags(flat_mean_field[0]))
flat_mean_field_1_sparse = ensure_csc_format(sp.diags(flat_mean_field[1]))

# Construct the large L matrices
L1_big = ensure_csc_format(sp.hstack(
    [
        (flat_mean_field_1_sparse @ -D1x + LAP / Re),
        -flat_mean_field_0_sparse @ D1x,
        -D1x
    ]
))

L2_big = ensure_csc_format(sp.hstack(
    [
        -flat_mean_field_1_sparse @ D1y,
        (flat_mean_field_0_sparse @ -D1y + LAP / Re),
        -D1y
    ]
))

L3_big = ensure_csc_format(sp.hstack([D1x, D1y, Z]))

# Combine the large L matrices
L_big = ensure_csc_format(sp.vstack([L1_big, L2_big, L3_big]))
sp.save_npz("stationary/10k/L_big.npy", L_big)


def randomised_gain(Lq: sp.spmatrix, omega: float, k=50):
    m = Lq.shape[0]
    I = sp.eye(m)
    H = -1j*omega*I - Lq
    Xi = np.random.randn(m, k)
    dL = sp_linalg.factorized(H)
    Y = dL(Xi)
    # Orthogonalize sketch using QR decomposition
    Q, _ = linalg.qr(Y, mode='economic')
    # Project Resolvent Opt. into reduced-basis
    B = (Q.T).dot(dL(I.toarray()))
    # SVD of Resolvent Opt. Projection
    U, S, V = linalg.svd(B, full_matrices=False)
    # Solve linear system to compute U vectors
    U = dL(V.T)
    Sigma = linalg.svd(U, compute_uv=False)
    return Sigma

# randomised_gain(L_big, 6)


omegaspan = np.linspace(1,20*np.pi, 40)
Sigma =[]
for omega in tqdm(omegaspan):
    S = randomised_gain(L_big, omega)
    Sigma.append(S)

fig, ax = plt.subplots(figsize=(3, 3))
# ax.set_yscale("log")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\sigma_u$")
ax.loglog(omegaspan, np.array(Sigma)[:,0])
ax.loglog(omegaspan, np.array(Sigma)[:,1])
ax.loglog(omegaspan, np.array(Sigma)[:,2])

plt.savefig(f"./stationary/figures/fullRAgain.pdf", dpi=600)
plt.close()



# def randomised_resolvent(Lq, omega, k):
#     m = Lq.shape[0]
#     I = sp.eye(m)
#     H = -1j*omega*I - Lq

#     # 1. Generate a random normal matrix
#     Xi = np.random.randn(m, k)

#     dL = sp_linalg.factorized(-1j * omega * sp.eye(m) - Lq)

#     # Sketch resolvent operator
#     Y = dL(Xi)

#     # Orthogonalize sketch using QR decomposition
#     Q, _ = linalg.qr(Y, mode='economic')

#     # Project Resolvent Opt. into reduced-basis
#     B = (Q.T).dot(dL(np.eye(m)))

#     # SVD of Resolvent Opt. Projection
#     U, S, V = linalg.svd(B, full_matrices=False)

#     # Solve linear system to compute U vectors
#     U = dL(V.T)

#     # U, Sigma, Vt = linalg.svd(U, full_matrices=False)
#     Sigma = linalg.svd(U, compute_uv=False)

#     # Sigma = np.diag(Sigma)
#     # V = V.dot(Vt)

#     return Sigma