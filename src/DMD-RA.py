import numpy as np
from scipy.linalg import svd, cholesky
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.animation as animation
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rcParams["image.cmap"] = "gist_earth"


def load_and_process_data(filepath):
    data = np.load(filepath)
    data = np.einsum("ijk -> kji", data)
    return data[::4, ::4, :]


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

flat_flucs = np.stack([u_flucs, v_flucs, p_flucs], axis=0).reshape(3, nx*ny, nt)

# Define inputs for DMD on the vertical velocity
flat_flucs.resize(3*nx*ny, nt)

print("Preprocess done")

# Helper function: adj()
def adj(A, Q):
    return np.linalg.lstsq(Q, np.dot(A.T, Q), rcond=None)[0]

# Helper function: normalize_basis()
def normalize_basis(V, Q):
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.sqrt(np.dot(np.dot(V[:, i].T, Q), V[:, i]))
    return V

# Helper function: eigen_dual()
def eigen_dual(A, Q, log_sort=False):
    Aadj = adj(A, Q)
    
    if log_sort:
        λ, V = np.linalg.eig(A)
        order = np.argsort(np.imag(np.log(λ)))
        λ = λ[order]
        V = V[:, order]
        
        λ̄, W = np.linalg.eig(Aadj)
        order_adj = np.argsort(-np.imag(np.log(λ̄)))
        λ̄ = λ̄[order_adj]
        W = W[:, order_adj]
    else:
        λ, V = np.linalg.eig(A)
        order = np.argsort(np.imag(λ))
        λ = λ[order]
        V = V[:, order]
        
        λ̄, W = np.linalg.eig(Aadj)
        order_adj = np.argsort(-np.real(λ̄))
        λ̄ = λ̄[order_adj]
        W = W[:, order_adj]
    
    V = normalize_basis(V, Q)
    W = normalize_basis(W, Q)
    
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.dot(np.dot(W[:, i].T, Q), V[:, i])
    
    return λ, V, W

def compute_SVD(flat_flucs):
        # Split the data into X and Y
    X = flat_flucs[:, :-1]
    Y = flat_flucs[:, 1:]
    # Compute the truncated SVD
    Ub, sb, Vb = np.linalg.svd(Y, full_matrices=False)
    Uf, sf, Vf = np.linalg.svd(X, full_matrices=False)
    return X,Y,Ub,sb,Vb,Uf,sf,Vf

X,Y,Ub,sb,Vb,Uf,sf,Vf = compute_SVD(flat_flucs)


def compute_DMD(X,Y,Ub,sb,Vb,Uf,sf,Vf, r=1000, tol=1e-6, dt=1):
    r = min(r, X.shape[1])
    # Compute the approximated matrix A_approx usinf fbDMD
    U_r = Ub[:, :r]
    S_r_inv = np.diag(1.0 / sb[:r])
    V_r = Vb.T[:, :r]
    Ab = U_r.T @ X @ (V_r @ S_r_inv)
    U_r = Uf[:, :r]
    S_r_inv = np.diag(1.0 / sf[:r])
    V_r = Vf.T[:, :r]
    # Compute the approximated matrix A_approx usinf fbDMD
    Af = U_r.T @ Y @ (V_r @ S_r_inv)
    A_approx = 1/2*(Af + np.linalg.inv(Ab))

    # Compute the dual eigenvalues and eigenvectors
    rho, W, Wadj = eigen_dual(A_approx, np.eye(r), True)

    # Compute matrices Psi and Phi
    Psi = Y @ (V_r @ (S_r_inv @ W))
    Phi = U_r @ Wadj

    # Normalize Psi and Phi
    for i in range(r):
        Psi[:, i] /= np.sqrt(np.dot(Psi[:, i].T, Psi[:, i]))
        Phi[:, i] /= np.sqrt(np.dot(Phi[:, i].T, Phi[:, i]))
        Psi[:, i] /= np.dot(Phi[:, i].T, Psi[:, i])

    # Compute vector b
    b = np.linalg.lstsq(Psi, X[:, 0], rcond=None)[0]

    # Filter based on the provided tolerance
    large = np.abs(b) > tol * np.max(np.abs(b))
    Psi = Psi[:, large]
    Phi = Phi[:, large]
    rho = rho[large]
    lambdas = np.log(rho) / dt
    b = b[large]

    return lambdas, Psi, Phi, b


def factorize_weights(Q):
    return np.linalg.cholesky(Q)


def opt_gain_optimized_identity(A, omega_span, m):
    R = []
    for omega in omega_span:
        singular_vals = np.linalg.svd(np.linalg.inv(-1j * omega * np.eye(A.shape[0]) - A), compute_uv=False)
        R.append(singular_vals[:m]**2)
    R = np.column_stack(R)
    return R


def opt_gain_with_lambda_optimized_identity(V, lambdas, omega_span, m=4):
    A_approx = np.diag(lambdas)
    return opt_gain_optimized_identity(A_approx, omega_span, m)
rs = np.arange(1, 202, 2)

for r in rs:
    lambdas, Psi, Phi, b = compute_DMD(X,Y,Ub,sb,Vb,Uf,sf,Vf, r=r, dt=dt)

    omegaSpan = np.linspace(1,1000, 2000)
    gain = opt_gain_with_lambda_optimized_identity(Psi, lambdas, omegaSpan)

    fig, ax = plt.subplots(figsize = (3,3))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\sigma_i$")
    # ax.set_xlim(0, 10)
    for i in range(min(r, 4)):
        ax.loglog(omegaSpan, np.sqrt(gain[i, :]))
    plt.savefig(f"stationary/figures/opt_gain_DMD_{r}.png", dpi=700)
    plt.close()

# max_gain_om = omegaSpan[np.argmax(np.sqrt(gain))] 

# Psi, Sigma, Phi = np.linalg.svd(F_tilde@np.linalg.inv((-1j*max_gain_om)*np.eye(Lambda.shape[0])-np.diag(Lambda))@np.linalg.inv(F_tilde))

# forcing = V_r @ np.linalg.inv(F_tilde)*Sigma
# forcing.resize(3, nx*ny, r)
# forcing.resize(3, nx, ny, r)

# lim = [-1e-5, 1e-5]
# fig, ax = plt.subplots(figsize=(5, 4))
# levels = np.linspace(lim[0], lim[1], 44)
# _cmap = sns.color_palette("seismic", as_cmap=True)

# cont = ax.contourf(pxs, pys, forcing[0, :, :, 1].T,
#                             levels=levels,
#                             vmin=lim[0],
#                             vmax=lim[1],
#                             # norm=norm,
#                             cmap=_cmap,
#                             extend="both",
#                         )

# ax.set_aspect(1)
# ax.set(xlabel=r"$x$", ylabel=r"$y$")

# plt.savefig("stationary/figures/forcing.png", dpi=700)
# plt.close()

# response = V_r @ np.linalg.inv(F_tilde)@Psi
# response.resize(3, nx*ny, r)
# response.resize(3, nx, ny, r)

# lim = [-1e-2, 1e-2]
# fig, ax = plt.subplots(figsize=(5, 4))
# levels = np.linspace(lim[0], lim[1], 44)
# _cmap = sns.color_palette("seismic", as_cmap=True)

# cont = ax.contourf(pxs, pys, response[0, :, :, 1].T,
#                             levels=levels,
#                             vmin=lim[0],
#                             vmax=lim[1],
#                             # norm=norm,
#                             cmap=_cmap,
#                             extend="both",
#                         )

# ax.set_aspect(1)
# ax.set(xlabel=r"$x$", ylabel=r"$y$")

# plt.savefig("stationary/figures/response.png", dpi=700)
# plt.close()


