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
    return data[::2, ::2, :]


u = load_and_process_data("data/stationary/10k/u.npy")
v = load_and_process_data("data/stationary/10k/v.npy")
p = load_and_process_data("data/stationary/10k/p.npy")

xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nx, ny, nt = v.shape

T = 15
dt = T / nt

# Compute fluctuations in a vectorized manner
u_mean = u.mean(axis=2, keepdims=True)
v_mean = v.mean(axis=2, keepdims=True)
p_mean = p.mean(axis=2, keepdims=True)
u_flucs = u - u_mean
v_flucs = v - v_mean
p_flucs = p - p_mean

flat_flucs = np.concatenate([u_flucs, v_flucs, p_flucs], axis=0).reshape(3 * nx * ny, nt)

print("Preprocess done")


def normalise_basis(V: np.ndarray):
    # Normalization with respect to identity matrix Q
    V_norms = np.linalg.norm(V, axis=0)
    V_normalised = V / V_norms
    return V_normalised


def eigen_dual(A: np.ndarray):
    # Compute eigen decomposition of A
    lambda_vals, V = np.linalg.eig(A)
    # Sort eigenvalues and eigenvectors by the real part
    p = np.argsort(-np.real(lambda_vals))
    lambda_vals = lambda_vals[p]
    V = V[:, p]
    
    # Compute eigen decomposition of A.T
    lambda_bar, W = np.linalg.eig(A.T)
    # Sort eigenvalues and eigenvectors by the real part
    p_bar = np.argsort(-np.real(lambda_bar))
    lambda_bar = lambda_bar[p_bar]
    W = W[:, p_bar]
    
    # Normalization using the simplified function
    V = normalise_basis(V)
    W = normalise_basis(W)
    
    # Additional normalization
    for i in range(V.shape[1]):
        V[:, i] = V[:, i] / np.dot(W[:, i], V[:, i])
        
    return lambda_vals, V, W


def DMD_modes(A, U_r, S_r_inv, V_r, tol=1e-6):
    rho, W, Wadj = eigen_dual(A)
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


def compute_SVD(flat_flucs: np.ndarray):
        # Split the data into X and Y
    X = flat_flucs[:, :-1]
    Y = flat_flucs[:, 1:]
    # Compute the truncated SVD
    Ub, sb, Vb = np.linalg.svd(Y, full_matrices=False)
    Uf, sf, Vf = np.linalg.svd(X, full_matrices=False)
    return X,Y,Ub,sb,Vb,Uf,sf,Vf

X,Y,Ub,sb,Vb,Uf,sf,Vf = compute_SVD(flat_flucs)


def compute_A(X,Y,Ub,sb,Vb,Uf,sf,Vf, r=1000):
    r = min(r, X.shape[1])
    # Compute the approximated matrix A_approx usinf fbDMD
    U_r = Ub[:, :r]
    S_r_inv = np.diag(1.0 / sb[:r])
    V_r = Vb.T[:, :r]
    Ab = U_r.T @ Y @ (V_r @ S_r_inv)
    U_r = Uf[:, :r]
    S_r_inv = np.diag(1.0 / sf[:r])
    V_r = Vf.T[:, :r]
    # Compute the approximated matrix A_approx using fbDMD
    Af = U_r.T @ Y @ (V_r @ S_r_inv)
    A_approx = 1/2*(Af + np.linalg.inv(Ab))
    return A_approx


def compute_lambda(A, dt):
    rho, V = np.linalg.eig(A)
    lam = np.log(rho) / dt
    return lam


def opt_gain(lambdas, omega_span, m=4):
    A = np.diag(lambdas)
    R = []
    for omega in omega_span:
        singular_vals = np.linalg.svd(np.linalg.inv(-1j * omega * np.eye(A.shape[0]) - A), compute_uv=False)
        R.append(singular_vals[:m]**2)
    R = np.column_stack(R)
    return R


def opt_forcing(lambdas, omega):
    A = np.diag(lambdas)
    Psi, Sig, Phi = np.linalg.svd(np.linalg.inv(-1j * omega * np.eye(A.shape[0]) - A), full_matrices=False, compute_uv=False)
    return Psi, Sig, Phi


rs = [2]

for r in rs:
    A = compute_A(X,Y,Ub,sb,Vb,Uf,sf,Vf, r=r)

    omegaSpan = np.linspace(1,1000, 2000)
    lambdas = compute_lambda(A, dt)
    gain = opt_gain(lambdas, omegaSpan)

    fig, ax = plt.subplots(figsize = (3,3))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\sigma_i$")
    # ax.set_xlim(0, 10)
    for i in range(min(r, 4)):
        ax.loglog(omegaSpan, np.sqrt(gain[i, :]))
    plt.savefig(f"stationary/figures/opt_gain_DMD_{r}.png", dpi=700)
    plt.close()

    max_gain_om = omegaSpan[np.argmax(np.sqrt(gain))] 
    Psi, Sig, Phi = opt_forcing(lambdas, max_gain_om)


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


