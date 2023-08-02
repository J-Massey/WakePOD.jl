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

pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)

# Compute fluctuations
u_mean = u.mean(axis=2, keepdims=True)
v_mean = v.mean(axis=2, keepdims=True)
p_mean = p.mean(axis=2, keepdims=True)
u_flucs = u - u_mean
v_flucs = v - v_mean
p_flucs = p - p_mean

flat_flucs = np.concatenate([u_flucs, v_flucs, p_flucs], axis=0).reshape(3 * nx * ny, nt)

fluc1 = flat_flucs[:, :-1]
fluc2 = flat_flucs[:, 1:]

print("Preprocess done")

def normalise_basis(W: np.ndarray):
    # Normalization with respect to identity matrix Q
    W_norms = np.linalg.norm(W, axis=0)
    W_normalised = W / W_norms
    return W_normalised

# def fbDMD(fluc1,fluc2,k):
# backwards
rs = [2] # input("Enter the number of DMD modes you'd like to retain (e.g., 2): ")
Ub,Sigmab,VTb = svd(fluc2,full_matrices=False)
Uf, Sigmaf, VTf = svd(fluc1, full_matrices=False)

r=nt
# for r in rs:
# Sigma_plot(Sigma)
U_r = Ub[:,:r]
Sigmar = np.diag(Sigmab[:r])
VT_r = VTb[:r,:]
Atildeb = np.linalg.solve(Sigmar.T,(U_r.T @ fluc1 @ VT_r.T).T).T


U_r = Uf[:, :r]
S_r = np.diag(Sigmaf[:r])
VT_r = VTf[:r, :]
Atildef = np.linalg.solve(S_r.T,(U_r.T @ fluc2 @ VT_r.T).T).T # Step 2 - Find the linear operator using psuedo inverse

# Find the linear operator
A_tilde = 1/2*(Atildef + np.linalg.inv(Atildeb))
rho, W = np.linalg.eig(A_tilde)
W = normalise_basis(W)

# Find the eigenfunction from spectral expansion
Lambda = np.log(rho)/dt

# Find the DMD modes
V_r = np.dot(np.dot(fluc2, VT_r.T), np.dot(np.linalg.inv(S_r), W))

V_r_star_Q = V_r.conj().T
V_r_star_Q_V_r = np.dot(V_r_star_Q, V_r)
# Cholesky factorization
F_tilde = cholesky(V_r_star_Q_V_r)

omegaSpan = np.linspace(1, 500, 1000)
gain = np.empty((omegaSpan.size, Lambda.size))
for idx, omega in tqdm(enumerate(omegaSpan)):
    R = np.linalg.svd(F_tilde@np.linalg.inv((-1j*omega)*np.eye(Lambda.shape[0])-np.diag(Lambda))@np.linalg.inv(F_tilde),
                    compute_uv=False)
    gain[idx] = R**2

fig, ax = plt.subplots(figsize = (3,3))
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\sigma_i$")
# ax.set_xlim(0, 10)
for i in range(0,min(r,4)):
    ax.loglog(omegaSpan, np.sqrt(gain[:, i]))
plt.savefig(f"stationary/figures/opt_gain_DMD_{r}.pdf", dpi=700)
plt.close()

max_gain_om = omegaSpan[np.argmax(np.sqrt(gain[:, 0]))]

Psi, Sigma, Phi = np.linalg.svd(F_tilde@np.linalg.inv((-1j*max_gain_om)*np.eye(Lambda.shape[0])-np.diag(Lambda))@np.linalg.inv(F_tilde))
for i in range(r):
    Psi[:, i] /= np.sqrt(np.dot(Psi[:, i].T, Psi[:, i]))
    Phi[:, i] /= np.sqrt(np.dot(Phi[:, i].T, Phi[:, i]))
    Psi[:, i] /= np.dot(Phi[:, i].T, Psi[:, i])

forcing = (V_r @ np.linalg.inv(F_tilde)@Psi).reshape(3, nx, ny, r)

field = forcing[1, :, :, 0].real
lim = min(abs(field.min()), field.max())
fig, ax = plt.subplots(figsize=(5, 4))
levels = np.linspace(-lim, lim, 44)
_cmap = sns.color_palette("seismic", as_cmap=True)

cont = ax.contourf(pxs, pys, field.T,
                            levels=levels,
                            vmin=-lim,
                            vmax=lim,
                            # norm=norm,
                            cmap=_cmap,
                            extend="both",
                        )

ax.set_aspect(1)
ax.set(xlabel=r"$x$", ylabel=r"$y$")

plt.savefig(f"stationary/figures/forcing{r}.pdf", dpi=700)
plt.close()

response = (V_r @ np.linalg.inv(F_tilde)@Phi.T).reshape(3, nx, ny, r)
field = response[1, :, :, 0].real
lim = min(abs(field.min()), field.max())
fig, ax = plt.subplots(figsize=(5, 4))
levels = np.linspace(-lim, lim, 44)
_cmap = sns.color_palette("seismic", as_cmap=True)

cont = ax.contourf(pxs, pys, field.T,
                            levels=levels,
                            vmin=-lim,
                            vmax=lim,
                            # norm=norm,
                            cmap=_cmap,
                            extend="both",
                        )

ax.set_aspect(1)
ax.set(xlabel=r"$x$", ylabel=r"$y$")

plt.savefig(f"stationary/figures/response{r}.pdf", dpi=700)
plt.close()

