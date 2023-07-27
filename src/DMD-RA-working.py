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

flat_flucs = np.stack([u_flucs, v_flucs, p_flucs], axis=0).reshape(3, nx * ny, nt)

# Define inputs for DMD on the vertical velocity
flat_flucs.resize(3*nx*ny, nt)
fluc1 = flat_flucs[:, :-1]
fluc2 = flat_flucs[:, 1:]

print("Preprocess done")

# def fbDMD(fluc1,fluc2,k):
# backwards
rs = [2,4,6,1000] # input("Enter the number of DMD modes you'd like to retain (e.g., 2): ")
Ub,Sigmab,VTb = svd(fluc2,full_matrices=False)
Uf, Sigmaf, VTf = svd(fluc1, full_matrices=False)

for r in rs:
    # Sigma_plot(Sigma)
    U_r = Ub[:,:r]
    Sigmar = np.diag(Sigmab[:r])
    VT_r = VTb[:r,:]
    Atildeb = np.linalg.solve(Sigmar.T,(U_r.T @ fluc1 @ VT_r.T).T).T

    U_r = Uf[:, :r]
    S_r = np.diag(Sigmaf[:r])
    VT_r = VTf[:r, :]
    Atildef = np.linalg.solve(S_r.T,(U_r.T @ fluc2 @ VT_r.T).T).T # Step 2 - Find the linear operator using psuedo inverse

    A_tilde = 1/2*(Atildef + np.linalg.inv(Atildeb))
    rho, W = np.linalg.eig(A_tilde)

    V_r = np.dot(np.dot(fluc2, VT_r.T), np.dot(np.linalg.inv(S_r), W))

    V_r_star_Q = V_r.conj().T
    V_r_star_Q_V_r = np.dot(V_r_star_Q, V_r)

    # Cholesky factorization
    F_tilde = cholesky(V_r_star_Q_V_r)

    Lambda = np.log(rho)/dt  # Spectral expansion
    omegaSpan = np.linspace(1, 1000, 2000)
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


