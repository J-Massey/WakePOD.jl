import numpy as np
from scipy.linalg import svd, cholesky
from tqdm import tqdm

from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.animation as animation
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"
plt.rcParams["image.cmap"] = "gist_earth"


def plot_flows(qi, fn, _cmap, lim):
    # Test plot
    fig, ax = plt.subplots(figsize=(5, 3))
    levels = np.linspace(lim[0], lim[1], 44)
    _cmap = sns.color_palette(_cmap, as_cmap=True)
    cs = ax.contourf(
        pxs,
        pys,
        qi,
        levels=levels,
        vmin=lim[0],
        vmax=lim[1],
        # norm=norm,
        cmap=_cmap,
        extend="both",
        # alpha=0.7,
    )
    ax.set_aspect(1)
    # ax.set_title(f"$\omega={frequencies_bsort[oms]:.2f},St={frequencies_bsort[oms]/(2*np.pi):.2f}$")
    # ax.plot(pxs, fwarp(pxs), c='k')
    plt.savefig(f"./swimming/figures/{fn}.png", dpi=700)
    plt.close()

case = "swimming"
fp = f"data/{case}/10k"

def load_and_process_data(filepath):
    data = np.load(filepath)
    data = np.einsum("ijk -> kji", data)
    return data[::2, ::2, :]


u = load_and_process_data(f"{fp}/u.npy")
v = load_and_process_data(f"{fp}/v.npy")
p = load_and_process_data(f"{fp}/p.npy")

xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nx, ny, nt = v.shape

T = 14
dt = T / nt
t = np.linspace(0, T, nt)

pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)


def bodify(arr, pxs):
    mask = ((pxs > 0) & (pxs < 1))
    return arr[mask, :, :]


u = bodify(u, pxs)
v = bodify(v, pxs)
p = bodify(p, pxs)

nx, ny, nt = u.shape
xlims, ylims = (0, 1), (-0.35, 0.35)
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)


def fwarp(t: float, pxs: np.ndarray):
    return -0.5*(0.28 * pxs**2 - 0.13 * pxs + 0.05) * np.sin(2*np.pi*(t - (1.42* pxs)))


def unwarp_velocity_field(qi, ts, pxs):
    dy = (- ylims[0] + ylims[1])/ny
    q_warped = np.full(qi.shape, np.nan)
    for idt, t in enumerate(ts):
        # Calculate the shifts for each x-coordinate
        fw = fwarp(t, pxs)
        shifts = np.round(fw/dy).astype(int)

        # Apply the shifts to the velocity field
        for i in range(qi.shape[0]):
            shift = shifts[i]
            if shift >= 0:
                q_warped[i, shift:, idt] = qi[i, :qi.shape[1]-shift, idt]
            else:
                shift = -shift
                q_warped[i, :-shift, idt] = qi[i, shift:, idt]
    
    return q_warped


u_unwarped = unwarp_velocity_field(u, t, pxs)
v_unwarped = unwarp_velocity_field(v, t, pxs)
p_unwarped = unwarp_velocity_field(p, t, pxs)

for idt in range(200):
    plot_flows(v_unwarped[:,:,idt].T, f"warp-gif/{idt}", "seismic", [-0.5, 0.5])


def clipped_field(arr, pys):
    mask = ((pys > -0.25) & (pys < 0.25))
    return arr[:, mask, :]

u_unwarped = clipped_field(u_unwarped, pys)
v_unwarped = clipped_field(v_unwarped, pys)
p_unwarped = clipped_field(p_unwarped, pys)

nx, ny, nt = u_unwarped.shape
xlims, ylims = (0, 1), (-0.25, 0.25)
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)


# Compute fluctuations 
u_mean = u_unwarped.mean(axis=2, keepdims=True)
v_mean = v_unwarped.mean(axis=2, keepdims=True)
p_mean = p_unwarped.mean(axis=2, keepdims=True)
u_flucs = u_unwarped - u_mean
v_flucs = v_unwarped - v_mean
p_flucs = p_unwarped - p_mean

flat_flucs = np.concatenate([u_flucs, v_flucs, p_flucs], axis=0).reshape(3 * nx * ny, nt)

fluc1 = flat_flucs[:, :-1]
fluc2 = flat_flucs[:, 1:]

print("Preprocess done")

Ub,Sigmab,VTb = svd(fluc2,full_matrices=False)
Uf, Sigmaf, VTf = svd(fluc1, full_matrices=False)

print("SVDone")

rs = [2, 4, 10, 20, nt-2] # input("Enter the number of DMD modes you'd like to retain (e.g., 2): ")
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

    # Find the linear operator
    A_tilde = 1/2*(Atildef + np.linalg.inv(Atildeb))
    rho, W = np.linalg.eig(A_tilde)

    # Find the eigenfunction from spectral expansion
    Lambda = np.log(rho)/dt

    # Find the DMD modes
    V_r = np.dot(np.dot(fluc2, VT_r.T), np.dot(np.linalg.inv(S_r), W))

    # Find the hermatian adjoint of the 
    V_r_star_Q = V_r.conj().T
    V_r_star_Q_V_r = np.dot(V_r_star_Q, V_r)
    # Cholesky factorization
    F_tilde = cholesky(V_r_star_Q_V_r)

    omegaSpan = np.linspace(0, 1000, 2000)
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
    plt.savefig(f"{case}/figures/opt_gain_DMD_{r}.png", dpi=700)
    plt.close()


# Find peaks in the gain data
peak_indices, _ = find_peaks(gain[:,0])

# Extract the omega values corresponding to these peaks
peak_omegas = omegaSpan[peak_indices]

for omega in peak_omegas:

    Psi, Sigma, Phi = np.linalg.svd(F_tilde@np.linalg.inv((-1j*omega)*np.eye(Lambda.shape[0])-np.diag(Lambda))@np.linalg.inv(F_tilde))
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
    ax.set(xlabel=r"$x$", ylabel=r"$y$", title=f"$f^*={omega/(2*np.pi):.2f}$")

    plt.savefig(f"{case}/figures/body/forcing_om_{omega/(2*np.pi):.0f}.png", dpi=700)
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
    ax.set(xlabel=r"$x$", ylabel=r"$y$", title=f"$f^*={omega/(2*np.pi):.2f}$")

    plt.savefig(f"{case}/figures/body/response_om_{omega/(2*np.pi):.0f}.png", dpi=700)
    plt.close()
