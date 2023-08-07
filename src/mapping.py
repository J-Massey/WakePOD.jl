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

fp = "data/swimming/10k"


def load_and_process_data(filepath):
    data = np.load(filepath)
    data = np.einsum("ijk -> kji", data)
    return data[::8, ::8, ::16]


u = load_and_process_data(f"{fp}/u.npy")
v = load_and_process_data(f"{fp}/v.npy")
p = load_and_process_data(f"{fp}/p.npy")

xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nx, ny, nt = v.shape

T = 14
dt = T / nt

pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)


def bodify(arr, pxs):
    mask = ((pxs > 0) & (pxs < 1))
    return arr[mask, :, :]


utest = bodify(u, pxs)
utest = utest[:, :, 0]
xlims, ylims = (0, 1), (-0.35, 0.35)
nx, ny, nt = utest.shape
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)


def warp_velocity_field(u, pxs, pys):
    # Calculate the shifts for each x-coordinate
    fwarp = 0.05 * pxs**2
    dy = (- ylims[0] + ylims[1])/ny
    shifts = np.round(fwarp/dy).astype(int)

    # Create a new velocity field filled with NaNs
    new_u_shifted = np.full(u.shape, np.nan)

    # Apply the shifts to the velocity field
    for i in range(u.shape[0]):
        shift = shifts[i]
        if shift >= 0:
            new_u_shifted[i, shift:] = u[i, :u.shape[1]-shift]
        else:
            shift = -shift
            new_u_shifted[i, :-shift] = u[i, shift:]
    
    return new_u_shifted


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
    )
    ax.set_aspect(1)
    # ax.set_title(f"$\omega={frequencies_bsort[oms]:.2f},St={frequencies_bsort[oms]/(2*np.pi):.2f}$")
    plt.savefig(f"./swimming/figures/{fn}.pdf", dpi=600)
    plt.close()

plot_flows(new_u_shifted.T, "warp_test", "icefire_r", [0., 1.4])