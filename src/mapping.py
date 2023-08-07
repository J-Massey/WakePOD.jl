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


fp = "data/swimming/10k"

def load_and_process_data(filepath):
    data = np.load(filepath)
    data = np.einsum("ijk -> kji", data)
    return data[::4, ::4, ::25]


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


utest = bodify(u, pxs)
nx, ny, nt = utest.shape
xlims, ylims = (0, 1), (-0.35, 0.35)
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)


def fwarp(t: float, pxs: np.ndarray):
    return -0.5*(0.28 * pxs**2 - 0.13 * pxs + 0.05) * np.sin(2*np.pi*(t - (1.42* pxs)))


def warp_velocity_field(u, ts, pxs):
    dy = (- ylims[0] + ylims[1])/ny
    u_warped = np.full(u.shape, np.nan)
    for idt, t in enumerate(ts):
        # Calculate the shifts for each x-coordinate
        fw = fwarp(t, pxs)
        shifts = np.round(fw/dy).astype(int)

        # Apply the shifts to the velocity field
        for i in range(u.shape[0]):
            shift = shifts[i]
            if shift >= 0:
                u_warped[i, shift:, idt] = u[i, :u.shape[1]-shift, idt]
            else:
                shift = -shift
                u_warped[i, :-shift, idt] = u[i, shift:, idt]
    
    return u_warped


new_u_shifted = warp_velocity_field(utest, t, pxs)

for idt in range(nt):
    plot_flows(new_u_shifted[:,:,idt].T, f"warp-gif/{idt}", "icefire_r", [0., 1.4])
