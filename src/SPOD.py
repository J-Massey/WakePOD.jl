import numpy as np
from lotusvis.spod import spod
import h5py
import os

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import matplotlib.animation as animation
plt.style.use(["science"])
plt.rcParams["font.size"] = "10.5"

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
    return data[::4, ::4, ::2]


u = load_and_process_data(f"{fp}/u.npy")
v = load_and_process_data(f"{fp}/v.npy")
p = load_and_process_data(f"{fp}/p.npy")

xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nx, ny, nt = v.shape

T = 14
dt = T / nt

pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)

def wakify(arr, pxs):
    mask = pxs > 1
    return arr[mask, :, :]

u = wakify(u, pxs)
v = wakify(v, pxs)
p = wakify(p, pxs)

xlims, ylims = (1, 2), (-0.35, 0.35)
nx, ny, nt = v.shape
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)


# Compute fluctuations
u_mean = u.mean(axis=2, keepdims=True)
v_mean = v.mean(axis=2, keepdims=True)
p_mean = p.mean(axis=2, keepdims=True)
u_flucs = u - u_mean
v_flucs = v - v_mean
p_flucs = p - p_mean

flatflucs = np.concatenate([u_flucs, v_flucs, p_flucs], axis=0).reshape(3 * nx * ny, nt)

print("FFT done")

# Define inputs for DMD on the vertical velocity
flatflucs = flatflucs.T

# def plot_flows(qi, fn, _cmap, lim):
#     # Test plot
#     fig, ax = plt.subplots(figsize=(5, 3))
#     levels = np.linspace(lim[0], lim[1], 44)
#     _cmap = sns.color_palette(_cmap, as_cmap=True)
#     cs = ax.contourf(
#         pxs,
#         pys,
#         qi,
#         levels=levels,
#         vmin=lim[0],
#         vmax=lim[1],
#         # norm=norm,
#         cmap=_cmap,
#         extend="both",
#     )
#     ax.set_aspect(1)
#     # ax.set_title(f"$\omega={frequencies_bsort[oms]:.2f},St={frequencies_bsort[oms]/(2*np.pi):.2f}$")
#     plt.savefig(f"./swimming/figures/{fn}.png", dpi=600)
#     plt.close()

# plot_flows(u_mean.T, "u", "icefire_r", [0, 1.2])
# plot_flows(v_mean.T, "v", "seismic", [-0.5, 0.5])
# plot_flows(p_mean.T, "p", "seismic", [-0.25, 0.25])
# print("Plotted")


spod(flatflucs,dt,"swimming/10k")

SPOD_LPf  = h5py.File(os.path.join('swimming/10k','SPOD_LPf.h5'),'r') # load data from h5 format

L = SPOD_LPf['L'][:,:]    # modal energy E(f, M)
# P = SPOD_LPf['P'][:,:,:]  # mode shape
f = 2*np.pi*SPOD_LPf['f'][:]      # frequency
SPOD_LPf.close()


hl_idx = 3
fig, ax = plt.subplots(figsize=(4, 3))
        
# loop over each mode
for imode in range(L.shape[1]):
    if imode < hl_idx:  # highlight modes with colors
        ax.loglog(f[0:-1],L[0:-1,imode],label='Mode '+str(imode+1)) # truncate last frequency
    elif imode == L.shape[1]-1:
        ax.loglog(f[0:-1],L[0:-1,imode],color='lightgrey',label='Others')
    else:
        ax.loglog(f[0:-1],L[0:-1,imode],color='lightgrey',label='')

# figure format
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$\lambda_i$')
ax.legend(loc='upper right')

plt.savefig(f"./swimming/figures/SPODspectrum.png", dpi=600)
plt.close()