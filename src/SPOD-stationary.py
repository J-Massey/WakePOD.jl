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

flow = np.load("data/stationary/10k/v.npy")
flow  = flow[::4, :, :]
print("Loaded")
xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nt, ny, nx = flow.shape
T = 28  # number of cycles
dt = T / nt
pxs = np.linspace(*xlims, nx)
pys = np.linspace(*ylims, ny)

# time_flucs = flow - flow.mean(axis=2)[:, :, None]
# np.save("data/stationary/time_flucs.npy", time_flucs)

# flow.shape
# del flow
flatflucs = flow.reshape(nt, nx*ny)

# Test plot
fig, ax = plt.subplots(figsize=(5, 3))
lim = [-0.5, 0.5]
levels = np.linspace(lim[0], lim[1], 44)
_cmap = sns.color_palette("seismic", as_cmap=True)
cs = ax.contourf(
    pxs,
    pys,
    flatflucs.mean(axis=0).reshape(ny, nx),
    levels=levels,
    vmin=lim[0],
    vmax=lim[1],
    # norm=norm,
    cmap=_cmap,
    extend="both",
)
ax.set_aspect(1)
# ax.set_title(f"$\omega={frequencies_bsort[oms]:.2f},St={frequencies_bsort[oms]/(2*np.pi):.2f}$")
plt.savefig(f"./stationary/figures/testv10k.pdf", dpi=600)
plt.close()

print("Plotted")


spod(flatflucs,dt,"stationary/10k",weight='default',nOvlp='default',window='default',method='fast', nDFT=nt/4)

SPOD_LPf  = h5py.File(os.path.join('stationary/10k','SPOD_LPf.h5'),'r') # load data from h5 format

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

plt.savefig(f"./stationary/figures/10kspectrum.pdf", dpi=600)
plt.close()