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

flow = np.load("data/stationary/100k/v.npy")
xlims, ylims = (-0.35, 2), (-0.35, 0.35)
nt, ny, nx = flow.shape
T = 8  # number of cycles
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
    flatflucs[0, :].reshape(ny, nx),
    levels=levels,
    vmin=lim[0],
    vmax=lim[1],
    # norm=norm,
    cmap=_cmap,
    extend="both",
)
ax.set_aspect(1)
# ax.set_title(f"$\omega={frequencies_bsort[oms]:.2f},St={frequencies_bsort[oms]/(2*np.pi):.2f}$")
plt.savefig(f"./stationary/figures/testv100k.pdf", dpi=600)
plt.close()



spod(flatflucs,dt,"stationary/100k",weight='default',nOvlp='default',window='default',method='fast', nDFT=nt/4)

SPOD_LPf  = h5py.File(os.path.join('stationary/100k','SPOD_LPf.h5'),'r') # load data from h5 format

L = SPOD_LPf['L'][:,:]    # modal energy E(f, M)
P = SPOD_LPf['P'][:,:,:]  # mode shape
f = SPOD_LPf['f'][:]      # frequency
SPOD_LPf.close()


hl_idx = 5
fig, ax = plt.subplots(figsize=(5, 3))
        
# loop over each mode
for imode in range(L.shape[1]):
    if imode < hl_idx:  # highlight modes with colors
        ax.loglog(f[0:-1],L[0:-1,imode],label='Mode '+str(imode+1)) # truncate last frequency
    elif imode == L.shape[1]-1:
        ax.loglog(f[0:-1],L[0:-1,imode],color='lightgrey',label='Others')
    else:
        ax.loglog(f[0:-1],L[0:-1,imode],color='lightgrey',label='')

# figure format
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPOD mode energy')
ax.legend(loc='best')

plt.savefig(f"./stationary/figures/100kspectrum.pdf", dpi=600)
plt.close()