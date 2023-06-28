"""
SPOD Python toolkit Ver 1.2

The script was originally written for analyzing compressor tip leakage flows:
    
  He, X., Fang, Z., Vahdati, M. & Rigas, G., (2021). Spectral Proper Orthogonal 
  Decomposition of Compressor Tip Leakage Flow. Physics of Fluids, 33(10).
  
An explict reference to the work above is highly appreciated if this script is
useful for your research.  

Xiao He (xiao.he2014@imperial.ac.uk), Zhou Fang
Last update: 24-Sep-2021
"""

# -------------------------------------------------------------------------
# import libraries
import numpy as np
from scipy.fft import fft, ifft
from scipy.special import gammaincinv
import time
import os
import psutil
import h5py
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# main function
def fft_wrapper(x,dt,nOvlp='default',nDFT='default',window='default'):
    '''
    Purpose: main function of spectral proper orthogonal decomposition
             (Towne, A., Schmidt, O., and Colonius, T., 2018, arXiv:1708.04393v2)
        
    Parameters
    ----------
    x         : 2D numpy array, float; space-time flow field data.
                nrow = number of time snapshots;
                ncol = number of grid point * number of variable
    dt        : float; time step between adjacent snapshots
    save_path : str; path to save the output data
    weight    : 1D numpy array; weight function (default unity)
                n = number of grid point * number of variable
    nOvlp     : int; number of overlap (default 50% nDFT)
    nDFT      : int; number of DFT points (default about 10% of number of time snapshots)    
    window    : 1D numpy array, float; window function values (default Henning)
                n = nDFT (default nDFT calculated from number of time snapshots)
    method    : string; 'fast' for fastest speed, 'lowRAM' for lowest RAM usage


    Return
    -------
    The SPOD results are written in the file '<save_path>/SPOD_LPf.h5'
    SPOD_LPf['L'] : 2D numpy array, float; modal energy E(f, M)
                    nrow = number of frequencies
                    ncol = number of modes (ranked in descending order by modal energy)
    SPOD_LPf['P'] : 3D numpy array, complex; mode shape
                    P.shape[0] = number of frequencies;
                    P.shape[1] = number of grid point * number of variable
                    P.shape[2] = number of modes (ranked in descending order by modal energy)
    SPOD_LPf['f'] : 1D numpy array, float; frequency
    '''
    
    time_start=time.time()

    print('--------------------------------------')  
    print('SPOD starts...'                        )
    print('--------------------------------------')    
    
    # ---------------------------------------------------------
    # 1. calculate SPOD parameters
    # ---------------------------------------------------------
    nt = np.shape(x)[0]
    nx = np.shape(x)[1]
     
    # SPOD parser
    [window, nOvlp, nDFT, nBlks] = spod_parser(nt, nx, window, nOvlp)

    #---------------------------------------------------------
    # 2. loop over number of blocks and generate Fourier 
    #    realizations (DFT)
    #---------------------------------------------------------
    print('--------------------------------------')
    print('Calculating temporal DFT'              )
    print('--------------------------------------')
    
    
    # calculate time-averaged result
    x_mean  = np.mean(x,axis=0)
    print(x_mean)

    # obtain frequency axis
    f = np.arange(0,int(np.ceil(nDFT/2)+1))
    f = f/dt/nDFT
    nFreq = f.shape[0]   
    print(nx, nFreq)
    
    # initialize all DFT result in frequency domain
    Q_hat = np.zeros((nx,nFreq,nBlks),dtype = complex) # RAM demanding here       
    # initialize block data in time domain

    Q_blk = np.zeros((nDFT,nx))

    Q_blk_hat = np.zeros((int(nx),int(nFreq)),dtype = complex)
    
    # loop over each block
    for iBlk in range(nBlks):
        
        # get time index for present block
        it_end   = min(iBlk*(nDFT-nOvlp)+nDFT, nt)
        it_start = it_end-nDFT
                    
        print('block', iBlk+1, '/', nBlks, '(', it_start+1, ':', it_end, ')')
        
        # subtract time-averaged results from the block
        Q_blk = x[it_start:it_end,:] - x_mean # column-wise broadcasting
        
        # add window function to avoid spectral leakage
        Q_blk = Q_blk.T * window # row-wise broadcasting
       
        # Fourier transform on block
        Q_blk_hat = 1/np.mean(window)/nDFT*fft(Q_blk)       
        Q_blk_hat = Q_blk_hat[:,0:nFreq]           
        
        # correct Fourier coefficients for one-sided spectrum
        Q_blk_hat[:,1:(nFreq-1)] *= 2
        
        # save block result to the whole domain result 
        Q_hat[:,:,iBlk] = Q_blk_hat

    # remove vars to release RAM
    del x, Q_blk, Q_blk_hat
        
    # memory usage
    process   = psutil.Process(os.getpid())
    RAM_usage = process.memory_info().rss/1024**3 # unit in GBs
        
    time_end=time.time()
    print('--------------------------------------'     )
    print('SPOD finished'                              )
    print('Memory usage: %.2f GB'%RAM_usage            )
    print('Run time    : %.2f s'%(time_end-time_start) )
    print('--------------------------------------'     )
    
    return Q_hat


# -------------------------------------------------------------------------
# sub-functions
def spod_parser(nt, nx, window, nOvlp):
    '''
    Purpose: determine data structure/shape for SPOD
        
    Parameters
    ----------
    nt     : int; number of time snapshots
    nx     : int; number of grid point * number of variable
    window : expect 1D numpy array, float; specified window function values
    weight : expect 1D numpy array; specified weight function
    nOvlp  : expect int; specified number of overlap
    nDFT   : expect int; specified number of DFT points (expect to be same as weight.shape[0])
    method : expect string; specified running mode of SPOD
    
    Returns
    -------
    window : 1D numpy array, float; window function values
    nOvlp  : int; calculated/specified number of overlap
    nDFT   : int; calculated/specified number of DFT points
    nBlks  : int; calculated/specified number of blocks
    '''
    wgt_name = 'unity'
    
    # default nDFT
    nDFT  = 2**(np.floor(np.log2(nt/2))) #!!! why /10 is recommended?
    nDFT  = int(nDFT)
    print('Using default nDFT...')
            
    window   = hammwin(nDFT)
    win_name = 'Hamming'
    print('Using default Hamming window...') 
    nOvlp = int(np.floor(nDFT/2))
    print('Using default nOvlp...')
    # calculate nBlks from nOvlp and nDFT    
    nBlks = int(np.floor((nt-nOvlp)/(nDFT-nOvlp)))
 
    # test feasibility
    if (nDFT < 4) or (nBlks < 2):
        raise ValueError('User sepcified window and nOvlp leads to wrong nDFT and nBlk.')

    print('--------------------------------------')
    print('SPOD parameters summary:'              )
    print('--------------------------------------')
    print('number of DFT points :', nDFT          )
    print('number of blocks is  :', nBlks         )
    print('number of overlap is :', nOvlp         )
    print('Window function      :', win_name      )
    print('Weight function      :', wgt_name      )
    
    return window, nOvlp, nDFT, nBlks

def hammwin(N):
    '''
    Purpose: standard Hamming window
    
    Parameters
    ----------
    N : int; window lengh

    Returns
    -------
    window : 1D numpy array; containing window function values
             n = nDFT
    '''
    
    window = np.arange(0, N)
    window = 0.54 - 0.46*np.cos(2*np.pi*window/(N-1))
    window = np.array(window)

    return window
