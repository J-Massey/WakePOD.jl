"""
@author: Jonathan Massey
@description: This script will read all the .png files in one folder,
              sort them in some sensible order, then make a gif called 
              'movie.gif'.
              
              To use it `cd /path/to/folder && python3 /path/to/gif_generation.py`
@contact: masseyjmo@gmail.com
"""


import imageio
import os
from tkinter import Tcl
import numpy as np
from pygifsicle import optimize


def fns(dirn):
    fns = [fn for fn in os.listdir(dirn) if fn.endswith(f".png")]
    fns = Tcl().call("lsort", "-dict", fns)
    return fns


def main(nm):
    cwd = os.getcwd()
    ims = fns(cwd)
    images = []
    for fn in ims:
        images.append(imageio.v3.imread(fn, plugin="pillow", mode="RGBA"))

    images = np.stack(images, axis=0)
    imageio.v3.imwrite(
        nm,
        images,
        plugin="pillow",
        # format="GIF-PIL",
        mode="RGBA",
        duration=2/images.size,
        loop=0,
        # transparency=100,
        disposal=2,
    )
    # optimize(nm)


if __name__ == "__main__":
    main("u_unwarped.gif")