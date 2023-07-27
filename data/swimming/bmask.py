
import os
from tkinter import Tcl
import numpy as np


def bmask():
    fnsu, fnsv, fnsp, fnsb = fns()
    for idx, (fnu, fnv, fnp, fnb) in enumerate(zip(fnsu, fnsv, fnsp, fnsb)):
        u = np.load(os.path.join("./data", fnu))
        v = np.load(os.path.join("./data", fnv))
        p = np.load(os.path.join("./data", fnp))
        b = np.load(os.path.join("./data", fnb))
        bmask = np.where(b <= 1, False, True)
        u = np.where(bmask, u, 0)
        np.save(os.path.join("./data", f"u_{idx}"), u)
        v = np.where(bmask, v, 0)
        np.save(os.path.join("./data", f"v_{idx}"), v)
        p = np.where(bmask, p, 0)
        np.save(os.path.join("./data", f"p_{idx}"), p)
        # Now remove the files
        print(f"Removing {fnu}, {fnv}, {fnb}")
        try:
            os.remove(os.path.join("./data", fnu))
            os.remove(os.path.join("./data", fnv))
            os.remove(os.path.join("./data", fnp))
            os.remove(os.path.join("./data", fnb))
        except FileNotFoundError:
            pass


def fns():
    fnsu = [
        fn
        for fn in os.listdir("./data")
        if fn.startswith("fluid_u") and fn.endswith(f".npy")
    ]
    fnsu = Tcl().call("lsort", "-dict", fnsu)
    fnsv = [
        fn
        for fn in os.listdir("./data")
        if fn.startswith("fluid_v") and fn.endswith(f".npy")
    ]
    fnsv = Tcl().call("lsort", "-dict", fnsv)
    fnsp = [
        fn
        for fn in os.listdir("./data")
        if fn.startswith("fluid_p") and fn.endswith(f".npy")
    ]
    fnsp = Tcl().call("lsort", "-dict", fnsp)
    fnsb = [
        fn
        for fn in os.listdir("./data")
        if fn.startswith('bodyF') and fn.endswith(f".npy")
    ]
    fnsb = Tcl().call("lsort", "-dict", fnsb)
    return fnsu, fnsv, fnsp, fnsb

if __name__ == "__main__":
    bmask()