import os
from tkinter import Tcl

import numpy as np
from tqdm import tqdm


def fns(data_dir, root):
    fn = [
        fn
        for fn in os.listdir(data_dir)
        if fn.startswith(root) and fn.endswith(f".npy")
    ]
    fn = Tcl().call("lsort", "-dict", fn)
    return fn


def collect_data(fns, data_dir="./data"):
    resize_shape = np.load(f"{data_dir}/{fns[0]}")
    resize_shape = np.shape(resize_shape.squeeze())
    data = []
    for fn in tqdm(fns, desc="Loading data"):
        snap = np.load(f"{data_dir}/{fn}").squeeze()
        snap = np.resize(snap, resize_shape)
        data.append(snap)
        os.remove(f"{data_dir}/{fn}")
    return np.array(data).squeeze()


if __name__ == "__main__":
    data_dir = "./10k"
    root = "u"
    fn = fns(data_dir, root)
    data = collect_data(fn, data_dir)
    np.save(f"{data_dir}/u.npy", data)
    root = "v"
    fn = fns(data_dir, root)
    data = collect_data(fn, data_dir)
    np.save(f"{data_dir}/v.npy", data)
    root = "p"
    fn = fns(data_dir, root)
    data = collect_data(fn, data_dir)
    np.save(f"{data_dir}/p.npy", data)
