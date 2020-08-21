#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    if args.filename.endswith('.dat'):
        data = np.loadtxt(args.filename)
        x, rho, vel, pre = data.T

    elif args.filename.endswith('.h5'):
        import h5py
        h5f = h5py.File(args.filename, 'r')
        x, (rho, vel, pre) = h5f['cell_centers'][()], h5f['primitive'][()].T

    plt.plot(x, rho, '-o', mfc='none')
    plt.show()
