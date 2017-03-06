import numpy as np
import bcolz
from matplotlib import pyplot as plt
import math

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def plot_imgs(imgs, n_col = 5):
    plt.subplots_adjust()
    count = imgs.shape[0]
    n_row = math.ceil(count / n_col)
    n_row = 2 if n_row == 1 else n_row
    fig, ax = plt.subplots(n_row, n_col, figsize=[16, 3 * n_row/2] )
    for i in range(n_row):
        for j in range(n_col):
            ij = j + i * n_col
            if ij < count:
                ax[i,j].axis('off')
                ax[i,j].imshow(imgs[ij])