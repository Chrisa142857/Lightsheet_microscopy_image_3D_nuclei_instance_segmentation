"""
Get list of centroids and rewrite structure indexes from new mask file (.mat)

"""
import os
import numpy as np
import argparse

from scipy.io import loadmat
from skimage.transform import resize

import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt


acquired_img_resolution = [1.21, 1.21, 4]  # Resolution of acquired images in um/pixel
mask_resolution = [10, 10, 10]  # Resolution of mask in um/pixel
mask_res = [i / j for i, j in zip(mask_resolution, acquired_img_resolution)]

main_directory = "/home/ok37/repos/3dunet-centroid/Results"
files = os.listdir(main_directory)

csv_files = [s for s in files if '.csv' in s]
mat_files = [s for s in files if '.mat' in s]

for i in range(len(csv_files)):
    print("Working on file:" + str(csv_files[i]))
    sample = csv_files[i].split('_')[0]
    mask_file = [s for s in mat_files if sample in s]

    mask = loadmat(os.path.join(main_directory, mask_file[0]))
    mask = mask["I_mask"]

    centroids = np.loadtxt(os.path.join(main_directory, csv_files[i]), delimiter=',')

    for j in range(len(centroids)):
        coord = np.floor((centroids[j, :3]-10) / mask_res)
        coord = np.asarray(coord, dtype=int)

        centroids[j, 3] = mask[tuple(coord)]

    save_name = csv_files[i]
    print("Writing file:" + str(csv_files[i]))
    np.savetxt(save_name, centroids.astype(int), delimiter=",", fmt='%u')
