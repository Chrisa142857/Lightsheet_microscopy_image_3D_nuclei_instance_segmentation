"""
Get list of centroids and remeasure channel intensities from high resolution .tiff series

"""
import os
import numpy as np
import cv2
import scipy.ndimage as ndimage


sample = 'WT11L'
markers = [2,3]
centroids_save = '/media/SteinLab5/WT11L/output'
#centroids_path = '/home/ok37/tc_pipeline/evaluation/Results'

centroids_file = os.path.join(centroids_save, sample + '_centroids2.csv')
images_path = os.path.join(centroids_save,'stitched')
centroids = np.loadtxt(os.path.join(centroids_file), delimiter=',')

img_files = sorted(os.listdir(images_path))
marker_files = [[file for file in img_files if "C" + str(marker) in file] for marker in markers]
z_idx = np.unique([[int(file.split('_')[1]) for file in marker] for marker in marker_files])

centroids_new = np.copy(centroids)

for i, z in enumerate(z_idx):
    print("Working on slice:" + str(z))
    cen_z = centroids[:, 2] == i
    if any(cen_z):
        for j, marker in enumerate(markers):
            I = cv2.imread(os.path.join(images_path, marker_files[j][i]), -1)
            print("Image:" + str(marker_files[j][i]))
            coord = centroids[cen_z, :2]
            coord = np.asarray(coord, dtype=int)

            #I = cv2.blur(I.astype('float32'), (2, 2))
            intensity = I[coord[:, 0], coord[:, 1]]
            print(np.mean(intensity))

            centroids_new[cen_z, 3+marker] = intensity

np.savetxt(centroids_file, centroids_new.astype(int), delimiter=",", fmt='%u')
