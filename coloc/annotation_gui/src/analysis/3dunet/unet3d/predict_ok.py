import os

import nibabel as nib

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

import SimpleITK as sitk
import numpy as np
import tables

from utils import pickle_load
from utils.patches import reconstruct_from_patches, get_patch_from_3d_data, compute_patch_indices
from augment import permute_data, generate_permutation_keys, reverse_permute_data

from train_isensee2017 import config

input_img_directory = "/ssd2/userdata/ok37/TOP16R/output/stitched"

def determine_chunk_size(resolution = [1.21,1.21,4], ):
    # Take z number of images
    # Take certain chunk size that keeps native 3D chunk size as a cube
    # Does not need to be isotropic
    # Trained unet is for 1x1x2.5
    a = 1



def run_prediction():
    files = os.listdir(input_img_directory)
    matching = [s for s in files if "C1" in s] #assume reference images are labeled C1

    # range(len(matching))
    for i in range(0, 1):
        file_path = os.path.join(input_img_directory, matching[i])
        img = sitk.ReadImage(file_path)
        plt.imshow(sitk.GetArrayViewFromImage(img))


if __name__ == "__main__":
    run_prediction()
