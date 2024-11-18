import numpy as np
import sys
import nibabel as nib

fn = sys.argv[1]
mask_fn = fn.replace('resampled', 'registered').replace('C1', 'MASK').replace('.nii', '_all.nii')
save_r = '/cajal/ACMUSERS/ziquanw/Lightsheet/allen_atlas_from-tao/fixed_LS_bg_masked'

x = nib.load(fn).get_fdata()
m = nib.load(mask_fn).get_fdata()
x[m==0] = 0

nib.save(nib.Nifti1Image(x, np.eye(4)), f"{save_r}/{fn.split('/')[-1]}.gz")
