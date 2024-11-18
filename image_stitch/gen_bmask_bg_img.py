import os 
import torch
from datetime import datetime
from PIL import Image
import numpy as np
import nibabel as nib
from tqdm import tqdm
import tifffile

def main():
    
    r='/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4'
    # p='pair6'
    # b='220416_L57D855P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_09-52-07'
    p='pair5'
    # b='220423_L57D855P5_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_09-02-27'
    b='220422_L57D855P4_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_16-54-32'
    for t in os.listdir(f'{r}/{p}/{b}'):
        if '.' in t: continue
        assert 'Ultra' in t, t
        print(datetime.now(), f"Generate {p}/{b}/{t}")
        gen(f'{r}/{p}/{b}/{t}')
        
    # for p in os.listdir(r):
    #     if 'pair' not in p: continue
    #     for b in os.listdir(f'{r}/{p}'):
    #         if b[0] == 'L': continue
    #         if b == '220805_L74D769P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_15-51-36.manyZstitch': continue
    #         for t in os.listdir(f'{r}/{p}/{b}'):
    #             assert 'Ultra' in t, t
    #             for f in os.listdir(f'{r}/{p}/{b}/{t}'):
    #                 if 'binary_mask.zip' not in f: continue
    #                 if os.path.exists(f'{r}/{p}/{b}/{t}/{f}'.replace('.zip', '.nii.gz')): continue
    #                 print(datetime.now(), f"Convert {p}/{b}/{t}/{f}")
    #                 gen(f'{r}/{p}/{b}/{t}/{f}')

def gen(dir):
    savefn = os.listdir(dir)[0].split('_')[0] + '_' + dir.split('/')[-1] + '_C01_tile-stack3d.ome.tif'
    rdir = dir.replace('image_before_stitch', 'results')
    # if os.path.exists(f"{rdir}/{savefn}"): return
    dratio = 0.5# 0.3
    zratio = 1# 4/2.5
    m = [None for fn in os.listdir(dir) if fn.endswith('.tif')]
    for fn in tqdm(os.listdir(dir), desc=f'{datetime.now()} Load tif image'):
        if fn.endswith('.tif'):
            zi = int(fn.split('Z')[1][:4])
            im = Image.open(f'{dir}/{fn}')
            imarray = np.array(im).astype(float)
            imarray = torch.from_numpy(imarray)
            m[zi-1] = imarray
    m = torch.stack(m)
    # m = np.stack(m)
    print(datetime.now(), 'Downsample 3D image')
    m = torch.nn.functional.interpolate(m[None, None], scale_factor=[dratio*zratio,dratio,dratio])[0,0]
    m = m.numpy()
    m = m.astype(np.uint16)
    # print(m.shape, m.dtype)
    # m = np.transpose(m, [2,1,0])
    # m = Image.fromarray(m, 'I;16')
    # m.save(f"{rdir}/{savefn}")
    tifffile.imsave(f"{rdir}/{savefn}", m, shape=m.shape, dtype='uint16', bigtiff=True, metadata={'spacing': 1, 'unit': 'um', 'axes': 'ZYX'})
    # zi = 0
    # fns = sort_stackname([fn for fn in os.listdir(rdir) if fn.endswith('mask.nii.gz')])
    # for fn in tqdm(fns, desc=f'{datetime.now()} Save back image'):
    #     depth = nib.load(f'{rdir}/{fn}').get_fdata().shape[0]
    #     nib.save(nib.Nifti1Image(m[zi:zi+depth], np.eye(4)), f"{rdir}/{fn.replace('binary_mask', 'downsampled_img')}")
    #     zi = zi + depth


def sort_stackname(stack_names):
    stack_z = []
    for stack_name in stack_names:
        stack_z.append(int(stack_name.split('zmin')[1].split('_')[0]))

    argsort = np.argsort(stack_z)
    return [stack_names[i] for i in argsort]

if __name__ == '__main__': main()