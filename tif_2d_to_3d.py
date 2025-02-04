
from multiprocessing import Pool
import tifffile, os, re, torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from datetime import datetime
# import ndtiff
from PIL import Image

def main():
    s = 0.1
    org_res = [4, .75, .75]
    # fn_r = 'downloads/tif_2d_list'
    # brainr = '/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/pair5/220423_L57D855P5_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_09-02-27'
    # for d in os.listdir(brainr):
    #     if 'Ultra' not in d or '.' in d: continue
    #     fn_r = f'{brainr}/{d}'
    #     fnlist, sortks = listdir_sorted(fn_r)
    #     print(datetime.now(), len(fnlist), fn_r)
    #     loader_pool = Pool(processes=min(len(fnlist),30))
    #     image_list = list(loader_pool.imap(load_tif, tqdm([img_fn for img_fn in fnlist], desc='Load images')))
    #     image_list = np.stack([np.asarray(img) for img in image_list])
    #     image_list = torch.nn.functional.interpolate(torch.from_numpy(image_list.astype(np.float32))[None, None], scale_factor=[s, s, s], mode='trilinear')[0,0].numpy()
    #     #image_list = nib.Nifti1Image(image_list, np.eye(4))
    #     tifffile.imwrite(f'{fn_r.replace(" ", "_")}.tif', image_list, compression='zlib')
    #     #nib.save(image_list, f'{fn_r.replace(" ", "_")}.nii.gz')
    
    # fn_r = '/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/pair19/L79D769P5/LS_image_stitched'
    # fnlist, sortks = listdir_sorted(fn_r)
    # print(datetime.now(), len(fnlist), fn_r)
    # loader_pool = Pool(processes=min(len(fnlist),30))
    # image_list = list(loader_pool.imap(load_tif, tqdm([img_fn for img_fn in fnlist], desc='Load images')))
    # image_list = np.stack([np.asarray(img) for img in image_list])
    # image_list = torch.nn.functional.interpolate(torch.from_numpy(image_list.astype(np.float32))[None, None], scale_factor=[s*org_res[0], s*org_res[1], s*org_res[2]], mode='trilinear')[0,0].numpy()
    # image_list = nib.Nifti1Image(image_list, np.eye(4))
    # nib.save(image_list, f'{fn_r.replace(" ", "_")}.nii.gz')

    fn_r = '/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/pair19/L79D769P5/binary_mask_stitched'
    fnlist, sortks = listdir_sorted(fn_r, ftail='.nii.gz')
    print(datetime.now(), len(fnlist), fn_r)
    loader_pool = Pool(processes=min(len(fnlist),30))
    image_list = list(loader_pool.imap(load_tif, tqdm([img_fn for img_fn in fnlist], desc='Load images')))
    image_list = np.stack([np.asarray(img) for img in image_list])
    image_list = torch.nn.functional.interpolate(torch.from_numpy(image_list.astype(np.float32))[None, None], scale_factor=[s*org_res[0], s*org_res[1], s*org_res[2]], mode='trilinear')[0,0].numpy()
    image_list = nib.Nifti1Image(image_list, np.eye(4))
    nib.save(image_list, f'{fn_r.replace(" ", "_")}.nii.gz')

def load_tif(filename):
    if filename.endswith('.tif'):
        im = Image.open(filename)
        imarray = np.array(im)
        return imarray
    elif filename.endswith('.nii') or filename.endswith('.nii.gz'):
        return np.array(nib.load(filename).get_fdata())

def listdir_sorted(path, tag='', ftail='.tif', sortkey='_C01_', sortkid=-1):
    fs = os.listdir(path)
    fs = [os.path.join(path, f) for f in fs if tag in f and f.endswith(ftail)]
    ks = []
    for f in fs:
        k = f.split('/')[-1].split(sortkey)[sortkid]
        k = int(re.sub("[^0-9]", "", k))
        ks.append(k)
    orgks = ks.copy()
    ks.sort()
    sorted_fs = []
    for k in ks:
        sorted_fs.append(fs[orgks.index(k)])
        
    return sorted_fs, ks

if __name__ == '__main__':
    main()
