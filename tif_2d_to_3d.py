
from multiprocessing import Pool
import tifffile, os, re, torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from datetime import datetime
# import ndtiff
from PIL import Image

def main():
    s = 0.25
    # fn_r = 'downloads/tif_2d_list'
    brainr = '/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/pair5/220423_L57D855P5_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_09-02-27'
    for d in os.listdir(brainr):
        if 'Ultra' not in d or '.' in d: continue
        fn_r = f'{brainr}/{d}'
        fnlist, sortks = listdir_sorted(fn_r)
        print(datetime.now(), len(fnlist), fn_r)
        loader_pool = Pool(processes=min(len(fnlist),30))
        image_list = list(loader_pool.imap(load_tif, tqdm([img_fn for img_fn in fnlist], desc='Load images')))
        image_list = np.stack([np.asarray(img) for img in image_list])
        image_list = torch.nn.functional.interpolate(torch.from_numpy(image_list.astype(np.float32))[None, None], scale_factor=[s, s, s], mode='trilinear')[0,0].numpy()
        #image_list = nib.Nifti1Image(image_list, np.eye(4))
        tifffile.imwrite(f'{fn_r.replace(" ", "_")}.tif', image_list, compression='zlib')
        #nib.save(image_list, f'{fn_r.replace(" ", "_")}.nii.gz')


def load_tif(filename):

    im = Image.open(filename)
    imarray = np.array(im)
    return imarray
    # with tifffile.TiffFile(filename) as tif:
    # with ndtiff.open(filename) as tif:
    #     ltif = len(tif.pages)
    #     try:
    #         full_shape = tif.shaped_metadata[0]['shape']
    #     except:
    #         try:
    #             page = tif.series[0][0]
    #             full_shape = tif.series[0].shape
    #         except:
    #             ltif = 0
    #     if ltif < 10:
    #         img = tif.asarray()
    #     else:
    #         page = tif.series[0][0]
    #         shape, dtype = page.shape, page.dtype
    #         ltif = int(np.prod(full_shape) / np.prod(shape))
    #         # io_logger.info(f'reading tiff with {ltif} planes')
    #         img = np.zeros((ltif, *shape), dtype=dtype)
    #         for i,page in enumerate(tif.series[0]):
    #             img[i] = page.asarray()
    #         img = img.reshape(full_shape)            
    # return img

def listdir_sorted(path, tag='', ftail='.ome.tif', sortkey='_', sortkid=1):
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
