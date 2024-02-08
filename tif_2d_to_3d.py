
from multiprocessing import Pool
import tifffile, os, re
import numpy as np
import nibabel as nib
from tqdm import tqdm

def main():
    fn_r = 'downloads/tif_2d_list'
    fnlist, sortks = listdir_sorted(fn_r)
    print(fnlist)
    loader_pool = Pool(processes=min(len(fnlist),30))
    image_list = list(loader_pool.imap(load_tif, tqdm([img_fn for img_fn in fnlist], desc='Load images')))
    image_list = np.stack([np.asarray(img) for img in image_list])
    image_list = nib.Nifti1Image(image_list, np.eye(4))
    nib.save(image_list, f'{fn_r}/image3d_concat_above_2d.nii')


def load_tif(filename):
    with tifffile.TiffFile(filename) as tif:
        ltif = len(tif.pages)
        try:
            full_shape = tif.shaped_metadata[0]['shape']
        except:
            try:
                page = tif.series[0][0]
                full_shape = tif.series[0].shape
            except:
                ltif = 0
        if ltif < 10:
            img = tif.asarray()
        else:
            page = tif.series[0][0]
            shape, dtype = page.shape, page.dtype
            ltif = int(np.prod(full_shape) / np.prod(shape))
            # io_logger.info(f'reading tiff with {ltif} planes')
            img = np.zeros((ltif, *shape), dtype=dtype)
            for i,page in enumerate(tif.series[0]):
                img[i] = page.asarray()
            img = img.reshape(full_shape)            
    return img

def listdir_sorted(path, tag='', ftail='_stitched.tif', sortkid=1):
    fs = os.listdir(path)
    fs = [os.path.join(path, f) for f in fs if tag in f and f.endswith(ftail)]
    ks = []
    for f in fs:
        k = f.split('/')[-1].split('_')[sortkid]
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