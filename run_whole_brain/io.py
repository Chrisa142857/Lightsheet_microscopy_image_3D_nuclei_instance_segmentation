import os
import numpy as np
import cv2
import tifffile
from tqdm import tqdm

def imread(filename):
    """ read in image with tif or image file type supported by cv2 """
    # ensure that extension check is not case sensitive
    ext = os.path.splitext(filename)[-1].lower()
    if ext== '.tif' or ext=='.tiff':
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
                for i,page in enumerate(tqdm(tif.series[0])):
                    img[i] = page.asarray()
                img = img.reshape(full_shape)            
        return img
    elif ext != '.npy':
        try:
            img = cv2.imread(filename, -1)#cv2.LOAD_IMAGE_ANYDEPTH)
            if img.ndim > 2:
                img = img[..., [2,1,0]]
            return img
        except Exception as e:
            # io_logger.critical('ERROR: could not read file, %s'%e)
            return None
    else:
        try:
            dat = np.load(filename, allow_pickle=True).item()
            masks = dat['masks']
            return masks
        except Exception as e:
            # io_logger.critical('ERROR: could not read masks from file, %s'%e)
            return None
