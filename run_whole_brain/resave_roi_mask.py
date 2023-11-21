import numpy as np
import nibabel as nib
import tifffile, os

def f(pair_tag, brain_tag):
    mask_zres=25
    img_zres=2.5
    zratio = mask_zres / img_zres
    imgr = f"/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched/"
    depth = max([int(f.split('_')[1]) for f in os.listdir(imgr) if '_C1_' in f])
    saver = '/lichtman/ziquanw/Lightsheet/roi_mask/%s' % pair_tag
    os.makedirs(saver, exist_ok=True)
    saver = '%s/%s' % (saver, brain_tag)
    os.makedirs(saver, exist_ok=True)
    maskn = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/registered/%s_MASK_topro_25_all.nii' % (pair_tag, brain_tag, brain_tag)
    mask = np.transpose(nib.load(maskn).get_fdata(), (2, 0, 1))[:, :, ::-1].copy()
    maskn = maskn.split('/')[-1]
    print(mask.shape, mask.dtype, mask[154].max(), mask[154].min())
    
    for z in range(depth):
        fn = '%s_MASK_topro_25_%04d.tif' % (brain_tag, z+1)
        with tifffile.TiffWriter(saver+'/'+fn) as tif:
            tif.save(mask[int(z/zratio)].astype(np.int16))

if __name__ == '__main__':
    
    brain_tag = 'L73D766P4' # L73D766P9
    pair_tag = 'pair15'
    f(pair_tag, brain_tag)