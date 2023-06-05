import os
import os.path as osp
import tifffile as tif
from tqdm import tqdm
import numpy as np

save_r = 'data_P4_P15_rescaled-as-P15'
os.makedirs(save_r, exist_ok=True)

def read_txt(p):
    with open(p, 'r') as f:
        return [l.split('/')[-1].replace('_masks.tif', '').replace('.tif', '') for l in f.read().split('\n')[:-1]]


def main(r):
    train_fs = read_txt('train_list.txt')
    test_fs = read_txt('test_list.txt')
    val_fs = read_txt('val_list.txt')
    imgnames = []
    for r,d,fs in os.walk(r):
        imgnames.extend([osp.join(r, f) for f in fs if f.endswith('.tif')])


    for n in tqdm(imgnames):
        fn = n.split('/')[-1].replace('_masks.tif', '').replace('.tif', '')
        if fn in train_fs: 
            tag = 'train'
        elif fn in val_fs: 
            tag = 'val'
        elif fn in test_fs: 
            tag = 'test'
        else:
            print('wrong type'); exit()
        os.makedirs(osp.join(save_r, tag), exist_ok=True)
        img = tif.imread(n)
        for i in range(len(img)):
            if 'mask' not in n:
                saven = osp.join(save_r, tag, fn + ('_%02d.tif' % i))
                out = img[i]
            else:
                saven = osp.join(save_r, tag, fn + ('_%02d_masks.tif' % i))
                out = np.array(img[i], dtype=np.int32)
            tif.imwrite(saven, out)

vol_r = 'Felix_P4_rescaled-as-P15'
r = '/ram/USERS/ziquanw/data/%s' % vol_r
main(r)
vol_r = 'Carolyn_org_Sept'
r = '/ram/USERS/ziquanw/data/%s' % vol_r
main(r)