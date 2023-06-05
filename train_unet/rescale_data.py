import os
import os.path as osp
import tifffile as tif
import imageio
from scipy.ndimage import zoom
import numpy as np
import torch.nn.functional as F
import torch
from tqdm import tqdm

src = 'Felix_P4'
tgt = 'Felix_P4_rescaled-as-P15'
scale_r = (4/2.5, 0.75/0.75, .75/.75)
new_shape = (102, 128, 128)
r = '/ram/USERS/ziquanw/data/%s' 
os.makedirs(r % tgt, exist_ok=True)
r = r % src
imgnames = []
for r,d,fs in os.walk(r):
    imgnames.extend([osp.join(r, f) for f in fs if f.endswith('.tif')])

for n in imgnames:
    img = np.stack(imageio.mimread(n))
    if 'mask' not in n:
        out = zoom(img, scale_r)
        print(img.shape, '->', out.shape)
    else:
        img = torch.from_numpy(img)
        out = torch.zeros(new_shape)
        for i in torch.unique(img):
            if i == 0: continue
            y = F.interpolate((img==i).unsqueeze(0).unsqueeze(0) * 1.0, scale_factor=scale_r, mode='nearest-exact').squeeze()
            out[y > 0] = i
        print(n)
        out = out.numpy()
    tif.imwrite(n.replace(src, tgt), out)