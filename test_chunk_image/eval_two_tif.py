import tifffile as tif
import torch
from utils import mask2mask_list
from test_stack_gap import eval_f1
import numpy as np

f1 = 'results3D_gap_[0]/nonstitch/cropped-in-[37]_Side0[01x00]sub1.tif'
f2 = '../data/Carolyn_org_Sept/masks/Side0[01x00]sub1_masks.tif'
whole_stack_mask = tif.imread(f1)
gt = tif.imread(f2)
eval_device = 0
prec, rec, f1, _, _, _ = eval_f1(torch.from_numpy(mask2mask_list(whole_stack_mask)).to('cuda:%d'%eval_device), torch.from_numpy(mask2mask_list(gt)).to('cuda:%d'%eval_device))
print(prec, rec, f1)

