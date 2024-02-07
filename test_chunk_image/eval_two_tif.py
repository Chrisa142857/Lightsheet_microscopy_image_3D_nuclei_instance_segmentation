import tifffile as tif
import torch
from utils import mask2mask_list, eval_f1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib as mpl



plt.rcParams['font.family'] = 'Arial'

def main(f1, f2):
    whole_stack_mask = tif.imread(f1)
    gt = tif.imread(f2)
    eval_device = 0
    prec, rec, f1, _, _, _ = eval_f1(torch.from_numpy(mask2mask_list(whole_stack_mask)).to('cuda:%d'%eval_device), torch.from_numpy(mask2mask_list(gt)).to('cuda:%d'%eval_device))
    # print(f1, f2)
    return prec, rec, f1

xs, y1s, y1err = [], [], []
y2s, y2err = [], []
y3s, y3err = [], []
labels = []

tag = 'n4'
precs, recs, f1s = [], [], []
imgn = 'Felix_01x00_C00_Z0800_sub5_2_cp_masks.tif'
f1 = f'../downloads/train_data/test_out_rescaled_{tag}/{imgn}'
f2 = f'/ram/USERS/ziquanw/data/Felix_P4/{imgn.replace("_cp", "")}'
prec, rec, f1 = main(f1, f2)
precs.append(prec)
recs.append(rec)
f1s.append(f1)
imgn = 'Side0[00x02]sub4_cp_masks.tif'
f1 = f'../downloads/train_data/test_out_rescaled_{tag}/{imgn}'
f2 = f'/ram/USERS/ziquanw/data/Carolyn_org_Sept/masks/{imgn.replace("_cp", "")}'
prec, rec, f1 = main(f1, f2)
precs.append(prec)
recs.append(rec)
f1s.append(f1)
imgn = 'Side0[01x00]sub1_cp_masks.tif'
f1 = f'../downloads/train_data/test_out_rescaled_{tag}/{imgn}'
f2 = f'/ram/USERS/ziquanw/data/Carolyn_org_Sept/masks/{imgn.replace("_cp", "")}'
prec, rec, f1 = main(f1, f2)
precs.append(prec)
recs.append(rec)
f1s.append(f1)
xs.append(int(tag[1:]))
y1s.append(np.mean(precs))
y2s.append(np.mean(rec))
y3s.append(np.mean(f1))
y1err.append(np.std(precs))
y2err.append(np.std(rec))
y3err.append(np.std(f1))


tag = 'n8'
precs, recs, f1s = [], [], []
imgn = 'Felix_01x00_C00_Z0800_sub5_2_cp_masks.tif'
f1 = f'../downloads/train_data/test_out_rescaled_{tag}/{imgn.replace("_cp", "")}'
f2 = f'/ram/USERS/ziquanw/data/Felix_P4/{imgn.replace("_cp", "")}'
prec, rec, f1 = main(f1, f2)
precs.append(prec)
recs.append(rec)
f1s.append(f1)
imgn = 'Side0[00x02]sub4_cp_masks.tif'
f1 = f'../downloads/train_data/test_out_rescaled_{tag}/{imgn.replace("_cp", "")}'
f2 = f'/ram/USERS/ziquanw/data/Carolyn_org_Sept/masks/{imgn.replace("_cp", "")}'
prec, rec, f1 = main(f1, f2)
precs.append(prec)
recs.append(rec)
f1s.append(f1)
imgn = 'Side0[01x00]sub1_cp_masks.tif'
f1 = f'../downloads/train_data/test_out_rescaled_{tag}/{imgn.replace("_cp", "")}'
f2 = f'/ram/USERS/ziquanw/data/Carolyn_org_Sept/masks/{imgn.replace("_cp", "")}'
prec, rec, f1 = main(f1, f2)
precs.append(prec)
recs.append(rec)
f1s.append(f1)
xs.append(int(tag[1:]))
y1s.append(np.mean(precs))
y2s.append(np.mean(rec))
y3s.append(np.mean(f1))
y1err.append(np.std(precs))
y2err.append(np.std(rec))
y3err.append(np.std(f1))

tag = 'n12'
precs, recs, f1s = [], [], []
imgn = 'Felix_01x00_C00_Z0800_sub5_2_cp_masks.tif'
f1 = f'../downloads/train_data/test_out_rescaled_{tag}/{imgn.replace("_cp", "")}'
f2 = f'/ram/USERS/ziquanw/data/Felix_P4/{imgn.replace("_cp", "")}'
prec, rec, f1 = main(f1, f2)
precs.append(prec)
recs.append(rec)
f1s.append(f1)
imgn = 'Side0[00x02]sub4_cp_masks.tif'
f1 = f'../downloads/train_data/test_out_rescaled_{tag}/{imgn.replace("_cp", "")}'
f2 = f'/ram/USERS/ziquanw/data/Carolyn_org_Sept/masks/{imgn.replace("_cp", "")}'
prec, rec, f1 = main(f1, f2)
precs.append(prec)
recs.append(rec)
f1s.append(f1)
imgn = 'Side0[01x00]sub1_cp_masks.tif'
f1 = f'../downloads/train_data/test_out_rescaled_{tag}/{imgn.replace("_cp", "")}'
f2 = f'/ram/USERS/ziquanw/data/Carolyn_org_Sept/masks/{imgn.replace("_cp", "")}'
prec, rec, f1 = main(f1, f2)
precs.append(prec)
recs.append(rec)
f1s.append(f1)
xs.append(int(tag[1:]))
y1s.append(np.mean(precs))
y2s.append(np.mean(rec))
y3s.append(np.mean(f1))
y1err.append(np.std(precs))
y2err.append(np.std(rec))
y3err.append(np.std(f1))

n16_prec = [0.9547, 0.9631]
n16_rec = [0.8993, 0.9032]
n16_f1 = [0.9282, 0.9301]
xs.append(16)
oxs = [0.25,0.5,0.75,1.0]
y1s.append(np.mean(n16_prec))
y2s.append(np.mean(n16_rec))
y3s.append(np.mean(n16_f1))
y1err.append(np.std(n16_prec))
y2err.append(np.std(n16_rec))
y3err.append(np.std(n16_f1))

total_num = 6847
def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y

cmp = mpl.colormaps['tab10']

plt.figure(figsize=(5,3))

xs = np.asarray(oxs)
proj_xs = np.asarray(oxs)
y1s = np.asarray(y1s)
y2s = np.asarray(y2s)
y3s = np.asarray(y3s)
plt.plot(proj_xs, LinearRegression().fit(xs.reshape(-1, 1), y1s.reshape(-1, 1)).predict(proj_xs.reshape(-1, 1)), color=cmp(0))
plt.plot(proj_xs, LinearRegression().fit(xs.reshape(-1, 1), y2s.reshape(-1, 1)).predict(proj_xs.reshape(-1, 1)), color=cmp(1))
plt.plot(proj_xs, LinearRegression().fit(xs.reshape(-1, 1), y3s.reshape(-1, 1)).predict(proj_xs.reshape(-1, 1)), color=cmp(2))


plt.scatter(xs, y1s, marker='s', label='Precision', color=cmp(0))
plt.scatter(xs, y2s, marker='o', label='Recall', color=cmp(1))
plt.scatter(xs, y3s, marker='^', label='F1 score', color=cmp(2))
plt.tight_layout()
plt.xticks(ticks=proj_xs, labels=["1,711", "3,423", "5,135", "6,847"])
plt.xlabel('train data size')
plt.ylabel('performance')
plt.legend()
plt.savefig('diff_train_num.png')
plt.savefig('diff_train_num.svg')
print(xs, y1s, y1err, y2s, y2err, y3s, y3err)