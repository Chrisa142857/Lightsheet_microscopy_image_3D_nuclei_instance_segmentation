import h5py, os, json
import nibabel as nib
import numpy as np
import torch
from datetime import datetime
import pandas
import matplotlib.pyplot as plt

def main():
    std_resolution = (2.5, .75, .75)
    in_resolution = (4, .75, .75)
    scale_r = [i/s for i, s in zip(in_resolution, std_resolution)]
    vol_scale = scale_r[0]*scale_r[1]*scale_r[2]
    _r = '/lichtman/ziquanw/Lightsheet/results/P4'
    mask_r = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/registered/%s_MASK_topro_25_all.nii'
    save_roi_id = 16001
    data_list = []
    for r, d, fs in os.walk(_r):
        data_list.extend([os.path.join(r, f) for f in fs if f.endswith('_remap.json')])
        # if len(fs) > 0: break
    out = []
    tags = []
    counts = []
    fig, axes = plt.subplots(2, 1)
    for di, remap_fn in enumerate(data_list):
        brain_tag = os.path.dirname(remap_fn).split('/')[-1]
        pair_tag = os.path.dirname(os.path.dirname(remap_fn)).split('/')[-1]
        print(datetime.now(), "Statistic brain", pair_tag, brain_tag)
        nuclei = torch.from_numpy(pandas.read_csv(os.path.join(_r, 'NIS_%s_%s_RoI%d.csv' % (pair_tag, brain_tag, save_roi_id))).to_numpy())
        center = nuclei[:, :3]  #  percentage
        
        volume = nuclei[:, 3] * 2.25  #  micron
        
        counts.append(center.shape[0])
        tags.append('%s-%s' % (pair_tag, brain_tag))
        axes[0].bar(di+1, center.shape[0])
        axes[1].violinplot(volume[volume<1000])
        break
    fig.show()
    fig.savefig('downloads/statistic.png')
    
if __name__ == '__main__': main()