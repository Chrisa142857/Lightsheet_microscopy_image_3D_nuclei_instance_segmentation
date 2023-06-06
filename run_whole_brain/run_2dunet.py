import tifffile as tif
import os, torch
from tqdm import tqdm, trange
import sys
import numpy as np
from torch.utils.data import DataLoader,Dataset
from unet import NISModel
import utils

def main():
    trained_model = 'downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_22_42_54.153497_epoch_21'
    # model = NISModel(device=torch.device('cuda:%d' % int(sys.argv[3])), pretrained_model=trained_model)
    model = NISModel(device=torch.device('cuda:0'), pretrained_model=trained_model)
    # model = models.CellposeModel(device=torch.device('cuda:0'), pretrained_model=trained_model)
    pair_tag = 'pair15'
    brain_tag = 'L73D766P4' # L73D766P9
    save_r = '/lichtman/ziquanw/Lightsheet/results/P4'
    save_r = '%s/%s' % (save_r, pair_tag)
    os.makedirs(save_r, exist_ok=True)
    save_r = '%s/%s' % (save_r, brain_tag)
    os.makedirs(save_r, exist_ok=True)
    img_r = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/stitched' % (pair_tag, brain_tag)
    mask_tile = utils.MaskTiledImg(maskn='/lichtman/Felix/Lightsheet/P4/%s/output_%s/registered/%s_MASK_topro_25_all.nii' % (pair_tag, brain_tag, brain_tag), img_zres=4)
    model.mask_tile = mask_tile
    eval_tag = '_C1_'
    fs = [os.path.join(img_r, f) for f in os.listdir(img_r) if eval_tag in f and f.endswith('.tif')]
    orig_fs = fs.copy()
    depth_start = 1
    for f in orig_fs:
        fs[filename_to_depth(f, depth_start)] = f
    # start = int(sys.argv[1])
    # end = int(sys.argv[2])
    start = 300
    end = 600
    dset = BrainSliceDataset(fs[start:end])
    dloader = DataLoader(dset,batch_size=1,shuffle=False,num_workers=8,collate_fn=collate_fn_pass)
    for data in tqdm(dloader):
        img, fn = data[0]
        model.mask_tile.z = filename_to_depth(fn, depth_start)
        prob = model.get_prob(img, diameter=None, batch_size=100, channels=[0,0])
        np.savez_compressed('%s/%s' % (save_r, fn.split('/')[-1].replace('.tif', '.npz')), prob=prob)

def filename_to_depth(f, depth_start):
    return int(f.split('/')[-1].split('_')[1]) - depth_start

class BrainSliceDataset(Dataset):
    def __init__(self, fs):
        self.fs = fs

    def __getitem__(self, i):
        return np.asarray(utils.imread(self.fs[i]))[np.newaxis,...], self.fs[i]

    def __len__(self):
        return len(self.fs)

def collate_fn_pass(x):
    return x

if __name__ == "__main__":
    main()