import tifffile as tif
import os, torch
from tqdm import tqdm, trange
import sys
import numpy as np
from torch.utils.data import DataLoader,Dataset
from unet import NISModel
import io

def main():
    trained_model = 'downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_22_42_54.153497_epoch_21'
    model = NISModel(device=torch.device('cuda:%d' % int(sys.argv[3])), pretrained_model=trained_model)
    # model = models.CellposeModel(device=torch.device('cuda:0'), pretrained_model=trained_model)
    pair_tag = 'pair15'
    brain_tag = 'L57D855P2'
    save_r = '/lichtman/ziquanw/Lightsheet/results/P4/%s/%s' % (pair_tag, brain_tag)
    img_r = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/stitched' % (pair_tag, brain_tag)
    eval_tag = '_C1_'
    fs = [os.path.join(img_r, f) for f in os.listdir(img_r) if eval_tag in f and f.endswith('.tif')]
    orig_fs = fs.copy()
    depth_start = 1
    for f in orig_fs:
        fs[int(f.split('/')[-1].split('_')[1]) - depth_start] = f
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    dset = BrainSliceDataset(fs[start:end])
    dloader = DataLoader(dset,batch_size=1,shuffle=False,num_workers=8,collate_fn=collate_fn_pass)
    for data in tqdm(dloader):
        img, fn = data[0]
        prob = model.get_prob(img, diameter=None, batch_size=100, channels=[0,0], do_3D=False)
        np.savez_compressed('%s/%s' % (save_r, fn.split('/')[-1].replace('.tif', '.npz')), prob=prob)

class BrainSliceDataset(Dataset):
    def __init__(self, fs):
        self.fs = fs

    def __getitem__(self, i):
        return np.asarray(io.imread(self.fs[i]))[np.newaxis,...], self.fs[i]

    def __len__(self):
        return len(self.fs)

def collate_fn_pass(x):
    return x

if __name__ == "__main__":
    main()