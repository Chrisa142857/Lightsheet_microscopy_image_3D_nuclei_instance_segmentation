import tifffile as tif
import os, torch
from tqdm import tqdm, trange
import sys
import numpy as np
from torch.utils.data import DataLoader,Dataset
from unet import NISModel
import utils
from multiprocessing import Pool
from datetime import datetime
from torch.multiprocessing import Process


def main():
    dir_n = 'flow_2d'
    pair_tag = 'pair15'
    brain_tag = 'L73D766P4' # L73D766P9
    save_r = '/lichtman/ziquanw/Lightsheet/results/P4'
    # std_resolution = (2.5, .75, .75)
    # in_resolution = (4, .75, .75)
    # scale_r = [i/s for i, s in zip(in_resolution, std_resolution)]
    trained_model = 'downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_22_42_54.153497_epoch_21'
    model = NISModel(device=torch.device('cuda:%d' % int(sys.argv[3])), pretrained_model=trained_model)
    # model = NISModel(device=torch.device('cuda:0'), pretrained_model=trained_model)
    save_r = '%s/%s' % (save_r, pair_tag)
    os.makedirs(save_r, exist_ok=True)
    save_r = '%s/%s' % (save_r, brain_tag)
    os.makedirs(save_r, exist_ok=True)
    save_r = '%s/%s' % (save_r, dir_n)
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
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    # start = 500
    # end = 600
    dset = BrainSliceDataset(fs[start:end]) if end > 0 else BrainSliceDataset(fs[start:])
    dloader = DataLoader(dset,batch_size=1,shuffle=False,num_workers=8,collate_fn=collate_fn_pass)
    save_processes = []
    si = 0
    print(datetime.now(), "Load slice %d" % (start + si))
    for data in tqdm(dloader, desc="Run 2D Unet from slice %d to %d" % (start, end)):
        si += 1
        img, fn = data[0]
        model.mask_tile.z = filename_to_depth(fn, depth_start)
        print(datetime.now(), "Input to model")
        prob = model.get_prob(img, diameter=None, batch_size=100, channels=[0,0])
        save_process = Process(target=save_output, args=('%s/%s' % (save_r, fn.split('/')[-1].replace('.tif', '.npy')), prob))
        save_process.start()
        save_processes.append(save_process)
        print(datetime.now(), "Load slice %d" % (start + si))
        for p in save_processes:
            if not p.is_alive(): del p
    for p in save_processes:
        if p.is_alive(): p.join()
        
def save_output(save_fn, prob):
    print(datetime.now(), "Save output to %s" % save_fn.split('/')[-1])
    np.save(save_fn, prob)

def filename_to_depth(f, depth_start):
    return int(f.split('/')[-1].split('_')[1]) - depth_start

def load_data(args):
    idx, dset = args
    return dset[idx]

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