import tifffile as tif
import os, torch
from tqdm import tqdm, trange
import sys, time
import numpy as np
from torch.utils.data import DataLoader,Dataset
from unet import NISModel
import utils
# from multiprocessing import Pool
from datetime import datetime
# from torch.multiprocessing import Process
import multiprocessing as mp
from kornia.filters.median import median_blur
num_cpus = mp.cpu_count()-1
print(datetime.now(), "Pipeline start with %d CPUs" % (num_cpus+1))

def main():
    print(datetime.now(), f"Start python program {sys.argv}", flush=True)
    # 2D to 3D: one chunk has 8 slices, based on the RAM limit. Different number has no effect to results
    slicen_3d = 8
    ##
    dir_n = 'flow_3d'
    # brain_tag = 'L73D766P4' # L73D766P9
    brain_tag = sys.argv[1]
    pair_tag = sys.argv[2] # 'pair15'
    save_r = '/lichtman/ziquanw/Lightsheet/results/P4'
    std_resolution = (2.5, .75, .75)
    in_resolution = (4, .75, .75)
    scale_r = [i/s for i, s in zip(in_resolution, std_resolution)]
    trained_model = 'downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_22_42_54.153497_epoch_21'
    device = torch.device('cuda:%d' % int(sys.argv[3]))
    model = NISModel(device=device, pretrained_model=trained_model)
    save_r = '%s/%s' % (save_r, pair_tag)
    os.makedirs(save_r, exist_ok=True)
    save_r = '%s/%s' % (save_r, brain_tag)
    os.makedirs(save_r, exist_ok=True)
    save_r = '%s/%s' % (save_r, dir_n)
    os.makedirs(save_r, exist_ok=True)
    img_r = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/stitched' % (pair_tag, brain_tag)
    mask_tile = utils.MaskTiledImg(maskn='/lichtman/Felix/Lightsheet/P4/%s/output_%s/registered/%s_MASK_topro_25_all.nii' % (pair_tag, brain_tag, brain_tag), img_zres=std_resolution[0])
    model.mask_tile = mask_tile
    eval_tag = '_C1_'
    fs = [os.path.join(img_r, f) for f in os.listdir(img_r) if eval_tag in f and f.endswith('.tif')]
    orig_fs = fs.copy()
    depth_start = 1
    for f in orig_fs:
        fs[filename_to_depth(f, depth_start)] = f
    start = 0
    end = -1
    dset = BrainSliceDataset(fs[start:end]) if end > 0 else BrainSliceDataset(fs[start:])
    # dloader = DataLoader(dset,batch_size=1,shuffle=False,num_workers=8,collate_fn=collate_fn_pass)
    # rescaled_chunk_depth = torch.nn.functional.interpolate(torch.zeros(1, 1, slicen_3d, 3, 3), scale_factor=scale_r, mode='nearest-exact').squeeze().shape[0]
    si = 1
    flow_2d = []
    pre_final_yx_flow, pre_last_second = None, None
    resampled_si = 0
    # process_2d_to_3d = None
    print(datetime.now(), "Load slice %d" % (start + si))
    for data in tqdm(dset, desc="Run 2D Unet from slice %d to %d" % (start, end)):
        img, fn = data
        model.mask_tile.z = filename_to_depth(fn, depth_start)
        if int(model.mask_tile.z/model.mask_tile.zratio) >= len(model.mask_tile.mask): 
            print(datetime.now(), "Brain mask end, break out")
            break
        print(datetime.now(), "Input to model")
        prob = model.get_prob(img, diameter=None, batch_size=100, channels=[0,0])
        flow_2d.append(prob)
        if len(flow_2d) == slicen_3d:
            flow_2d = preproc_flow2d(np.stack(flow_2d, axis=0), pre_final_yx_flow, scale_r)
            resampled_si = one_chunk_2d_to_3d(resampled_si, flow_2d, pre_last_second, save_r, fn, device)
            pre_final_yx_flow = flow_2d[:, -1]
            pre_last_second = flow_2d[:2, -2]
            print(datetime.now(), f"Keep prev chunk's last two slices, shapes are {pre_final_yx_flow.shape}, {pre_last_second.shape}")
            flow_2d = []
        si += 1
        print(datetime.now(), "Load slice %d" % (start + si))

    if len(flow_2d) > 0:
        print(datetime.now(), "Start 2D to 3D")
        flow_2d = preproc_flow2d(np.stack(flow_2d, axis=0), pre_final_yx_flow, scale_r)
        one_chunk_2d_to_3d(resampled_si, flow_2d, pre_last_second, save_r, fn, device)
        print(datetime.now(), "Done 2D to 3D")

def preproc_flow2d(flow_2d, pre_final_yx_flow, scale_r):
    flow_2d = torch.from_numpy(flow_2d.transpose((3, 0, 1, 2)))
    print(datetime.now(), f"Resample to standard resolution from shape {flow_2d.shape}")
    flow_2d = torch.nn.functional.interpolate(flow_2d.unsqueeze(0), scale_factor=scale_r, mode='nearest-exact').squeeze()#.numpy()
    if pre_final_yx_flow is not None: 
        # pre_final_yx_flow = pre_final_yx_flow.permute((2, 0, 1))
        # pre_last_second = pre_last_second.permute((2, 0, 1))
        flow_2d = torch.cat([pre_final_yx_flow.unsqueeze(1), flow_2d], dim=1)
    print(datetime.now(), f"Resample to shape {flow_2d.shape}")
    return flow_2d

def one_chunk_2d_to_3d(resampled_si, flow_2d, pre_last_second, save_r, fn, device):
    print(datetime.now(), "Start 2D to 3D at resampled slice %d" % resampled_si)
    save_processes = []
    for i in range(flow_2d.shape[1]-1):
        yx_flow = flow_2d[:2, i]
        cellprob = flow_2d[2, i]
        next_yx_flow = flow_2d[:2, i+1]
        if i > 0:
            pre_yx_flow = flow_2d[:2, i-1]
        else:
            pre_yx_flow = pre_last_second
        ##########################################################################
        dP = sim_grad_z(i, yx_flow, cellprob, pre_yx_flow, next_yx_flow, device=device)
        ##########################################################################
        print(datetime.now(), "Save 3D flow map %d" % resampled_si)
        save_path = '%s/%s' % (save_r, fn.split('/')[-1].replace('.tif', '_resample%04d.npy' % resampled_si))
        if len(save_processes) > 0: save_processes[-1].join()
        if len(save_processes) > num_cpus: 
            for p in save_processes: p.join(); time.sleep(0.1); p.terminate()
            save_processes = []
        save_processes.append(mp.Process(target=np.save, args=(save_path, dP)))
        save_processes[-1].start()
        resampled_si += 1
    print(datetime.now(), "Done 2D to 3D")
    return resampled_si

def sim_grad_z(i, yx_flow, cellprob, pre_yx_flow, next_yx_flow, device='cpu'):
    stagen = 7
    filter_size = 3
    filter = median_blur
    grad_Z = torch.zeros_like(cellprob)
    if pre_yx_flow is None: pass
    elif next_yx_flow is None: pass
    else: 
        print(datetime.now(), "Do median filter pyramid for flow map %d" % i)
        pre_yx_flow = pre_yx_flow.to(device)
        next_yx_flow = next_yx_flow.to(device)
        inside = (cellprob.unsqueeze(0).unsqueeze(0) > 0).to(device)
        pre = ((pre_yx_flow**2).sum(0)**0.5).unsqueeze(0).unsqueeze(0)
        next = ((next_yx_flow**2).sum(0)**0.5).unsqueeze(0).unsqueeze(0)
        grad_Z = next - pre
        for _ in range(stagen):
            pre_nextstage = filter(pre, filter_size)
            next_nextstage = filter(next, filter_size)
            grad_Z = grad_Z + (next - pre_nextstage)
            grad_Z = grad_Z + (next_nextstage - pre)
            pre = pre_nextstage
            next = next_nextstage
        grad_Z = grad_Z / (1 + stagen*2)
        grad_Z = grad_Z * inside
        grad_Z = grad_Z.squeeze().cpu()
    dP = torch.stack((grad_Z, yx_flow[0], yx_flow[1], cellprob),
                dim=0) # (dZ, dY, dX))
    return dP.numpy()


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