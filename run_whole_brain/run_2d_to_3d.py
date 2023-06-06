import os, torch
from tqdm import trange
import numpy as np
import os
# import pnumpy as pn
# pn.enable()
from datetime import datetime
from torch.multiprocessing import Process

from kornia.filters.median import median_blur

def main():
    ## Previous step: 2D Unet
    dir_n = 'flow_2d'
    pair_tag = 'pair15'
    brain_tag = 'L73D766P4' # L73D766P9
    data_r = '/lichtman/ziquanw/Lightsheet/results/P4'
    data_r = '%s/%s' % (data_r, pair_tag)
    data_r = '%s/%s' % (data_r, brain_tag)
    fdir_yx_plane = '%s/%s' % (data_r, dir_n)
    img_r = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/stitched' % (pair_tag, brain_tag)
    assert os.path.exists(fdir_yx_plane), "You need to run 2D Unet first"
    ## Current step:
    device = 'cuda:1'
    dir_n = 'flow_2d'
    fs = sort_fs(os.listdir(img_r), get_depth_from_filename)
    deepest_i = get_depth_from_filename(fs[-1])
    save_r = '%s/%s' % (data_r, dir_n)
    os.makedirs(save_r, exist_ok=True)
    simulate_grad_z(save_r, fdir_yx_plane, (2, 0, 1), deepest_i, device)

def simulate_grad_z(save_r, fdir_yx_plane, ipm, deepest_i, device):
    stagen = 7
    filter_size = 3
    filter = median_blur
    while True:
        pre_yx_flow = None
        fs = sort_fs(os.listdir(fdir_yx_plane), get_depth_from_filename)
        paths = [os.path.join(fdir_yx_plane, f) for f in fs]
        save_processes = []
        not_done_num = sum([1 for path in paths if not os.path.exists(os.path.join(save_r, path.split('/')[-1]))])
        if not_done_num == 0: break
        for pi in range(len(paths)):
            path = paths[pi]
            i = get_depth_from_filename(fs[pi])
            if os.path.exists(os.path.join(save_r, path.split('/')[-1])): continue
            print(datetime.now(), "Load slice %s" % path.split('/')[-1])
            next_fn = set_depth_in_filename(fs[pi], i+2 if i < deepest_i else deepest_i+1)
            next_path = os.path.join(fdir_yx_plane, next_fn)
            if not os.path.exists(next_path): continue
            prev_fn = set_depth_in_filename(fs[pi], i if i > 0 else 1)
            prev_path = os.path.join(fdir_yx_plane, prev_fn)
            if not os.path.exists(prev_path): continue
            if i == 0:
                yf = np.load(path).transpose(ipm) 
                yx_flow = yf[:2]
                cellprob = yf[2]
            elif pre_yx_flow is None:
                pre_yx_flow = np.load(prev_path).transpose(ipm)[:2]
            if i < deepest_i:
                next_yf = np.load(next_path).transpose(ipm) 
                next_yx_flow = next_yf[:2]
                next_cellprob = next_yf[2]
            grad_Z = np.zeros_like(yf[0])
            if i == 0: pass
            elif i == deepest_i: pass
            else: 
                print(datetime.now(), "Do median filter pyramid")
                grad_Z = torch.from_numpy(grad_Z[np.newaxis, np.newaxis]).to(device)
                pre = (pre_yx_flow**2).sum(axis=0)**0.5
                pre = torch.from_numpy(pre[np.newaxis, np.newaxis]).to(device)
                next = (next_yx_flow**2).sum(axis=0)**0.5
                next = torch.from_numpy(next[np.newaxis, np.newaxis]).to(device)
                grad_Z += (next - pre)
                for _ in range(stagen):
                    pre_nextstage = filter(pre, filter_size)
                    next_nextstage = filter(next, filter_size)
                    grad_Z += (next - pre_nextstage)
                    grad_Z += (next_nextstage - pre)
                    pre = pre_nextstage
                    next = next_nextstage
                grad_Z /= (1 + stagen*2)
                grad_Z *= (torch.from_numpy(cellprob[np.newaxis, np.newaxis]).to(device) > 0)
                grad_Z = grad_Z.squeeze().numpy()
            dP = np.stack((grad_Z, yx_flow[0], yx_flow[1], cellprob),
                        axis=0) # (dZ, dY, dX))
            print(datetime.now(), "Save slice %s" % path.split('/')[-1])
            save_process = Process(target=save_output, args=(os.path.join(save_r, path.split('/')[-1]), dP))
            save_process.start()
            save_processes.append(save_process)
            for p in save_processes:
                if not p.is_alive(): del p
            pre_yx_flow = yx_flow
            yx_flow = next_yx_flow
            cellprob = next_cellprob
  
        for p in save_processes:
            if p.is_alive(): p.join()

def save_output(save_fn, prob):
    print(datetime.now(), "Save output to %s" % save_fn.split('/')[-1])
    np.save(save_fn, prob)

def sort_fs(fs, get_i):
    out = [0 for _ in range(len(fs))]
    for fn in fs:
        i = get_i(fn)
        out[i] = fn
    return out

def get_depth_from_filename(fn):
    return int(fn.split('_')[1])-1

def set_depth_in_filename(fn, depth):
    out = ''
    for i, item in enumerate(fn.split('_')):
        if i == 1: out += '%04d_' % depth
        else: out += item + '_'
    return out[:-1]

if __name__ == "__main__":
    main()