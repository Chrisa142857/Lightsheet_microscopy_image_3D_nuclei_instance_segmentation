import torch
from utils import imread
import os
import re
import numpy as np
from multiprocessing import Pool
from tqdm import trange, tqdm
from datetime import datetime

def main(data_root, pair_tag, brain_tag, img_tag="C2_"):
    seg_pad = (4, 20, 20) # hard coded in CPP
    seg_root = f"/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{pair_tag}/{brain_tag}"
    img_root = f"/{data_root}/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched"
    save_root = f"/cajal/ACMUSERS/ziquanw/Lightsheet/statistics/{pair_tag}"
    os.makedirs(save_root, exist_ok=True)
    img_paths, _ = listdir_sorted(img_root, img_tag)
    seg_paths, sortks = listdir_sorted(seg_root, "NIScpp", ftail="seg.zip", sortkid=3)
    seg_depths = [sortks[i+1]-sortks[i] for i in range(len(sortks)-1)]
    seg_depths += [torch.load(seg_paths[-1]).shape[0]]
    seg_whole_shape = [seg_depths[-1]+sortks[-1]]
    img_whole_shape = [len(img_paths)]
    scale_r = [i/s for s, i in zip(seg_whole_shape, img_whole_shape)]
    stack_paths = []
    for i in range(len(seg_paths)):
        zmin = sortks[i]
        load_path = []
        for j in range(seg_depths[i]):
            j = j + zmin
            img_path = img_paths[int(scale_r[0]*j)]
            load_path.append([img_path])
        stack_paths.append(load_path)

    instance_statistic = {}
    stitch_remap = torch.load(f"{seg_root}/{brain_tag}_remap.zip") # old id to new id map
    total_vol = {}
    total_center = {}
    total_pt = {}
    for i in trange(len(seg_paths), desc=f"Start statistics for {pair_tag} {brain_tag} of {img_tag}intensity"):
        seg_path = seg_paths[i]
        zmin = sortks[i]
        print(datetime.now(), f"zmin: {zmin}, zmax: {zmin+seg_depths[i]}")
        label_path = seg_path.replace('seg.zip', 'instance_label.zip')
        vol_path = seg_path.replace('seg.zip', 'instance_volume.zip')
        ct_path = seg_path.replace('seg.zip', 'instance_center.zip')
        pt_path = seg_path.replace('seg.zip', 'contour.zip')
        label = torch.load(label_path)
        vol = torch.load(vol_path)
        pt = torch.load(pt_path)
        ct = torch.load(ct_path)
        pt[:, 0] = pt[:, 0] - seg_pad[0]
        pt[:, 1] = pt[:, 1] - seg_pad[1]
        pt[:, 2] = pt[:, 2] - seg_pad[2]
        z, y, x = pt.T
        # z = z + zmin
        assert z.min() >= 0 and z.max() < seg_depths[i], f"mask depth {seg_depths[i]}, while z.min(): {z.min()} and z.max(): {z.max()}"
        load_path = stack_paths[i]
        print(datetime.now(), f"Load images")
        with Pool(10) as p:
            img_stack = list(p.starmap(imread, load_path))
        img_stack = torch.from_numpy(np.stack(img_stack).astype(np.int32))
        print(datetime.now(), f"Start statistics:")
        pv = 0
        # for l, v in zip(label, vol):
        for j in trange(len(label), desc='Get intensity of each instance'):
            l = label[j]
            assert l.item()>0, l
            l = l.item()
            v = vol[j]
            if v < 10:
                pv = v
                continue
            c = ct[j]
            if l in stitch_remap[0]:
                l = stitch_remap[1, stitch_remap[0]==l]

            if l not in instance_statistic:
                instance_statistic[l] = img_stack[z[pv:v], y[pv:v], x[pv:v]]
            else:
                instance_statistic[l] = torch.cat([instance_statistic[l], img_stack[z[pv:v], y[pv:v], x[pv:v]]])
            
            if l not in total_vol:
                total_vol[l] = v
            else:
                total_vol[l] = total_vol[l] + v

            if l not in total_center:
                total_center[l] = c
            else:
                total_center[l] = (total_center[l] + c) / 2
            
            if l not in total_pt:
                total_pt[l] = pt[pv:v]
            else:
                total_pt[l] = torch.cat([total_pt[l], pt[pv:v]])
            
            pv = v
    print(datetime.now(), f"Save")
    total_label = torch.LongTensor(list(total_pt.keys()))
    torch.save(total_label, f"{save_root}/{brain_tag}_nis_label.zip")
    torch.save(torch.stack(list(total_center.values())), f"{save_root}/{brain_tag}_nis_center.zip")
    torch.save(torch.stack(list(total_vol.values())), f"{save_root}/{brain_tag}_nis_volume.zip")
    torch.save(list(total_pt.values()), f"{save_root}/{brain_tag}_nis_coordinate.zip")
    torch.save(list(instance_statistic.values()), f"{save_root}/{brain_tag}_nis_{img_tag}intensity.zip")


def listdir_sorted(path, tag, ftail='_stitched.tif', sortkid=1):
    fs = os.listdir(path)
    fs = [os.path.join(path, f) for f in fs if tag in f and f.endswith(ftail)]
    ks = []
    for f in fs:
        k = f.split('/')[-1].split('_')[sortkid]
        k = int(re.sub("[^0-9]", "", k))
        ks.append(k)
    orgks = ks.copy()
    ks.sort()
    sorted_fs = []
    for k in ks:
        sorted_fs.append(fs[orgks.index(k)])
        
    return sorted_fs, ks


if __name__=="__main__":
    data_root = 'lichtman'
    pair_tag = 'pair19'
    brain_tag = 'L79D769P5'
    img_tag = 'C2_'
    main(data_root, pair_tag, brain_tag, img_tag)

    data_root = 'lichtman'
    pair_tag = 'pair19'
    brain_tag = 'L79D769P8'
    main(data_root, pair_tag, brain_tag, img_tag)
