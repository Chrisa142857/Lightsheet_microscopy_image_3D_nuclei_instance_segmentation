import numpy as np
from datetime import datetime
import time
import torch, os, copy
from tqdm import trange, tqdm

import nibabel as nib

def main():
    # r='/cajal/ACMUSERS/ziquanw/Lightsheet/results/P14'
    r='/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4'
    p='pair6'
    b='220416_L57D855P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_09-52-07'
    for t in os.listdir(f'{r}/{p}/{b}'):
        assert 'Ultra' in t, t
        for f in os.listdir(f'{r}/{p}/{b}/{t}'):
            if 'binary_mask.zip' not in f: continue
            # if os.path.exists(f'{r}/{p}/{b}/{t}/{f}'.replace('.zip', '.nii.gz')): continue
            print(datetime.now(), f"Convert {p}/{b}/{t}/{f}")
            convert_torch2nii(f'{r}/{p}/{b}/{t}/{f}')
    # '''
    # convert coordinate to bounding box
    # '''
    # for p in os.listdir(r):
    #     if 'pair' not in p: continue
    #     for b in os.listdir(f'{r}/{p}'):
    #         if b[0] == 'L': continue
    #         if b == '220805_L74D769P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_15-51-36.manyZstitch': continue
    #         for t in os.listdir(f'{r}/{p}/{b}'):
    #             assert 'Ultra' in t, t
    #             coord_to_bbox_one_tile(p, b, t)

    # '''
    # convert binary_mask.zip to binary_mask.nii.gz
    # '''
    # for p in os.listdir(r):
    #     if 'pair' not in p: continue
    #     for b in os.listdir(f'{r}/{p}'):
    #         if b[0] == 'L': continue
    #         if b == '220805_L74D769P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_15-51-36.manyZstitch': continue
    #         for t in os.listdir(f'{r}/{p}/{b}'):
    #             assert 'Ultra' in t, t
    #             for f in os.listdir(f'{r}/{p}/{b}/{t}'):
    #                 if 'binary_mask.zip' not in f: continue
    #                 # if os.path.exists(f'{r}/{p}/{b}/{t}/{f}'.replace('.zip', '.nii.gz')): continue
    #                 print(datetime.now(), f"Convert {p}/{b}/{t}/{f}")
    #                 convert_torch2nii(f'{r}/{p}/{b}/{t}/{f}')
                    

def convert_torch2nii(fn):
    dratio = 0.3
    m = torch.load(fn).float()
    m = torch.nn.functional.interpolate(m[None, None], scale_factor=[dratio,dratio,dratio])[0,0]
    m = m.numpy()
    nib.save(nib.Nifti1Image(m, np.eye(4)), fn.replace('.zip', '.nii.gz'))

def coord_to_bbox_one_tile(ptag='pair4', btag='220904_L35D719P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_21-49-38', tile_name='UltraII[%02d x %02d]'):
    device = 'cuda:1'
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}/{btag}'
    # tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path)])
    # root = result_path + '/UltraII[%02d x %02d]'
    root = f'{result_path}/{tile_name}'
    # stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    # ncol, nrow = tile_loc.max(0)+1
    print(datetime.now(), "Start", ptag, btag.split('_')[1])
    # for _j in range(nrow):
    #     for _i in range(ncol):
    #         k = f'{_i}-{_j}'
    for stack_name in os.listdir(root):
        if 'instance_center' not in stack_name: continue
        bboxfn = f"{root}/{stack_name.replace('instance_center', 'instance_bbox')}"
        coordfn = f"{root}/{stack_name.replace('instance_center', 'instance_coordinate')}"
        volfn = f"{root}/{stack_name.replace('instance_center', 'instance_volume')}"
        if not os.path.exists(bboxfn) and os.path.exists(coordfn):
            _vol = torch.load(volfn).long()
            _coord = torch.load(coordfn)
            print(f'{datetime.now()}: Get bounding box of NIS of tile {tile_name} {stack_name}')
            bbox = coord_to_bbox(_coord, _vol.to(device), device).cpu()
            print(f'{datetime.now()}: Done with {len(bbox)} bboxes')
            torch.save(bbox, bboxfn)


def coord_to_bbox(coord, vol, device):
    if (vol == 0).any():
        print('Exist NIS has zero volume')
        valid_nis = vol!=0
        vol_cumsum = vol.cumsum(0)
        vol_cumsum = [0] + vol_cumsum.tolist()
        valid_nis = torch.where(valid_nis)[0]
        valid_coord = []
        for i in valid_nis:
            valid_coord.append(torch.arange(vol_cumsum[i], vol_cumsum[i+1]))
            assert valid_coord[-1].shape[0] == vol[i], f'{i}: {valid_coord[-1].shape[0]} != {vol[i]}, {vol_cumsum[i-1]}, {vol_cumsum[i]}, {vol_cumsum[i+1]}'

        valid_coord = torch.cat(valid_coord)
        vol = vol[valid_nis]
        coord = coord[valid_coord]

    coord_list = torch.tensor_split(coord, vol.cumsum(0)[:-1].cpu(), dim=0)
    lrange = 100000
    bboxs = []
    for i in range(0, len(coord_list), lrange):
        coord = torch.nn.utils.rnn.pad_sequence(coord_list[i:i+lrange], batch_first=True).to(device)
        assert coord.shape[1] == vol[i:i+lrange].max(), f'{coord.shape[1]} != {vol[i:i+lrange].max()}'
        coord_sort = coord.sort(dim=1, descending=True)[0]
        coord_min = coord_sort[torch.arange(coord_sort.shape[0]), vol[i:i+lrange]-1, :] # N x 3
        coord_max = coord_sort[:, 0] # N x 3
        assert (coord_min!=0).any(), 'coord_min is all zero'
        bbox = torch.cat([coord_min, coord_max], 1) # N x 6
        bboxs.append(bbox)
    bboxs = torch.cat(bboxs)
    return bboxs

if __name__ == '__main__': main()