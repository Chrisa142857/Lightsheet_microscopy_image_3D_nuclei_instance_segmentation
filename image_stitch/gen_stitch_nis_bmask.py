import numpy as np
import os, torch, json
import nibabel as nib
from datetime import datetime
from tqdm import trange, tqdm
from PIL import Image
import copy

OVERLAP_R = 0.2
zratio = 2.5/4

def main():
    # p='pair5'
    # b='220423_L57D855P5_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_09-02-27'
    # gen_stitch_nis_bmask(p, b)
    ################################################
    result_r='/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4'
    for p in os.listdir(result_r):
        if 'pair' not in p: continue
        for b in os.listdir(f'{result_r}/{p}'):
            if b[0] == 'L': continue
            if b == '220423_L57D855P5_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_09-02-27': continue
            gen_stitch_nis_bmask(p, b)

def gen_stitch_nis_bmask(ptag, btag, device='cuda:0'):
    overlap_r = OVERLAP_R
    result_r='/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4'
    save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/{ptag}/{btag.split("_")[1]}'
    if os.path.exists(f'{save_path}/binary_mask_stitched'): return
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}/{btag}'
    rm_labels = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/{ptag}/{btag.split("_")[1]}/doubled_NIS_label/{btag.split("_")[1]}_doubled_label.zip'
    if not os.path.exists(rm_labels): return
    rm_labels = torch.load(rm_labels)
    bmask_dict = {}
    for t in os.listdir(f'{result_r}/{ptag}/{btag}'):
        assert 'Ultra' in t, t
        i, j = t.split(' x ')
        i, j = int(i[-2:]), int(j[:2])
        k = f'{i}-{j}'
        if k in rm_labels:
            rm_label = rm_labels[k]
        else:
            rm_label = []
        print(t)
        bmask = load_bmask(f'{result_r}/{ptag}/{btag}/{t}', rm_label, device)
        if len(bmask) == 0: continue
        bmask = bmask.float()
        bmask = torch.nn.functional.interpolate(bmask[None, None], scale_factor=[zratio,1,1])[0,0]>0
        print(bmask.shape)
        bmask_dict[k] = bmask

    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path)])
    ncol, nrow = tile_loc.max(0)+1
    assert len(tile_loc) == nrow*ncol, f'tile of raw data is not complete, tile location: {tile_loc}'

    root = result_path + '/UltraII[%02d x %02d]'
    stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    stack_names = sort_stackname(stack_names)
    # print(stack_names)

    for stack_name in stack_names:
        meta_name = stack_name.replace('instance_center', 'seg_meta')
        zstart = int(stack_name.split('zmin')[1].split('_')[0])
        zstart = int(zstart*zratio)
        seg_shape = torch.load(f'{root % (0, 0)}/{meta_name}')
        zend = int(zstart + seg_shape[0].item()*zratio)

    zstart = 0 
    tform_xy_max = [0.05*seg_shape[1], 0.05*seg_shape[2]]

    tile_w = int(seg_shape[1].item()*(1-overlap_r))
    tile_h = int(seg_shape[2].item()*(1-overlap_r))
    overlap_w = int(seg_shape[1].item()*overlap_r)
    overlap_h = int(seg_shape[2].item()*overlap_r)
    
    os.makedirs(f'{save_path}/binary_mask_stitched', exist_ok=True)
    pre_startx, pre_starty = {}, {}

    tform_stack_coarse = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_coarse.json', 'r', encoding='utf-8'))
    tform_stack_refine = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine.json', 'r', encoding='utf-8'))
    if os.path.exists(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine_ptreg.json'):
        tform_stack_ptreg = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine_ptreg.json', 'r', encoding='utf-8'))
    else:
        tform_stack_ptreg = None
    '''
    Get lefttop location of tile stack after stitching
    '''
    tile_lt_loc = {
        f'{i}-{j}': [i*seg_shape[1]*(1-overlap_r), j*seg_shape[2]*(1-overlap_r)] for i in range(ncol) for j in range(nrow)
    }
    ## Coarse
    tformed_tile_lt_loc = {0: copy.deepcopy(tile_lt_loc)}
    for k in tform_stack_coarse:
        tz, tx, ty = tform_stack_coarse[k]
        tformed_tile_lt_loc[0][k][0] = tformed_tile_lt_loc[0][k][0] + tx
        tformed_tile_lt_loc[0][k][1] = tformed_tile_lt_loc[0][k][1] + ty
    ## Refine
    tformed_tile_lt_loc_refine = {zi: copy.deepcopy(tformed_tile_lt_loc[0]) for zi in range(zstart, zend)}
    for zi in range(zstart, zend):
        for k in tform_stack_refine[zi]:
            tx, ty = tform_stack_refine[zi][k]
            tformed_tile_lt_loc_refine[zi][k][0] = tformed_tile_lt_loc_refine[zi][k][0] + tx
            tformed_tile_lt_loc_refine[zi][k][1] = tformed_tile_lt_loc_refine[zi][k][1] + ty
    new_header, affine_m = init_nib_header()
    print(zstart, zend, len(tformed_tile_lt_loc_refine))
    for zi in tqdm(tformed_tile_lt_loc_refine, desc='Apply transformation'):
        valid_wsi = True
        for ijstr in tformed_tile_lt_loc_refine[zi]:
            i, j = ijstr.split('-')
            i, j = int(i), int(j)
            tz = tform_stack_coarse[f'{i}-{j}'][0]
            tz = int(np.around(tz))
            wsii = zi - tz
            valid_wsi = wsii>=0 and wsii<len(tformed_tile_lt_loc_refine)
            if not valid_wsi: break
            
        if not valid_wsi: continue
        wsi = torch.zeros((tile_w*ncol+overlap_w+1, tile_h*nrow+overlap_h+1)).bool()
        for ijstr in tformed_tile_lt_loc_refine[zi]:
            if ijstr not in bmask_dict: continue
            if ijstr not in pre_startx:
                pre_startx[ijstr] = None
                pre_starty[ijstr] = None
            
            i, j = ijstr.split('-')
            i, j = int(i), int(j)
                
            tz = tform_stack_coarse[f'{i}-{j}'][0]
            tz = int(np.around(tz))
            wsii = zi - tz
            if wsii >= bmask_dict[ijstr].shape[0]:
                continue
            
            tile_bmask = bmask_dict[ijstr][wsii]

            if abs(tform_stack_refine[zi][ijstr][0]) > tform_xy_max[0] or abs(tform_stack_refine[zi][ijstr][1]) > tform_xy_max[1]:
                if pre_startx[ijstr] is not None: 
                    startx, starty = pre_startx[ijstr], pre_starty[ijstr]
                else:
                    for zii in range(zi, len(tform_stack_refine)):
                        if abs(tform_stack_refine[zii][ijstr][0]) <= tform_xy_max[0] and abs(tform_stack_refine[zii][ijstr][1]) <= tform_xy_max[1]:
                            zii = max(zii + tz, 0)
                            zii = min(max(list(tformed_tile_lt_loc_refine.keys())), zii)
                            startx, starty = tformed_tile_lt_loc_refine[zii][ijstr]
                            break
            else:
                startx, starty = tformed_tile_lt_loc_refine[zi][ijstr]
            
            pre_startx[ijstr] = startx
            pre_starty[ijstr] = starty

            if tform_stack_ptreg is not None:
                if ijstr in tform_stack_ptreg[zi]:
                    tx, ty = tform_stack_ptreg[zi][ijstr]
                    startx = startx + tx
                    starty = starty + ty

            startx, endx = int(startx), tile_bmask.shape[0] + int(startx)
            starty, endy = int(starty), tile_bmask.shape[1] + int(starty)
            if startx < 0:
                tile_bmask = tile_bmask[-startx:]
                startx = 0
            if starty < 0:
                tile_bmask = tile_bmask[:, -starty:]
                starty = 0
            if endx > wsi.shape[0]: 
                tile_bmask = tile_bmask[:-(endx-wsi.shape[0])]
                endx = wsi.shape[0]

            if endy > wsi.shape[1]: 
                tile_bmask = tile_bmask[:, :-(endy-wsi.shape[1])]
                endy = wsi.shape[1]

            cur_tile = wsi[startx:endx, starty:endy]

            wsi[startx:endx, starty:endy] = torch.logical_or(tile_bmask, cur_tile)
        wsi = wsi.float().numpy()
        save_fn = f'{save_path}/binary_mask_stitched/{btag.split("_")[1]}_TOPRO_C01_Z{zi:04d}.ptreg_stitch.bmask.nii.gz'
    
        nib.save(nib.Nifti1Image(wsi[..., None].transpose(1,0,2), affine_m, header=new_header), save_fn)


def load_bmask(result_r, rm_label, device):
    stack_names = [fn for fn in os.listdir(result_r) if 'binary_mask.zip' in fn]
    stack_names = sort_stackname(stack_names)
    lrange = 1000

    bmasks = []
    # print(stack_names)
    for sn in stack_names:
        print(datetime.now(), f"Loading {sn}")
        bmask = torch.load(f'{result_r}/{sn}')
        nis_coord = torch.load(f"{result_r}/{sn.replace('binary_mask', 'instance_coordinate')}")
        nis_vol = torch.load(f"{result_r}/{sn.replace('binary_mask', 'instance_volume')}").to(device)
        nis_label = torch.load(f"{result_r}/{sn.replace('binary_mask', 'instance_label')}").to(device)
        # print(datetime.now(), f"Removing doubled NIS")
        nis_vol_cumsum = torch.cat([torch.FloatTensor([0]).to(device), nis_vol.cumsum(0)]).long()
        nis_coord_padmask = torch.ones(len(nis_coord)).bool()
        nis_coord_padmask = torch.tensor_split(nis_coord_padmask, nis_vol_cumsum[1:-1].cpu())
        # print(bmask.shape, nis_coord.shape, nis_vol.shape, nis_vol_cumsum.shape, len(nis_coord_padmask))
        nis_coord_padmask = torch.nn.utils.rnn.pad_sequence(nis_coord_padmask, batch_first=True, padding_value=False)
        max_vol_arange = torch.arange(nis_coord_padmask.shape[1]).to(device)
    
        print("before remove doubled nis coordinate", nis_coord.shape)
        rm_ind = []
        if len(rm_label) > 0:
            rm_label = rm_label.to(device)
            for labeli in trange(0, len(nis_label), lrange, desc=f"{datetime.now()} Removing doubled NIS"):
                label_batch = nis_label[labeli:labeli+lrange]
                label2rm = label_batch[:, None] == rm_label[None, :]
                do_rm = label2rm.any(1)
                do_rm_ind = torch.arange(labeli, labeli+len(label_batch), device=device)[do_rm]
                do_rm_coord_ind_start = nis_vol_cumsum[do_rm_ind]
                do_rm_coord_ind = do_rm_coord_ind_start[:, None] + max_vol_arange[None, :]
                do_rm_coord_ind = do_rm_coord_ind[nis_coord_padmask[do_rm_ind.cpu()]]
                rm_ind.append(do_rm_coord_ind.cpu())
        
        if len(rm_ind) > 0:
            rm_ind = torch.cat(rm_ind)
            nis_coord = nis_coord[rm_ind].long()
            bmask[nis_coord[:, 0], nis_coord[:, 1], nis_coord[:, 2]] = 0

        print("removed doubled nis coordinate", nis_coord.shape)
        bmasks.append(bmask)
    if len(bmasks)>0:
        bmasks = torch.cat(bmasks)
    return bmasks
        
def init_nib_header():
    mask_fn = "/lichtman/Felix/Lightsheet/P4/pair15/output_L73D766P4/registered/L73D766P4_MASK_topro_25_all.nii"
    new_header = nib.load(mask_fn).header
    new_header['quatern_b'] = 0.5
    new_header['quatern_c'] = -0.5
    new_header['quatern_d'] = 0.5
    new_header['qoffset_x'] = -0.0
    new_header['qoffset_y'] = -0.0
    new_header['qoffset_z'] = 0.0
    affine_m = np.eye(4, 4)
    affine_m[:3, :3] = 0
    affine_m[0, 1] = -1
    affine_m[1, 2] = -1
    affine_m[2, 0] = 1
    return new_header, affine_m

def sort_stackname(stack_names):
    stack_z = []
    for stack_name in stack_names:
        stack_z.append(int(stack_name.split('zmin')[1].split('_')[0]))

    argsort = np.argsort(stack_z)
    return [stack_names[i] for i in argsort]


if __name__ == '__main__': main()