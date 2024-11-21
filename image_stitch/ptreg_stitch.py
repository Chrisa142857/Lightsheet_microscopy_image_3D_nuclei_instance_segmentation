import numpy as np
from datetime import datetime
import time
import torch, os, copy
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import trange, tqdm
from matplotlib import colormaps
import nibabel as nib
from PIL import Image
import json
from torchvision.ops._utils import _upcast
from torch_scatter import scatter_max
from cpd_translation import RigidRegistration
from get_bbox_apply_tform import get_bbox_with_tform

ZRANGE = 15
OVERLAP_R = 0.2
zratio = 2.5/4

def main():
    
    # stitch_tile_ij_lst = [
    #     [[1,4]],
    # ]
    # stitch_slice_ranges_lst = [
    #     [[0.7, 1]],
    # ]
    # ptags = [
    #     'pair4',
    # ]
    # btags = [
    #     '220902_L35D719P3_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-41-36'
    # ]
    # for stitch_tile_ij, stitch_slice_ranges, ptag, btag in zip(stitch_tile_ij_lst, stitch_slice_ranges_lst, ptags, btags):
    #     stitch_by_ptreg(stitch_tile_ij, stitch_slice_ranges, ptag, btag)

    # stitch_tile_ij_lst = [
    #     [[i,j] for i in range(4) for j in range(5) if i!=0 or j!=0],
    # ]
    # stitch_slice_ranges_lst = [
    #     [[0.7, 1] for _ in range(len(stitch_tile_ij_lst[0]))],
    # ]
    # stitch_slice_ranges_lst = [
    #     [[0, 1] for _ in range(len(stitch_tile_ij_lst[0]))],
    # ]
    # ptags = [
    #     'pair5',
    # ]
    # btags = [
    #     '220423_L57D855P5_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_09-02-27'
    # ]
    # for stitch_tile_ij, stitch_slice_ranges, ptag, btag in zip(stitch_tile_ij_lst, stitch_slice_ranges_lst, ptags, btags):
    #     stitch_by_ptreg(stitch_tile_ij, stitch_slice_ranges, ptag, btag)

    # ptags = [
    #     'pair8',
    # ]
    # btags = [
    #     '220430_L59D878P5_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_13-03-21'
    # ]
    # for stitch_tile_ij, stitch_slice_ranges, ptag, btag in zip(stitch_tile_ij_lst, stitch_slice_ranges_lst, ptags, btags):
    #     stitch_by_ptreg(stitch_tile_ij, stitch_slice_ranges, ptag, btag)


    stitch_tile_ij_lst = [
        [[2,4], [2,0], [1,0]],
        [[3,4], [0,1]],
        [[0,1], [0,2], [0,3], [0,4]]
    ]
    stitch_slice_ranges_lst = [
        [[0.7,1], [0,0.3], [0.5,1]],
        [[0,1], [0.7,1]],
        [[0,1], [0,1], [0,1], [0,1]],
    ]
    ptags = [
        'pair13',
        'pair14',
        'pair18'
    ]
    btags = [
        '220827_L69D764P6_OUT_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-47-13',
        '220722_L73D766P5_OUT_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_16-49-30',
        '220809_L77D764P8_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_09-52-21',
    ]
    for stitch_tile_ij, stitch_slice_ranges, ptag, btag in zip(stitch_tile_ij_lst, stitch_slice_ranges_lst, ptags, btags):
        stitch_by_ptreg(stitch_tile_ij, stitch_slice_ranges, ptag, btag)



def stitch_by_ptreg(stitch_tile_ij, stitch_slice_ranges,
    ptag='pair4',
    btag='220904_L35D719P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_21-49-38',
    device = 'cuda:0'
):    
    
    overlap_r = OVERLAP_R

    save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/{ptag}/{btag.split("_")[1]}'
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}/{btag}'
    root = result_path + '/UltraII[%02d x %02d]'
    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path) if 'Ultra' in fn])
    ncol, nrow = tile_loc.max(0)+1
    assert len(tile_loc) == nrow*ncol, f'tile of raw data is not complete, tile location: {tile_loc}'
    stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    stack_names = sort_stackname(stack_names)
    for stack_name in stack_names:
        meta_name = stack_name.replace('instance_center', 'seg_meta')
        zstart = int(stack_name.split('zmin')[1].split('_')[0])
        zstart = int(zstart*zratio)
        seg_shape = torch.load(f'{root % (0, 0)}/{meta_name}')
        zend = int(zstart + seg_shape[0].item()*zratio)
    
    stitch_slice_ranges = [[int(r[0]*zend), int(r[1]*zend)] for r in stitch_slice_ranges]
    zstart = 0 

    stack_nis_bbox, stack_nis_label, tformed_tile_lt_loc_refined = get_bbox_with_tform(ptag, btag)
    # tform_stack_coarse = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_coarse.json', 'r', encoding='utf-8'))
    # tform_stack_refine = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine.json', 'r', encoding='utf-8'))

    # '''
    # Get lefttop location of tile stack after stitching
    # '''
    # tile_lt_loc = {
    #     f'{i}-{j}': [i*seg_shape[1]*(1-overlap_r), j*seg_shape[2]*(1-overlap_r)] for i in range(ncol) for j in range(nrow)
    # }
    # zstart = 0
    # ## Coarse
    # tformed_tile_lt_loc = {0: copy.deepcopy(tile_lt_loc)}
    # for k in tform_stack_coarse:
    #     tz, tx, ty = tform_stack_coarse[k]
    #     tformed_tile_lt_loc[0][k][0] = tformed_tile_lt_loc[0][k][0] + tx
    #     tformed_tile_lt_loc[0][k][1] = tformed_tile_lt_loc[0][k][1] + ty

    # ## Refine
    # tformed_tile_lt_loc_refined = {zi: copy.deepcopy(tformed_tile_lt_loc[0]) for zi in range(zstart, zend)}
    # for zi in range(zstart, zend):
    #     for k in tform_stack_refine[zi]:
    #         tx, ty = tform_stack_refine[zi][k]
    #         tformed_tile_lt_loc_refined[zi][k][0] = tformed_tile_lt_loc_refined[zi][k][0] + tx
    #         tformed_tile_lt_loc_refined[zi][k][1] = tformed_tile_lt_loc_refined[zi][k][1] + ty
    
    # '''
    # Get bbox 
    # '''
    # stack_nis_bbox = {}
    # stack_nis_label = {}
    # tform_xy_max = [0.05*seg_shape[1], 0.05*seg_shape[2]]
    # print(datetime.now(), "Load bounding box after transform")
    # for _j in range(nrow):
    #     for _i in range(ncol):
    #         k = f'{_i}-{_j}'
    #         tz = tform_stack_coarse[k][0]
    #         _bbox = []
    #         _label = []
    #         if k not in stack_nis_bbox: stack_nis_bbox[k] = []
    #         if k not in stack_nis_label: stack_nis_label[k] = []
    #         for stack_name in stack_names:
    #             zstart = int(stack_name.split('zmin')[1].split('_')[0])
    #             zstart = int(zstart*zratio)
    #             labelfn = f"{root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_label')}"
    #             label = torch.load(labelfn).long()
    #             bboxfn = f"{root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_bbox')}"
    #             assert os.path.exists(bboxfn), bboxfn
                    
    #             bbox = torch.load(bboxfn)
    #             bbox[:, 0] = bbox[:, 0] * zratio + zstart
    #             bbox[:, 3] = bbox[:, 3] * zratio + zstart
                    
    #             _bbox.append(bbox)
    #             _label.append(label)
    #         _bbox = torch.cat(_bbox)
    #         _label = torch.cat(_label)
    #         _bbox[:, 0] = _bbox[:, 0] + tz
    #         _bbox[:, 3] = _bbox[:, 3] + tz
    #         ct = (_bbox[:, :3] + _bbox[:, 3:]) / 2
    #         pre_refine_lt_x = None
    #         pre_refine_lt_y = None
    #         for zi in tformed_tile_lt_loc_refined.keys():
    #             ct_zmask = torch.where(ct[:, 0].long() == zi)[0]
    #             if len(ct_zmask) == 0: continue
    #             bbox = _bbox[ct_zmask].clone()
    #             label = _label[ct_zmask]
                
    #             if abs(tform_stack_refine[zi][k][0]) > tform_xy_max[0] or abs(tform_stack_refine[zi][k][1]) > tform_xy_max[1]:
    #                 if pre_refine_lt_x is not None:
    #                     refine_lt_x = pre_refine_lt_x
    #                     refine_lt_y = pre_refine_lt_y
    #                 else:
    #                     for zii in range(zi, len(tform_stack_refine)):
    #                         if abs(tform_stack_refine[zii][k][0]) <= tform_xy_max[0] and abs(tform_stack_refine[zii][k][1]) <= tform_xy_max[1]:
    #                             refine_lt_x = tformed_tile_lt_loc_refined[zii][k][0]
    #                             refine_lt_y = tformed_tile_lt_loc_refined[zii][k][1]
    #                             break
    #             else:
    #                 refine_lt_x = tformed_tile_lt_loc_refined[zi][k][0]
    #                 refine_lt_y = tformed_tile_lt_loc_refined[zi][k][1]
    #             pre_refine_lt_x = refine_lt_x
    #             pre_refine_lt_y = refine_lt_y
                    
    #             bbox[:, 1] = bbox[:, 1] + refine_lt_x
    #             bbox[:, 2] = bbox[:, 2] + refine_lt_y
    #             bbox[:, 4] = bbox[:, 4] + refine_lt_x
    #             bbox[:, 5] = bbox[:, 5] + refine_lt_y
    #             bbox = bbox.clip(min=0)
    #             stack_nis_bbox[k].append(bbox)
    #             stack_nis_label[k].append(label)
            
    #         stack_nis_bbox[k] = torch.cat(stack_nis_bbox[k])
    #         stack_nis_label[k] = torch.cat(stack_nis_label[k])
    '''
    Get centers
    '''
    corse_tformed_tile_lt_loc = tformed_tile_lt_loc_refined
    new_tile_pt_coarse = {}
    for k in stack_nis_bbox:
        bbox = stack_nis_bbox[k]
        ct = (bbox[:, :3] + bbox[:, 3:])/2
        new_tile_pt_coarse[k] = ct

    '''
    CPD Registration
    '''
    zstart = 0
    zrange = ZRANGE
    lrange = 1000
    ptreg_min_pt_num = 100
    gpu_cap_pt_num = 30000
    neighbor = [[-1, 0], [0, -1], [-1, -1], [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1]]
    tformed_tile_lt_loc = {zi: copy.deepcopy(corse_tformed_tile_lt_loc[zi]) for zi in range(zstart, zend)}
    refined_ks = [[_i, _j] for _i in range(ncol) for _j in range(nrow) if [_i, _j] not in stitch_tile_ij]
    _i, _j = refined_ks[0]
    new_tile_pt_refine = {f'{_i}-{_j}': new_tile_pt_coarse[f'{_i}-{_j}'] for _i, _j in refined_ks}
    
    if os.path.exists(f'{save_path}/doubled_NIS_label/{btag.split("_")[1]}_doubled_label.zip'):
        rm_label = torch.load(f'{save_path}/doubled_NIS_label/{btag.split("_")[1]}_doubled_label.zip')
        for k in rm_label:
            rm_label[k] = rm_label[k].to(device)
    else:
        rm_label = {}
    
    for k in tqdm(new_tile_pt_refine, desc=f'{datetime.now()} Load stitched tiles'):
        label = stack_nis_label[k].to(device)
        pt = new_tile_pt_refine[k]
        if k in rm_label:
            keep_ind = []
            for labeli in range(0, len(label), lrange):
                label_batch = label[labeli:labeli+lrange]
                label2rm = label_batch[:, None] == rm_label[k][None, :]
                do_rm = label2rm.any(1)
                keep_ind.append(torch.arange(labeli, labeli+len(label_batch), device=device)[torch.logical_not(do_rm)])
            if len(label) > 0:
                keep_ind = torch.cat(keep_ind)
                pt = pt[keep_ind.cpu()]
        new_tile_pt_refine[k] = pt

    tform_stack_ptreg = [{} for zi in range(zstart, zend)]
    for (i, j), (zstart, _zend) in zip(stitch_tile_ij, stitch_slice_ranges):
        pt_mov_tile = new_tile_pt_coarse[f'{i}-{j}'].clone()
        if _zend == -1: _zend = zend

        zslice = _zend-1
        while zslice > zstart:
            zup = max(0, zslice - zrange)
            zdown = zslice
            m1 = pt_mov_tile[:, 0].long()>=zup
            m2 = pt_mov_tile[:, 0].long()<zdown
            pt_mov_zmask = torch.where(torch.logical_and(m1, m2))[0]      
            pt_mov = pt_mov_tile[pt_mov_zmask, :].clone()

            pt_mov_w = seg_shape[1]
            pt_mov_h = seg_shape[2]

            pt_mov_lt_loc = tformed_tile_lt_loc[zslice][f'{i}-{j}']
            pt_tgt = []
            pmask_tgt = []
            mov_pmasks = []
            tgt_tile_id = []
            for pi, pj in neighbor:
                if f'{i+pi}-{j+pj}' in new_tile_pt_refine:
                    pt_lt_loc = tformed_tile_lt_loc[zslice][f'{i+pi}-{j+pj}']
                    pt = new_tile_pt_refine[f'{i+pi}-{j+pj}']
                    pt_zmask = torch.where(torch.logical_and(pt[:, 0].long()>=zup, pt[:, 0].long()<zdown))[0]      
                    pt = pt[pt_zmask, :]
                    tgt_pmasks = []
                    if pt.shape[0] > 0:
                        mov_pmask = []
                        pt_tgt.append(pt)
                        pt_w = seg_shape[1]
                        pt_h = seg_shape[2]
                        if pi < 0:
                            tgt_pmasks.append(pt[:, 1]>=(pt_lt_loc[0]+pt_w*(1-overlap_r)))
                            mov_pmask.append(pt_mov[:, 1]<=(pt_mov_lt_loc[0]+pt_mov_w*(overlap_r)))
                        elif pi > 0:
                            tgt_pmasks.append(pt[:, 1]<=(pt_lt_loc[0]+pt_w*(overlap_r)))
                            mov_pmask.append(pt_mov[:, 1]>=(pt_mov_lt_loc[0]+pt_mov_w*(1-overlap_r)))
                        if pj < 0:
                            tgt_pmasks.append(pt[:, 2]>=(pt_lt_loc[1]+pt_h*(1-overlap_r)))
                            mov_pmask.append(pt_mov[:, 2]<=(pt_mov_lt_loc[1]+pt_mov_h*(overlap_r)))
                        elif pj > 0:
                            tgt_pmasks.append(pt[:, 2]<=(pt_lt_loc[1]+pt_h*(overlap_r)))
                            mov_pmask.append(pt_mov[:, 2]>=(pt_mov_lt_loc[1]+pt_mov_h*(1-overlap_r)))

                        pmask = tgt_pmasks[0]
                        for m in tgt_pmasks[1:]:
                            pmask = torch.logical_and(pmask, m)

                        pmask_tgt.append(pmask)

                        pmask = mov_pmask[0]
                        for m in mov_pmask[1:]:
                            pmask = torch.logical_and(pmask, m)

                        mov_pmasks.append(pmask)

                        tgt_tile_id.append(f'{i+pi}-{j+pj}')

            if len(pmask_tgt) > 0:
                pt_tgt_ = torch.cat(pt_tgt)
                pmask_tgt = torch.cat(pmask_tgt)
                pt_tgt = pt_tgt_[pmask_tgt, :].clone()

                pmask_mov = mov_pmasks[0]
                for m in mov_pmasks:
                    pmask_mov = torch.logical_or(pmask_mov, m)
                pt_mov = pt_mov[pmask_mov, :]

            else:
                pt_tgt = pt_tgt_ = torch.zeros(0, 3)


            if (pt_mov.shape[0] < ptreg_min_pt_num or pt_tgt.shape[0] < ptreg_min_pt_num) and pt_mov.shape[0]*pt_tgt.shape[0] < gpu_cap_pt_num**2 and zup > 0:
                zrange = zrange + ZRANGE
                continue
            else:
                if pt_mov.shape[0] > 0 and pt_tgt.shape[0] > 0 and pt_mov.shape[0]*pt_tgt.shape[0] < gpu_cap_pt_num**2:
                    print(f'Tile moving ({i:02d},{j:02d})', f'Tile targets {tgt_tile_id}')
                    print(datetime.now(), f'Point registration from slice {zup} to {zdown}, moving pt # {pt_mov.shape[0]}, target pt # {pt_tgt.shape[0]}')
                    s_reg, R_reg, t_reg, correspondence = point_reg_cpd(pt_mov, pt_tgt, only_overlap=False, device=device, force_z0=True)
                    t_reg = t_reg.cpu()
                    print(datetime.now(), "Done translation:", t_reg.tolist())
                    # t_reg[0,0] = 0
                    ## Apply transformation to pt in a tile
                else:
                    t_reg = torch.zeros(1, 3)

                if len(stitch_tile_ij) > 1:
                    pt = pt_mov_tile[pt_mov_zmask, :].clone()
                    pt[:, 1:] = pt[:, 1:] + t_reg[:, 1:]
                    if pt_mov.shape[0] > 0 and pt_tgt.shape[0] > 0:
                        ## Remove matched points
                        min_dis = distance_mat(pt.to(device), pt_tgt_.to(device)).cpu()
                        mov_matched_index = torch.where(min_dis<=10)[0]
                        ###########################################################
                        mask = torch.ones(pt.shape[0]).bool()
                        mask[mov_matched_index] = False
                        pt = pt[mask]
                    if f'{i}-{j}' not in new_tile_pt_refine: 
                        new_tile_pt_refine[f'{i}-{j}'] = pt
                    else:
                        new_tile_pt_refine[f'{i}-{j}'] = torch.cat([new_tile_pt_refine[f'{i}-{j}'], pt])

                _, tx, ty = t_reg[0].tolist()
                for zi in range(zup, zdown):
                    tform_stack_ptreg[zi][f'{i}-{j}'] = [tx, ty]
                    # tformed_tile_lt_loc[zi][f'{i}-{j}'][0] = tformed_tile_lt_loc[zi][f'{i}-{j}'][0] + tx
                    # tformed_tile_lt_loc[zi][f'{i}-{j}'][1] = tformed_tile_lt_loc[zi][f'{i}-{j}'][1] + ty

        #             zslice = zslice + zrange
                zslice = zslice - zrange
                zrange = ZRANGE

            torch.cuda.empty_cache()

    with open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine_ptreg.json', 'w', encoding='utf-8') as f:
        json.dump(tform_stack_ptreg, f, ensure_ascii=False, indent=4)


def distance_mat(pt1, pt2):
    lrange = 100
    min_dis = []
    for li in range(0, len(pt1), lrange):
        d = pt1[li:li+lrange, None, :] - pt2[None, :, :]
        d = (d**2).sum(-1).sqrt()
        min_d = d.min(1)[0]
        min_dis.append(min_d)
    min_dis = torch.cat(min_dis)
    return min_dis

    
def point_reg_cpd(pt_mov, pt_tgt, only_overlap=True, device='cpu', **kargs):
    if only_overlap:
        pmask_mov, pmask_tgt = point_overlap_index(pt_mov, pt_tgt)
        pt_tgt = pt_tgt[pmask_tgt].float()
        pt_mov = pt_mov[pmask_mov].float()
        
    # create a RigidRegistration object
    reg = RigidRegistration(X=pt_tgt.numpy(), Y=pt_mov.numpy(), device=device, tolerance=0.00001, **kargs)
    # run the registration & collect the results
    TY, (s_reg, R_reg, t_reg) = reg.register()
    correspondence = reg.P
    del reg
    return s_reg, R_reg, t_reg, correspondence

def sort_stackname(stack_names):
    stack_z = []
    for stack_name in stack_names:
        stack_z.append(int(stack_name.split('zmin')[1].split('_')[0]))

    argsort = np.argsort(stack_z)
    return [stack_names[i] for i in argsort]

if __name__ == '__main__': main()