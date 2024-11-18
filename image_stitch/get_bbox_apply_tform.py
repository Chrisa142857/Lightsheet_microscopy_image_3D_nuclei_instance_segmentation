import torch, os, json, copy
import numpy as np
# from tqdm import trange, tqdm
# from final_stitch import ZRANGE, OVERLAP_R
# import nibabel as nib
from datetime import datetime
# from torchvision.ops._utils import _upcast
# from torch_scatter import scatter_max

OVERLAP_R = 0.2
zratio = 2.5/4

def get_bbox_with_tform(ptag, btag):
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

    zstart = 0 

    if os.path.exists(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_manual.json'):
        tform_stack_manual = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_manual.json', 'r', encoding='utf-8'))

    else:
        tform_stack_manual = None
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
    out_tformed_tile_lt_loc = {zi: {} for zi in range(zstart, zend)}
    if tform_stack_manual is not None:
        tformed_tile_lt_loc_refined = {zi: copy.deepcopy(tile_lt_loc) for zi in range(zstart, zend)}
        for zi in range(zstart, zend):
            for k in tform_stack_manual[zi]:
                tz, tx, ty = tform_stack_manual[zi][k]
                tformed_tile_lt_loc_refined[zi][k][0] = tformed_tile_lt_loc_refined[zi][k][0] + tx
                tformed_tile_lt_loc_refined[zi][k][1] = tformed_tile_lt_loc_refined[zi][k][1] + ty
    else:
        ## Coarse
        tformed_tile_lt_loc = {0: copy.deepcopy(tile_lt_loc)}
        for k in tform_stack_coarse:
            tz, tx, ty = tform_stack_coarse[k]
            tformed_tile_lt_loc[0][k][0] = tformed_tile_lt_loc[0][k][0] + tx
            tformed_tile_lt_loc[0][k][1] = tformed_tile_lt_loc[0][k][1] + ty
        ## Refine
        tformed_tile_lt_loc_refined = {zi: copy.deepcopy(tformed_tile_lt_loc[0]) for zi in range(zstart, zend)}
        for zi in range(zstart, zend):
            for k in tform_stack_refine[zi]:
                tx, ty = tform_stack_refine[zi][k]
                tformed_tile_lt_loc_refined[zi][k][0] = tformed_tile_lt_loc_refined[zi][k][0] + tx
                tformed_tile_lt_loc_refined[zi][k][1] = tformed_tile_lt_loc_refined[zi][k][1] + ty

    '''
    Get bbox and apply tform for NMS
    '''
    stack_nis_bbox = {}
    stack_nis_label = {}
    tform_xy_max = [0.05*seg_shape[1], 0.05*seg_shape[2]]
    # for _j in trange(nrow, desc='Get bounding box of NIS'):
    print(datetime.now(), "Load bounding box after transform")
    for _j in range(nrow):
        for _i in range(ncol):
            k = f'{_i}-{_j}'
            _bbox = []
            _label = []
            for zi in tformed_tile_lt_loc_refined.keys():
                if k not in out_tformed_tile_lt_loc[zi]: 
                    out_tformed_tile_lt_loc[zi][k] = [tile_lt_loc[k][0], tile_lt_loc[k][1]]
            for stack_name in stack_names:
                if not os.path.exists(f"{root % (_i, _j)}/{stack_name}"): continue
                zstart = int(stack_name.split('zmin')[1].split('_')[0])
                zstart = int(zstart*zratio)
                labelfn = f"{root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_label')}"
                label = torch.load(labelfn).long()
                bboxfn = f"{root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_bbox')}"
                bbox = torch.load(bboxfn)
                bbox[:, 0] = bbox[:, 0] * zratio + zstart
                bbox[:, 3] = bbox[:, 3] * zratio + zstart
                    
                _bbox.append(bbox)
                _label.append(label)
            if len(_bbox) == 0: continue

            if k not in stack_nis_bbox: stack_nis_bbox[k] = []
            if k not in stack_nis_label: stack_nis_label[k] = []
            _bbox = torch.cat(_bbox)
            _label = torch.cat(_label)
            
            if tform_stack_manual is None:
                tz = tform_stack_coarse[k][0]
                _bbox[:, 0] = _bbox[:, 0] + tz
                _bbox[:, 3] = _bbox[:, 3] + tz
            ct = (_bbox[:, :3] + _bbox[:, 3:]) / 2
            pre_refine_lt_x = tile_lt_loc[k][0]
            pre_refine_lt_y = tile_lt_loc[k][1]
            pre_tform_zi = list(tformed_tile_lt_loc_refined.keys())[0]
            for zi in tformed_tile_lt_loc_refined.keys():
                ct_zmask = torch.where(ct[:, 0].long() == zi)[0]
                if len(ct_zmask) == 0: continue
                bbox = _bbox[ct_zmask].clone()
                label = _label[ct_zmask]
                if tform_stack_manual is None:
                    if abs(tform_stack_refine[zi][k][0]) > tform_xy_max[0] or abs(tform_stack_refine[zi][k][1]) > tform_xy_max[1]:
                        if pre_refine_lt_x is not None:
                            refine_lt_x = pre_refine_lt_x
                            refine_lt_y = pre_refine_lt_y
                        else:
                            for zii in range(zi, len(tform_stack_refine)):
                                if abs(tform_stack_refine[zii][k][0]) <= tform_xy_max[0] and abs(tform_stack_refine[zii][k][1]) <= tform_xy_max[1]:
                                    refine_lt_x = tformed_tile_lt_loc_refined[zii][k][0]
                                    refine_lt_y = tformed_tile_lt_loc_refined[zii][k][1]
                                    break
                    else:
                        refine_lt_x = tformed_tile_lt_loc_refined[zi][k][0]
                        refine_lt_y = tformed_tile_lt_loc_refined[zi][k][1]
                else:
                    refine_lt_x = tformed_tile_lt_loc_refined[zi][k][0]
                    refine_lt_y = tformed_tile_lt_loc_refined[zi][k][1]

                
                if tform_stack_manual is not None:
                    tform_zi = int(np.round(zi + tform_stack_manual[zi][k][0]).item())
                else:
                    tform_zi = zi
                if tform_zi in out_tformed_tile_lt_loc:
                    out_tformed_tile_lt_loc[tform_zi][k] = [refine_lt_x, refine_lt_y]
                    pre_tform_zi = tform_zi
                else:
                    out_tformed_tile_lt_loc[pre_tform_zi][k] = [pre_refine_lt_x, pre_refine_lt_y]
                pre_refine_lt_x = refine_lt_x
                pre_refine_lt_y = refine_lt_y
                if tform_stack_manual is None:
                    if tform_stack_ptreg is not None:
                        if k in tform_stack_ptreg[zi]:
                            tx, ty = tform_stack_ptreg[zi][k]
                            refine_lt_x = refine_lt_x + tx
                            refine_lt_y = refine_lt_y + ty

                bbox[:, 1] = bbox[:, 1] + refine_lt_x
                bbox[:, 2] = bbox[:, 2] + refine_lt_y
                bbox[:, 4] = bbox[:, 4] + refine_lt_x
                bbox[:, 5] = bbox[:, 5] + refine_lt_y
                if tform_stack_manual is not None:
                    bbox[:, 0] = bbox[:, 0] + tform_stack_manual[zi][k][0]
                    bbox[:, 3] = bbox[:, 3] + tform_stack_manual[zi][k][0]
                

                bbox = bbox.clip(min=0)
                stack_nis_bbox[k].append(bbox)
                stack_nis_label[k].append(label)

            stack_nis_bbox[k] = torch.cat(stack_nis_bbox[k])
            stack_nis_label[k] = torch.cat(stack_nis_label[k])
    return stack_nis_bbox, stack_nis_label, out_tformed_tile_lt_loc

def sort_stackname(stack_names):
    stack_z = []
    for stack_name in stack_names:
        stack_z.append(int(stack_name.split('zmin')[1].split('_')[0]))

    argsort = np.argsort(stack_z)
    return [stack_names[i] for i in argsort]
