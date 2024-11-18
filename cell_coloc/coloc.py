import torch, os
import numpy as np
from tqdm import tqdm, trange
from datetime import datetime
from PIL import Image
# import seaborn as sns
# import matplotlib.pyplot as plt
import nibabel as nib
from skimage.filters import threshold_otsu

OVERLAP_R = 0.2
zratio = 2.5/4

def main():
    
    brain_tag_ls = [       
        'L35D719P3',
        'L35D719P5',
        'L64D804P3',
        'L64D804P9',
        'L73D766P4',
        'L73D766P9',
        'L74D769P4',
        'L74D769P8',
        'L77D764P2',
        'L77D764P9',
        'L79D769P7',
        'L79D769P9',
        'L91D814P2',
        'L91D814P6',
    ]
    # ptag='pair4'
    # # btag='220902_L35D719P3_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-41-36'
    # btag='220904_L35D719P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_21-49-38'
    # coloc(ptag, btag)
    # cell_type_map(ptag, btag)
    r = '/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4'
    pbtag_ls = []
    for ptag in os.listdir(r):
        for btag in os.listdir(f'{r}/{ptag}'):
            pbtag_ls.append([ptag, btag])
    pbtag_ls = list(reversed(pbtag_ls))
    for ptag, btag in pbtag_ls[4:]:
        if btag.split('_')[1] not in brain_tag_ls: continue
        # coloc(ptag, btag)
        try:
            coloc(ptag, btag)
        except KeyboardInterrupt:
            return
        except Exception as error:
            print("ERROR:", error)

def coloc(ptag, btag, device='cpu'):
    overlap_r = OVERLAP_R
    lrange = 1000
    channel1_tag = 'C00'
    channel2_tag = 'C02'
    root1 = '/cajal/Felix/Lightsheet/P4'
    root2 = '/lichtman/Felix/Lightsheet/P4'
    root = f'{root1}/{ptag}/{btag}' if os.path.exists(f'{root1}/{ptag}/{btag}') else f'{root2}/{ptag}/{btag}'
    assert os.path.exists(root), root
    img_tag = 'UltraII[%02d x %02d]'
    save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/colocalization/P4/{ptag}/{btag.split("_")[1]}/{img_tag}'
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}/{btag}'
    result_root = result_path + '/UltraII[%02d x %02d]'
    stack_names = [f for f in os.listdir(result_root % (0, 0)) if f.endswith('instance_center.zip')]
    stack_names = sort_stackname(stack_names)
    meta_name = stack_names[0].replace('instance_center', 'seg_meta')
    seg_shape = torch.load(f'{result_root % (0, 0)}/{meta_name}')
    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path)])
    ncol, nrow = tile_loc.max(0)+1
    seg_shape = [s.item() for s in seg_shape]
    tile_lt_loc = {
        f'{i}-{j}': [i*seg_shape[1]*(1-overlap_r), j*seg_shape[2]*(1-overlap_r)] for i in range(ncol) for j in range(nrow)
    }
    whole_brain_shape = [seg_shape[0]] + list(np.array(list(tile_lt_loc.values())).max(0) + np.array(seg_shape[1:]))
    whole_brain_shape = [int(s) for s in whole_brain_shape]
    D = len(whole_brain_shape)
    print("stack_names, ncol, nrow, whole_brain_shape:\n", stack_names, ncol, nrow, whole_brain_shape)
    down_ratio = 0.1 # for brain depth
    for i in range(ncol):
        for j in range(nrow):
            if os.path.exists(f'{save_path % (i, j)}/{btag.split("_")[1]}_{channel1_tag}_fgmask.nii'): continue
            os.makedirs(save_path % (i, j), exist_ok=True)
            img_fns = [fn for fn in os.listdir(root) if img_tag % (i, j) in fn]
            seg_shape[0] = len(img_fns) // 3
            img3d_c1 = np.zeros(seg_shape)
            for fn in tqdm(img_fns, desc=f'{datetime.now()} Load tiles {i},{j} {channel1_tag}'):
                if channel1_tag not in fn: continue
                zi = int(fn.split(' Z')[1][:4])
                img = Image.open(f'{root}/{fn}')
                img = np.asarray(img)
                img3d_c1[zi] = img
                
            # print(img3d_c1.max(), img3d_c1.min())
            
            img3d_c2 = np.zeros(seg_shape)
            for fn in tqdm(img_fns, desc=f'{datetime.now()} Load tiles {i},{j} {channel2_tag}'):
                if channel2_tag not in fn: continue
                zi = int(fn.split(' Z')[1][:4])
                img = Image.open(f'{root}/{fn}')
                img = np.asarray(img)
                img3d_c2[zi] = img
                
            # print(img3d_c2.max(), img3d_c2.min())
                
            binary_mask = []
            bshape = 0
            for stack_name in stack_names:
                fn = f"{result_root % (i, j)}/{stack_name.replace('instance_center', 'binary_mask')}"
                b = torch.load(fn).to(device)
                new_b = []
                for xi in range(b.shape[1]):
                    new_b.append(torch.nn.functional.interpolate(b[None, None, :, xi].float(), scale_factor=[zratio, 1])[0, 0] > 0)
                new_b = torch.stack(new_b, dim=1).cpu()
                bshape += new_b.shape[0]
                binary_mask.append(new_b)

            binary_mask = torch.cat(binary_mask+[torch.zeros(seg_shape[0]-bshape, seg_shape[1], seg_shape[2]).bool()]).to(device)

            # print(binary_mask.shape, binary_mask.dtype)
            
            img3d_c1 = torch.from_numpy(img3d_c1).float()
            img3d_c2 = torch.from_numpy(img3d_c2).float()
            '''
            Brain depth computation by distance between grid center to brain center 
            '''
            brain_center = torch.LongTensor([(s/2) for s in whole_brain_shape]).to(device)
            tile_grid = torch.meshgrid(torch.arange(int(img3d_c1.shape[0]*down_ratio)), torch.arange(int(img3d_c1.shape[1]*down_ratio)), torch.arange(int(img3d_c1.shape[2]*down_ratio)))
            tile_grid = torch.stack(tile_grid).to(device)
            tile_grid[1] = tile_grid[1] + down_ratio*tile_lt_loc[f'{i}-{j}'][0]
            tile_grid[2] = tile_grid[2] + down_ratio*tile_lt_loc[f'{i}-{j}'][1]
            tile_grid = tile_grid / down_ratio
            grid_dis_to_center = tile_grid - brain_center[:, None, None, None]
            grid_dis_to_center = (grid_dis_to_center**2).sum(0).sqrt()
            org_grid_dis_to_center = torch.nn.functional.interpolate(grid_dis_to_center[None,None].cpu(), size=img3d_c1.shape)[0,0]#.to(device)
            '''
            Run Otsu for every distance to center around NIS via inflate
            '''
            inflate_size = 7
            inflater = torch.nn.MaxPool2d(3, stride=1, padding=1).to(device)
            around_nis = torch.zeros_like(binary_mask).bool()
            for z in trange(len(binary_mask)):
                bg = binary_mask[z].float().to(device)
                for _ in range(inflate_size):
                    bg = inflater(bg[None, None])[0, 0]
                bg = bg > 0
                around_nis[z] = bg#.cpu()
            around_nis = around_nis.cpu()
            # print(around_nis.shape, around_nis.dtype)

            nbins = 256
            distance_group_n = 50
            dinterval = int((grid_dis_to_center.max() - grid_dis_to_center.min())/distance_group_n)
            dranges = [[d, d+dinterval] for d in range(grid_dis_to_center.min().long(), grid_dis_to_center.max().long()+1, dinterval)]
            # print(dranges[:10])
            fg_mask_c1 = torch.zeros_like(img3d_c1).bool()
            fg_mask_c2 = torch.zeros_like(img3d_c2).bool()
            th_c1s = {}
            th_c2s = {}
            
            for di, (dmin, dmax) in tqdm(enumerate(dranges), total=len(dranges)):
            #     print(dmin, dmax)
                dmask = torch.logical_and(org_grid_dis_to_center >= dmin, org_grid_dis_to_center < dmax)
                mask = torch.logical_and(dmask, around_nis)#.cpu()
                if not mask.any(): continue
                c1 = img3d_c1[mask].numpy()
                th_c1 = threshold_otsu(image=c1, nbins=nbins)
                c2 = img3d_c2[mask].numpy()
                th_c2 = threshold_otsu(image=c2, nbins=nbins)
            #     print(c1.shape, th_c1, th_c2)
                fg_mask_c1[torch.logical_and(torch.logical_and(img3d_c1>th_c1, dmask), binary_mask)] = True
                fg_mask_c2[torch.logical_and(torch.logical_and(img3d_c2>th_c2, dmask), binary_mask)] = True
                th_k = f'brain-depth=[{dmin},{dmax}],grid-ratio={down_ratio}'
                th_c1s[th_k] = th_c1
                th_c2s[th_k] = th_c2
            
            torch.save(th_c1s, f'{save_path % (i, j)}/{btag.split("_")[1]}_{channel1_tag}_thresholds.zip')
            torch.save(th_c2s, f'{save_path % (i, j)}/{btag.split("_")[1]}_{channel2_tag}_thresholds.zip')
            nib.save(nib.Nifti1Image(fg_mask_c1.numpy().astype(np.uint8), np.eye(4)), f'{save_path % (i, j)}/{btag.split("_")[1]}_{channel1_tag}_fgmask.nii')
            nib.save(nib.Nifti1Image(fg_mask_c2.numpy().astype(np.uint8), np.eye(4)), f'{save_path % (i, j)}/{btag.split("_")[1]}_{channel2_tag}_fgmask.nii')

            # intensity_diff = img3d_c1 - img3d_c2
            # intensity_diff = torch.from_numpy(intensity_diff).float()

            # '''
            # Load bounding box
            # '''
            # bbox = []
            # label = []
            # for stack_name in stack_names:
            #     zstart = int(stack_name.split('zmin')[1].split('_')[0])
            #     zstart = int(zstart*zratio)
            #     bboxfn = f"{result_root % (i, j)}/{stack_name.replace('instance_center', 'instance_bbox')}"
            #     b = torch.load(bboxfn).float()
            #     b[:, 0] = b[:, 0]*zratio + zstart
            #     b[:, 3] = b[:, 3]*zratio + zstart
            #     bbox.append(b)
            #     labelfn = f"{result_root % (i, j)}/{stack_name.replace('instance_center', 'instance_label')}"
            #     label.append(torch.load(labelfn).long())

            # bbox = torch.cat(bbox).to(device)
            # label = torch.cat(label).to(device)
            # ###########################
            # zstitch_remap = torch.load(f"{result_root % (i, j)}/{btag.split('_')[1]}_remap.zip").to(device)
            # print("z-stitch remap dict shape", zstitch_remap.shape)

            # ## loc: gnn stitch source (current tile) nis index, stitch_remap_loc: index of pairs in the stitch remap list
            # loc, stitch_remap_loc = [], []
            # for lrangei in range(0, len(label), lrange):
            #     lo, stitch_remap_lo = torch.where(label[lrangei:lrangei+lrange, None] == zstitch_remap[0, None, :])
            #     loc.append(lo+lrangei)
            #     stitch_remap_loc.append(stitch_remap_lo)
            # loc, stitch_remap_loc = torch.cat(loc), torch.cat(stitch_remap_loc)

            # ## pre_loc: gnn stitch target (previous tile) nis index, tloc: index of remaining Z stitch pairs after nis being removed by X-Y stitching
            # pre_loc, tloc = [], []
            # for lrangei in range(0, len(label), lrange):
            #     pre_lo, tlo = torch.where(label[lrangei:lrangei+lrange, None] == zstitch_remap[1, None, stitch_remap_loc])
            #     pre_loc.append(pre_lo+lrangei)
            #     tloc.append(tlo)
            # pre_loc, tloc = torch.cat(pre_loc), torch.cat(tloc)

            # ## source nis is removed from keeping mask
            # keep_mask = torch.ones(len(bbox)).bool()
            # keep_mask[loc] = False

            # # merge stitched source nis to target nis
            # loc = loc[tloc]
            # bbox[pre_loc, :3] = torch.stack([bbox[loc, :3], bbox[pre_loc, :3]]).min(0)[0]
            # bbox[pre_loc, 3:] = torch.stack([bbox[loc, 3:], bbox[pre_loc, 3:]]).max(0)[0]

            # bbox = bbox[keep_mask]
            # label = label[keep_mask]
            # ###########################

            # intensity_diff = intensity_diff.to(device)
            # bbox[:, :3] = bbox[:, :3].clip(min=0)
            # bbox[:, 3] = bbox[:, 3].clip(max=seg_shape[0])
            # bbox[:, 4] = bbox[:, 4].clip(max=seg_shape[1])
            # bbox[:, 5] = bbox[:, 5].clip(max=seg_shape[2])
            # area = []
            # nis_intensity_diff = []
            # nis_avgintensity_c1 = []
            # nis_avgintensity_c2 = []
            # for b in tqdm(bbox, desc=f'{datetime.now()} Get intensity difference of tile {i},{j} inside NIS'):
            #     b = b.long()
            #     diff = intensity_diff[b[0]:b[3], b[1]:b[4], b[2]:b[5]]
            #     nis_intensity_diff.append(diff)
            #     area.append(len(diff))
            #     nis_avgintensity_c1.append(img3d_c1[b[0]:b[3], b[1]:b[4], b[2]:b[5]].mean().item())
            #     nis_avgintensity_c2.append(img3d_c2[b[0]:b[3], b[1]:b[4], b[2]:b[5]].mean().item())
            
            # sns_data = {
            #     'avg intensity': nis_avgintensity_c1+nis_avgintensity_c2,
            #     'channel': ['C00' for _ in range(len(nis_avgintensity_c1))] + ['C02' for _ in range(len(nis_avgintensity_c2))],
            # }
            # sns.displot(data=sns_data, x="avg intensity", hue='channel', kde=True)
            # plt.savefig(f'{save_path % (i, j)}/{btag.split("_")[1]}_coloc_avg_intensity_dist.png')
            # plt.savefig(f'{save_path % (i, j)}/{btag.split("_")[1]}_coloc_avg_intensity_dist.svg')
            # plt.close()
            # nis_is_c1 = []
            # for diff in tqdm(nis_intensity_diff, desc=f'{datetime.now()} Coloc by intensity difference of tile {i},{j} inside NIS'):
            #     nis_is_c1.append(diff.mean() > 0)
            
            # torch.save(nis_is_c1, f'{save_path % (i, j)}/{btag.split("_")[1]}_coloc_is_c1.zip')

            
def cell_type_map(ptag, btag, device='cuda:2'):
    '''
    Whole brain map
    '''
    overlap_r = OVERLAP_R
    new_header, affine_m = init_nib_header()

    vol_unit = 0.75*0.75*2.5 # 2.5 is the fixed Z resolution of NIS
    downsample_res = [25, 25, 25]
    seg_res = [4, 0.75, 0.75]
    save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/{ptag}/{btag.split("_")[1]}'
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}/{btag}'
    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path)])

    ncol, nrow = tile_loc.max(0)+1
    # print(tile_loc, nrow, ncol)
    assert len(tile_loc) == nrow*ncol, f'tile of raw data is not complete, tile location: {tile_loc}'

    root = result_path + '/UltraII[%02d x %02d]'
    stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    stack_names = sort_stackname(stack_names)
    sshape = [0, 0, 0]
    for si, stack_name in enumerate(stack_names):
        meta_name = stack_name.replace('instance_center', 'seg_meta')
        zstart = int(stack_name.split('zmin')[1].split('_')[0])
        zstart = int(zstart*zratio)
        seg_shape = torch.load(f'{root % (0, 0)}/{meta_name}')
        zend = int(zstart + seg_shape[0].item()*zratio)
        if sshape[0] < zend:
            sshape = [zend, int(seg_shape[1].item()*(1-overlap_r)*(ncol-1)+seg_shape[1].item()), int(seg_shape[2].item()*(1-overlap_r)*(nrow-1)+seg_shape[2].item())]
    
    tform_xy_max = [0.05*seg_shape[1], 0.05*seg_shape[2]]
    dratio = [s/d for s, d in zip(seg_res, downsample_res)]
    dshape = [0, 0, 0]
    dshape[0] = int(sshape[0] * dratio[0])
    dshape[1] = int(sshape[1] * dratio[1])
    dshape[2] = int(sshape[2] * dratio[2])
    print(datetime.now(), f"Downsample space shape: {dshape}, ratio: {dratio}, LS space shape {sshape}")
    zstart = 0
    tile_lt_loc = {
        f'{i}-{j}': [i*seg_shape[1]*(1-overlap_r), j*seg_shape[2]*(1-overlap_r)] for i in range(ncol) for j in range(nrow)
    }
    
    rm_label = torch.load(f'{save_path}/doubled_NIS_label/{btag.split("_")[1]}_doubled_label.zip')
    for k in rm_label:
        rm_label[k] = rm_label[k].to(device)
        print(k, rm_label[k].shape, rm_label[k].min(), rm_label[k].max())
    
    tform_stack_coarse = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_coarse.json', 'r', encoding='utf-8'))
    tform_stack_refine = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine.json', 'r', encoding='utf-8'))
    if os.path.exists(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine_ptreg.json'):
        tform_stack_ptreg = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine_ptreg.json', 'r', encoding='utf-8'))
        print(datetime.now(), "Loaded tform of coarse, refine and pt-reg", [tform_stack_ptreg[zi].keys() for zi in range(len(tform_stack_ptreg)) if len(tform_stack_ptreg[zi].keys()) > 0][-1])
    else:
        tform_stack_ptreg = None
        print(datetime.now(), "Loaded tform of coarse, refine")
    '''
    Whole brain map
    '''
    centers = []
    vols = []
    labels = []
    tile_centers = {}
    tile_vols = {}
    tile_labels = {}
    keep_masks = {}

    lrange = 1000
    for i in range(ncol):
        for j in range(nrow):
            k = f'{i}-{j}'
            print(datetime.now(), k)
            i, j = k.split('-')
            i, j = int(i), int(j)
            bbox = []
            label = []
            for stack_name in stack_names:
                zstart = int(stack_name.split('zmin')[1].split('_')[0])
                zstart = int(zstart*zratio)
                bboxfn = f"{root % (i, j)}/{stack_name.replace('instance_center', 'instance_bbox')}"
                b = torch.load(bboxfn).float()
                b[:, 0] = b[:, 0]*zratio + zstart
                b[:, 3] = b[:, 3]*zratio + zstart
                b[:, 1] = b[:, 1] + tile_lt_loc[k][0]
                b[:, 4] = b[:, 4] + tile_lt_loc[k][0]
                b[:, 2] = b[:, 2] + tile_lt_loc[k][1]
                b[:, 5] = b[:, 5] + tile_lt_loc[k][1]
                bbox.append(b)
                labelfn = f"{root % (i, j)}/{stack_name.replace('instance_center', 'instance_label')}"
                label.append(torch.load(labelfn).long())

            bbox = torch.cat(bbox).to(device)
            label = torch.cat(label).to(device)
            pt = (bbox[:, :3] + bbox[:, 3:]) / 2
            pt[:, 0] = pt[:, 0] + tform_stack_coarse[k][0]
            pt[:, 1] = pt[:, 1] + tform_stack_coarse[k][1] 
            pt[:, 2] = pt[:, 2] + tform_stack_coarse[k][2] 
            pre_refine_tx = None
            pre_refine_ty = None
            for z in range(len(tform_stack_refine)):
                pt_zmask = pt[:,0].long()==z
                if abs(tform_stack_refine[z][k][0]) > tform_xy_max[0] or abs(tform_stack_refine[z][k][1]) > tform_xy_max[1]:
                    if pre_refine_tx is not None:
                        refine_tx = pre_refine_tx
                        refine_ty = pre_refine_ty
                    else:
                        for zi in range(z, len(tform_stack_refine)):
                            if abs(tform_stack_refine[zi][k][0]) <= tform_xy_max[0] and abs(tform_stack_refine[zi][k][1]) <= tform_xy_max[1]:
                                refine_tx = tform_stack_refine[zi][k][0]
                                refine_ty = tform_stack_refine[zi][k][1]
                                break
                else:
                    refine_tx = tform_stack_refine[z][k][0]
                    refine_ty = tform_stack_refine[z][k][1]
                    
                pt[pt_zmask, 1] = pt[pt_zmask, 1] + refine_tx
                pt[pt_zmask, 2] = pt[pt_zmask, 2] + refine_ty
                pre_refine_tx = refine_tx
                pre_refine_ty = refine_ty
                if tform_stack_ptreg is not None:
                    if k in tform_stack_ptreg[z]:
                        tx, ty = tform_stack_ptreg[z][k]
                        pt[pt_zmask, 1] = pt[pt_zmask, 1] + tx
                        pt[pt_zmask, 2] = pt[pt_zmask, 2] + ty

            vol = []
        #     vol = torch.cat(vol).to(device)
            torch.cuda.empty_cache()
            ########
            print("before remove doubled pt", pt.shape, label.shape)
            if k in rm_label:
                keep_ind = []
                for labeli in range(0, len(label), lrange):
                    label_batch = label[labeli:labeli+lrange]
                    label2rm = label_batch[:, None] == rm_label[k][None, :]
                    do_rm = label2rm.any(1)
        #                         rm_label[f'{i}-{j}'] = rm_label[f'{i}-{j}'][torch.logical_not(label2rm.any(0))]
                    keep_ind.append(torch.arange(labeli, labeli+len(label_batch), device=device)[torch.logical_not(do_rm)])
                if len(label) > 0:
                    keep_ind = torch.cat(keep_ind)
                    pt = pt[keep_ind]
        #             vol = vol[keep_ind]
                    label = label[keep_ind]

            print("after remove doubled pt", pt.shape, label.shape)
            ####################
            zstitch_remap = torch.load(f"{root % (i, j)}/{btag.split('_')[1]}_remap.zip").to(device)
            print("z-stitch remap dict shape", zstitch_remap.shape)

            ## loc: gnn stitch source (current tile) nis index, stitch_remap_loc: index of pairs in the stitch remap list
            loc, stitch_remap_loc = [], []
            for lrangei in range(0, len(label), lrange):
                lo, stitch_remap_lo = torch.where(label[lrangei:lrangei+lrange, None] == zstitch_remap[0, None, :])
                loc.append(lo+lrangei)
                stitch_remap_loc.append(stitch_remap_lo)
            loc, stitch_remap_loc = torch.cat(loc), torch.cat(stitch_remap_loc)

            ## pre_loc: gnn stitch target (previous tile) nis index, tloc: index of remaining Z stitch pairs after nis being removed by X-Y stitching
            pre_loc, tloc = [], []
            for lrangei in range(0, len(label), lrange):
                pre_lo, tlo = torch.where(label[lrangei:lrangei+lrange, None] == zstitch_remap[1, None, stitch_remap_loc])
                pre_loc.append(pre_lo+lrangei)
                tloc.append(tlo)
            pre_loc, tloc = torch.cat(pre_loc), torch.cat(tloc)

            ## source nis is removed from keeping mask
            keep_mask = torch.ones(len(pt)).bool()
            keep_mask[loc] = False
        #     keep_masks[stack_name][f'{i}-{j}'] = torch.logical_and(keep_masks[stack_name][f'{i}-{j}'], keep_mask)

            # merge stitched source nis to target nis
            loc = loc[tloc]
            pt[pre_loc] = (pt[loc] + pt[pre_loc]) / 2
        #     vol[pre_loc] = vol[loc] + pre_vol[pre_loc]

            pt = pt[keep_mask]
        #     vol = vol[keep_mask]
            label = label[keep_mask]
            ###########################

            print("after remove z-stitched pt", pt.shape, label.shape)
            centers.append(pt.cpu())
        #     vols.append(vol.cpu())
            labels.append(label.cpu())
        

    centers = torch.cat(centers).cpu()
    # vols = torch.cat(vols).cpu()
    vols = torch.zeros(10)
    torch.cuda.empty_cache()
    print(datetime.now(), "Whole-brain nuclei counting", centers.shape[0])
    density_map, volavg_map = downsample(centers, vols, dratio, dshape, device, skip_vol=True)
    print(datetime.now(), f"Saving total density {ptag} {btag.split('_')[1]}, Max density {density_map.max()}")
    os.makedirs(f'{save_path}/whole_brain_map', exist_ok=True)
    if tform_stack_ptreg is not None:
        nib.save(nib.Nifti1Image(density_map.numpy().astype(np.float64), affine_m, header=new_header), f'{save_path}/whole_brain_map/NIS_density_after_rm2_{ptag}_{btag.split("_")[1]}.nii.gz')
    else:
        nib.save(nib.Nifti1Image(density_map.numpy().astype(np.float64), affine_m, header=new_header), f'{save_path}/whole_brain_map/NIS_density_after_rm_{ptag}_{btag.split("_")[1]}.nii.gz')


def sort_stackname(stack_names):
    stack_z = []
    for stack_name in stack_names:
        stack_z.append(int(stack_name.split('zmin')[1].split('_')[0]))

    argsort = np.argsort(stack_z)
    return [stack_names[i] for i in argsort]

def downsample(center, vol, ratio, dshape, device, skip_vol=False):
    center = center.to(device)
    vol = vol.float().to(device)
    center[:,0] = center[:,0] * ratio[0]
    center[:,1] = center[:,1] * ratio[1]
    center[:,2] = center[:,2] * ratio[2]
    z = center[:, 0].clip(min=0, max=dshape[0]-0.9)
    y = center[:, 1].clip(min=0, max=dshape[1]-0.9)
    x = center[:, 2].clip(min=0, max=dshape[2]-0.9)
    # print(center.shape, z, x, y)
    loc = torch.arange(dshape[0]*dshape[1]*dshape[2]).view(dshape[0], dshape[1], dshape[2]).to(device) 
    loc = loc[(z.long(), y.long(), x.long())] # all nis location in the downsample space
    loc_count = loc.bincount() 
    loc_count = loc_count[loc_count!=0] 
    atlas_loc = loc.unique().to(device) # unique location in the downsample space
    ## volume avg & local intensity
    vol_avg = None
    if not skip_vol:
        loc_argsort = loc.argsort().cpu()
        loc_splits = loc_count.cumsum(0).cpu()
        loc_vol = torch.tensor_split(vol[loc_argsort], loc_splits)
        assert len(loc_vol[-1]) == 0
        loc_vol = loc_vol[:-1]
        loc_vol = torch.nn.utils.rnn.pad_sequence(loc_vol, batch_first=True, padding_value=-1)
        loc_fg = loc_vol!=-1
        loc_num = loc_fg.sum(1)
        loc_vol[loc_vol==-1] = 0
        vol_avg = torch.zeros(dshape[0]*dshape[1]*dshape[2]).float()#.to(device)
        vol_avg[atlas_loc] = (loc_vol.sum(1) / loc_num).cpu().float()
        # for loci in tqdm(atlas_loc, desc="Collect NIS property in local cube"): 
        #     where_loc = torch.where(loc==loci)[0]
        #     vol_avg[loci] = vol[where_loc].mean()
        vol_avg = vol_avg.view(dshape[0], dshape[1], dshape[2])#.cpu()
    ## density map
    density = torch.zeros(dshape[0]*dshape[1]*dshape[2], dtype=torch.float64).to(device)
    density[atlas_loc] = loc_count.double() #/ center.shape[0]
    density = density.view(dshape[0], dshape[1], dshape[2]).cpu()
    return density, vol_avg

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

if __name__ == '__main__': main()