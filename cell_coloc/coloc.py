import torch, os, sys
import numpy as np
from tqdm import tqdm, trange
from datetime import datetime
from PIL import Image
# import seaborn as sns
# import matplotlib.pyplot as plt
import nibabel as nib
from skimage.filters import threshold_otsu
from torch_scatter import scatter_mean
from multiprocessing import Pool

OVERLAP_R = 0.2
zratio = 2.5/4

def main():
    if len(sys.argv)>1:
        run_reverse = sys.argv[1]=="1"
    else:
        run_reverse = False
    brain_tag_ls = [       
        # 'L35D719P3',
        # 'L35D719P5',
        # 'L64D804P3',
        # 'L64D804P9',
        # 'L73D766P4',
        # 'L73D766P9',
        # 'L74D769P4',
        # 'L74D769P8',
        # 'L77D764P2',
        # 'L77D764P9',
        # 'L79D769P7',
        # 'L79D769P9',
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
        coloc(ptag, btag, run_reverse=run_reverse)
        # try:
        #     coloc(ptag, btag)
        # except KeyboardInterrupt:
        #     return
        # except Exception as error:
        #     print("ERROR:", error)

def coloc(ptag, btag, device='cpu', run_reverse=False):
    overlap_r = OVERLAP_R
    lrange = 1000
    channel1_tag = 'C00'
    channel2_tag = 'C02'
    root1 = '/cajal/Felix/Lightsheet/P4'
    root2 = '/lichtman/Felix/Lightsheet/P4'
    root = f'{root1}/{ptag}/{btag}' if os.path.exists(f'{root1}/{ptag}/{btag}') else f'{root2}/{ptag}/{btag}'
    assert os.path.exists(root), root
    img_tag = 'UltraII[%02d x %02d]'
    # save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/colocalization/P4/{ptag}/{btag.split("_")[1]}/{img_tag}'
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
    brain_center = torch.LongTensor([(s/2) for s in whole_brain_shape]).to(device)
    nbins = 256
    distance_group_n = 30
    # dinterval = int((grid_dis_to_center.max() - grid_dis_to_center.min())/distance_group_n)
    # dranges = [[d, d+dinterval] for d in range(grid_dis_to_center.min().long(), grid_dis_to_center.max().long()+1, dinterval)]
    brain_radius = (brain_center**2).sum().sqrt().item()
    log_spaced_values = [0] + list(np.logspace(np.log10(1), np.log10(brain_radius), num=distance_group_n))
    _dranges = [[brain_radius-log_spaced_values[logi+1], brain_radius-log_spaced_values[logi]] for logi in range(len(log_spaced_values)-1)]
    dranges = []
    tile_radius = (torch.FloatTensor(seg_shape)**2).sum().sqrt().item()
    for dmin, dmax in _dranges:
        if dmin < tile_radius:
            dmin = 0
            dranges.append([0, dmax])    
            break
        dranges.append([dmin, dmax])

    print(dranges)
    print("stack_names, ncol, nrow, whole_brain_shape:\n", stack_names, ncol, nrow, whole_brain_shape)
    down_ratio = 0.1 # for brain depth
    if run_reverse:
        col_iterator = range(1, ncol)
        # col_iterator = range(ncol-1, -1, -1)
        row_iterator = range(nrow-2, -1, -1)
    else:
        col_iterator = range(ncol)
        row_iterator = range(nrow)

    brain_c1 = {f'{dmin},{dmax}': {} for dmin, dmax in dranges}
    brain_c2 = {f'{dmin},{dmax}': {} for dmin, dmax in dranges}
    nis_avg_items = {}
    for i in col_iterator:
        for j in row_iterator:
            ijkey = f'{i}{j}'
            # if f'{i}{j}' in ['00', '01', '02']: continue
            # if os.path.exists(f'{save_path % (i, j)}/{btag.split("_")[1]}_NIScpp_results_zmin0_instance_C00_layer.zip'): continue
            # os.makedirs(save_path % (i, j), exist_ok=True)
            img_fns = [fn for fn in os.listdir(root) if img_tag % (i, j) in fn]
            seg_shape[0] = len(img_fns) // 3
            img3d_c1 = np.zeros(seg_shape)
            with Pool(processes=30) as loader_pool:
                image_list = list(loader_pool.imap(Image.open, tqdm([f'{root}/{fn}' for fn in img_fns if channel1_tag in fn], desc=f'{datetime.now()} Load tiles {i},{j} {channel1_tag}')))
            fnlist = [fn for fn in img_fns if channel1_tag in fn]
            for img, fn in zip(image_list, fnlist):
                zi = int(fn.split(' Z')[1][:4])
                img3d_c1[zi] = np.asarray(img)
                
            # '''
            # Brain depth computation by distance between grid center to brain center 
            # '''
            # tile_grid = torch.meshgrid(torch.arange(int(img3d_c1.shape[0]*down_ratio)), torch.arange(int(img3d_c1.shape[1]*down_ratio)), torch.arange(int(img3d_c1.shape[2]*down_ratio)))
            # tile_grid = torch.stack(tile_grid).to(device)
            # tile_grid[1] = tile_grid[1] + down_ratio*tile_lt_loc[f'{i}-{j}'][0]
            # tile_grid[2] = tile_grid[2] + down_ratio*tile_lt_loc[f'{i}-{j}'][1]
            # tile_grid = tile_grid / down_ratio
            # grid_dis_to_center = tile_grid - brain_center[:, None, None, None]
            # grid_dis_to_center = (grid_dis_to_center**2).sum(0).sqrt()#.cpu()
            # org_grid_dis_to_center = torch.nn.functional.interpolate(grid_dis_to_center[None,None], size=img3d_c1.shape)[0,0]#.to(device)

            img3d_c2 = np.zeros(seg_shape)
            with Pool(processes=30) as loader_pool:
                image_list = list(loader_pool.imap(Image.open, tqdm([f'{root}/{fn}' for fn in img_fns if channel2_tag in fn], desc=f'{datetime.now()} Load tiles {i},{j} {channel1_tag}')))
            fnlist = [fn for fn in img_fns if channel2_tag in fn]
            for img, fn in zip(image_list, fnlist):
                zi = int(fn.split(' Z')[1][:4])
                img3d_c2[zi] = np.asarray(img)
                
            img3d_c1 = torch.from_numpy(img3d_c1).float()#.to(device)
            img3d_c2 = torch.from_numpy(img3d_c2).float()#.to(device)

            # print(img3d_c2.max(), img3d_c2.min())
                
            binary_mask = []
            bshape = 0
            nis_dis2bcenter = []
            nis_avg_c1_ls = []
            nis_avg_c2_ls = []
            nis_c1_save_fns = []
            nis_c2_save_fns = []
            print(datetime.now(), 'Prepare NIS avg intensity')
            for stack_name in stack_names:
                zmin = int(stack_name.split('_zmin')[1].split('_')[0])
                fn = f"{result_root % (i, j)}/{stack_name.replace('instance_center', 'binary_mask')}"
                center_fn = f"{result_root % (i, j)}/{stack_name}"
                coord_fn = f"{result_root % (i, j)}/{stack_name.replace('instance_center', 'instance_coordinate')}"
                vol_fn = f"{result_root % (i, j)}/{stack_name.replace('instance_center', 'instance_volume')}"
                b = torch.load(fn)#.to(device)
                _coord = torch.load(coord_fn)#.to(device)
                _vol = torch.load(vol_fn)#.to(device)
                center = torch.load(center_fn).to(device)
                center[:, 0] = (center[:, 0] + zmin)*zratio
                _coord[:, 0] = (_coord[:, 0] + zmin)*zratio
                center[:, 1] = center[:, 1] + tile_lt_loc[f'{i}-{j}'][0]
                # _coord[:, 1] = _coord[:, 1] + tile_lt_loc[f'{i}-{j}'][0]
                center[:, 2] = center[:, 2] + tile_lt_loc[f'{i}-{j}'][1]
                # _coord[:, 2] = _coord[:, 2] + tile_lt_loc[f'{i}-{j}'][1]
                nis_avg_c1 = []
                nis_avg_c2 = []
                cumsum_vol = _vol.cumsum(0)
                cumsum_vol = [0] + cumsum_vol.tolist()
                _scatter_ind = torch.arange(lrange).to(device)
                for li in range(0, len(_vol), lrange):
                    vol = _vol[li:li+lrange].to(device)
                    start = cumsum_vol[li]
                    end = cumsum_vol[li+lrange] if li+lrange < len(cumsum_vol) else cumsum_vol[-1]
                    coord = _coord[start:end]#.to(device)
                    # print(coord.max(0), img3d_c1.shape)
                    coord_c1 = img3d_c1[coord[:, 0].long(), coord[:, 1].long(), coord[:, 2].long()].to(device)
                    coord_c2 = img3d_c2[coord[:, 0].long(), coord[:, 1].long(), coord[:, 2].long()].to(device)
                    scatter_ind = _scatter_ind[:len(vol)].repeat_interleave(vol)
                    nis_avg_c1.append(scatter_mean(coord_c1, scatter_ind))
                    nis_avg_c2.append(scatter_mean(coord_c2, scatter_ind))
                nis_avg_c1 = torch.cat(nis_avg_c1)
                nis_avg_c2 = torch.cat(nis_avg_c2)
                nis_dis_bcenter = center - brain_center[None]
                nis_dis_bcenter = (nis_dis_bcenter**2).sum(1).sqrt()
                del coord_c1, coord_c2, coord, vol, center

                nis_dis2bcenter.append(nis_dis_bcenter.cpu())
                nis_avg_c1_ls.append(nis_avg_c1.cpu())
                nis_avg_c2_ls.append(nis_avg_c2.cpu())
                nis_c1_save_fns.append(f"{result_root % (i, j)}/{stack_name.replace('instance_center', f'instance_{channel1_tag}_layer')}")
                nis_c2_save_fns.append(f"{result_root % (i, j)}/{stack_name.replace('instance_center', f'instance_{channel2_tag}_layer')}")

                new_b = []
                for xi in range(b.shape[1]):
                    new_b.append(torch.nn.functional.interpolate(b[None, None, :, xi].float().to(device), scale_factor=[zratio, 1])[0, 0] > 0)
                new_b = torch.stack(new_b, dim=1)
                bshape += new_b.shape[0]
                binary_mask.append(new_b)

            binary_mask = torch.cat(binary_mask+[torch.zeros(seg_shape[0]-bshape, seg_shape[1], seg_shape[2]).bool().to(binary_mask[0].device)])#.to(device_img)
            # img3d_c1 = img3d_c1.cpu()#.to(device_img)
            # img3d_c2 = img3d_c2.cpu()#.to(device_img)
            # print(binary_mask.shape, binary_mask.dtype)
            nis_avg_items[ijkey]  = [nis_dis2bcenter,nis_avg_c1_ls,nis_avg_c2_ls,nis_c1_save_fns,nis_c2_save_fns]
            '''
            Run Otsu for every distance to center around NIS via inflate
            '''
            inflate_size = 11
            reduce_size = 4 # will remove small nis
            inflater = torch.nn.MaxPool2d(3, stride=1, padding=1).to(device)
            reducer = torch.nn.MaxPool2d(3, stride=1, padding=1).to(device)
            around_nis = binary_mask
            for z in range(len(binary_mask)):
                bg = binary_mask[z].float().to(device)
                for _ in range(reduce_size):
                    bg = reducer(bg[None, None]*-1)[0, 0]*-1
                for _ in range(inflate_size):
                    bg = inflater(bg[None, None])[0, 0]
                bg = bg > 0
                around_nis[z] = bg#.cpu()
            nis_z, nis_x, nis_y = torch.where(around_nis)
            print(nis_z.min(), nis_z.max(), nis_z.shape, nis_x.min(), nis_x.max(), nis_x.shape, nis_y.min(), nis_y.max(), nis_y.shape)
            nis_loc = torch.stack([nis_z, nis_x, nis_y], 1).long() # N x 3
            nis_x = nis_x + tile_lt_loc[f'{i}-{j}'][0]
            nis_y = nis_y + tile_lt_loc[f'{i}-{j}'][1]
            nismask_dis_to_bcenter = ((nis_z-brain_center[0])**2+(nis_x-brain_center[1])**2+(nis_y-brain_center[2])**2).sqrt()
            print(nismask_dis_to_bcenter.min(), nismask_dis_to_bcenter.max(), nismask_dis_to_bcenter.shape)
            # xs, xe = torch.where(around_nis.any(0).any(0))[0][[0,-1]]
            # ys, ye = torch.where(around_nis.any(0).any(1))[0][[0,-1]]
            # zs, ze = torch.where(around_nis.any(1).any(1))[0][[0,-1]]
            # around_nis = around_nis.cpu()
            # print(around_nis.shape, around_nis.dtype)

            # fg_mask_c1 = torch.zeros_like(binary_mask).bool()
            # fg_mask_c2 = torch.zeros_like(binary_mask).bool()
            # binary_mask = binary_mask.detach().cpu()
            # th_c1s = {}
            # th_c2s = {}
            # nis_c1_label_dict = {}
            # nis_c2_label_dict = {}
            for di, (dmin, dmax) in tqdm(enumerate(dranges), total=len(dranges), desc=f'{datetime.now()} Get pixel around NIS'):
            #     print(dmin, dmax)
                if dmin > nismask_dis_to_bcenter.max() or dmax < nismask_dis_to_bcenter.min(): continue
                dmask = torch.logical_and(nismask_dis_to_bcenter >= dmin, nismask_dis_to_bcenter < dmax)
                loc1, loc2, loc3 = nis_loc[dmask].T.cpu()
                # dmask = torch.logical_and(grid_dis_to_center >= dmin, grid_dis_to_center < dmax)
                # if not dmask.any(): continue
                # dmask = dmask.repeat_interleave(int(1/down_ratio),0).repeat_interleave(int(1/down_ratio),1).repeat_interleave(int(1/down_ratio),2)
                # if dmask.shape[0] < around_nis.shape[0]:
                #     dmask = torch.cat([dmask, torch.zeros(around_nis.shape[0]-dmask.shape[0], dmask.shape[1], dmask.shape[2], device=dmask.device)], 0)
                # elif dmask.shape[0] > around_nis.shape[0]:
                #     dmask = dmask[:around_nis.shape[0]]
                # if dmask.shape[1] < around_nis.shape[1]:
                #     dmask = torch.cat([dmask, torch.zeros(dmask.shape[0], around_nis.shape[1]-dmask.shape[1], dmask.shape[2], device=dmask.device)], 1)
                # elif dmask.shape[1] > around_nis.shape[1]:
                #     dmask = dmask[:, :around_nis.shape[1]]
                # if dmask.shape[2] < around_nis.shape[2]:
                #     dmask = torch.cat([dmask, torch.zeros(dmask.shape[0], dmask.shape[1], around_nis.shape[2]-dmask.shape[2], device=dmask.device)], 2)
                # elif dmask.shape[2] > around_nis.shape[2]:
                #     dmask = dmask[..., :around_nis.shape[2]]
                # mask = torch.logical_and(dmask, around_nis).cpu()
                # # dmask = dmask.cpu()
                # if not mask.any(): 
                #     print(dmin, dmax, 'no NIS')
                #     continue
                c1 = img3d_c1[loc1, loc2, loc3]#.cpu().numpy()
                # th_c1 = threshold_otsu(image=c1, nbins=nbins)
                c2 = img3d_c2[loc1, loc2, loc3]#.cpu().numpy()
                # th_c2 = threshold_otsu(image=c2, nbins=nbins)
                brain_c1[f'{dmin},{dmax}'][ijkey] = c1
                brain_c2[f'{dmin},{dmax}'][ijkey] = c2
                print('Store pixels', c1.shape, c1.max(), c1.min(), c2.shape, c2.max(), c2.min())
            #     print(c1.shape, th_c1, th_c2)
                # fg_mask_c1[torch.logical_and(torch.logical_and(img3d_c1>th_c1, dmask), binary_mask)] = True
                # fg_mask_c2[torch.logical_and(torch.logical_and(img3d_c2>th_c2, dmask), binary_mask)] = True
                # th_k = f'[[{i},{j}],[{dmin},{dmax}],[{sum([len(nis) for nis in nis_avg_c1_ls])}]]'
                # th_c1s[th_k] = th_c1
                # th_c2s[th_k] = th_c2
    th_c1_dict = {}
    th_c2_dict = {}
    for dkey in brain_c1:
        if len(brain_c1[dkey]) == 0: continue
        c1 = torch.cat(list(brain_c1[dkey].values())).cpu().numpy()
        th_c1 = threshold_otsu(image=c1, nbins=nbins)
        c2 = torch.cat(list(brain_c2[dkey].values())).cpu().numpy()
        th_c2 = threshold_otsu(image=c2, nbins=nbins)
        th_c1_dict[dkey] = th_c1
        th_c2_dict[dkey] = th_c2
    del brain_c1, brain_c2

    for i in col_iterator:
        for j in row_iterator:
            nis_c1_label_dict = {}
            nis_c2_label_dict = {}
            nis_dis2bcenter, nis_avg_c1_ls, nis_avg_c2_ls, nis_c1_save_fns, nis_c2_save_fns = nis_avg_items[ijkey]
            for nis_dis_bcenter, nis_avg_c1, nis_avg_c2, nis_c1_save_fn, nis_c2_save_fn in zip(nis_dis2bcenter, nis_avg_c1_ls, nis_avg_c2_ls, nis_c1_save_fns, nis_c2_save_fns):
                for dkey in th_c1_dict:
                    dmin, dmax = dkey.split(',')
                    dmin, dmax = float(dmin), float(dmax)
                    nismask = torch.logical_and(nis_dis_bcenter >= dmin, nis_dis_bcenter < dmax)#.cpu()
                    if nis_c1_save_fn not in nis_c1_label_dict: 
                        nis_c1_label_dict[nis_c1_save_fn] = torch.zeros(len(nis_dis_bcenter)).bool().to(device)
                        nis_c2_label_dict[nis_c2_save_fn] = torch.zeros(len(nis_dis_bcenter)).bool().to(device)

                    nis_c1_label_dict[nis_c1_save_fn][nismask] = nis_avg_c1[nismask] > th_c1
                    nis_c2_label_dict[nis_c2_save_fn][nismask] = nis_avg_c2[nismask] > th_c2
            
            for nis_c1_save_fn in nis_c1_label_dict:
                print('Saving', nis_c1_save_fn)
                torch.save(nis_c1_label_dict[nis_c1_save_fn].detach().cpu(), nis_c1_save_fn)
            for nis_c2_save_fn in nis_c2_label_dict:
                print('Saving', nis_c2_save_fn)
                torch.save(nis_c2_label_dict[nis_c2_save_fn].detach().cpu(), nis_c2_save_fn)

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

def independent_coloc_label(nis_intensity_avg, device, bg_intensity_avg):
    dis1 = (nis_intensity_avg[:, 1] - bg_intensity_avg[1]).abs().std()
    dis2 = (nis_intensity_avg[:, 2] - bg_intensity_avg[2]).abs().std()
    nis_intensity_avg = nis_intensity_avg.to(device)
    p__mask = nis_intensity_avg[:, 1] > (bg_intensity_avg[1] + dis1/3)
    n__mask = nis_intensity_avg[:, 1] <= (bg_intensity_avg[1] + dis1/3)
    _p_mask = nis_intensity_avg[:, 2] > (bg_intensity_avg[2] + dis2/1.5)
    _n_mask = nis_intensity_avg[:, 2] <= (bg_intensity_avg[2] + dis2/1.5)
    pp_mask = torch.logical_and(p__mask, _p_mask)
    pn_mask = torch.logical_and(p__mask, _n_mask)
    np_mask = torch.logical_and(n__mask, _p_mask)
    nn_mask = torch.logical_and(n__mask, _n_mask)
    return pn_mask.cpu(), np_mask.cpu(), pp_mask.cpu(), nn_mask.cpu()
            
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