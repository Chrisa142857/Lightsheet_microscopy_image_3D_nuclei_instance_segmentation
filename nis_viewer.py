import torch, os, re
from multiprocessing import Pool
from tqdm import trange
import vedo
import numpy as np
import nibabel as nib
# from run_whole_brain.utils import imread
from tqdm import tqdm
from PIL import Image

def nis_mask(pair_tag, brain_tag, img_r, P_tag='P4'):
    # device = 'cuda:1'
    # seg_root = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/{P_tag}/{pair_tag}/{brain_tag}'
    z_ratio = 4/2.5
    save_root = f'/cajal/ACMUSERS/ziquanw/Lightsheet/renders/{P_tag}/{pair_tag}'
    tgt_z = 718+24
    tgt_x = 0.25
    tgt_y = 0.5
    width = 200
    z_range = [int((tgt_z-24)//z_ratio), int((tgt_z+24)//z_ratio)]
    # stack_shape = [60,200,200]
    seg = torch.load(f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/{P_tag}/{pair_tag}/{brain_tag}/{brain_tag}_NIScpp_results_zmin{tgt_z-24}_seg.zip')
    tgt_x = int(seg.shape[1]*tgt_x)
    tgt_y = int(seg.shape[2]*tgt_y)
    y_range = [tgt_y-width, tgt_y+width]
    x_range = [tgt_x-width, tgt_x+width]
    mask_img = seg[:, y_range[0]:y_range[1]+1,x_range[0]:x_range[1]+1]
    stack_seg_onfg = mask_img>0
    mask_img[stack_seg_onfg] = mask_img[stack_seg_onfg]-mask_img[stack_seg_onfg].min()+1
    mask_img = torch.nn.functional.interpolate(mask_img[None, None].float(), scale_factor=[1/z_ratio, 1, 1], mode='nearest').squeeze().long()
    # stat_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/statistics/{P_tag}'
    # print('Loading all NIS')
    # orgcenter = torch.load(f'{stat_r}/{pair_tag}/{brain_tag}_nis_center.zip')#.to(device)
    # nis_z = torch.load(f'{stat_r}/{pair_tag}/{brain_tag}_nis_coordinate_d0.zip')
    # nis_y = torch.load(f'{stat_r}/{pair_tag}/{brain_tag}_nis_coordinate_d1.zip')
    # nis_x = torch.load(f'{stat_r}/{pair_tag}/{brain_tag}_nis_coordinate_d2.zip')
    # print('Loaded all NIS')
    # f1 = orgcenter[:, 0].long() <= tgt_z+(stack_shape[0]//2)
    # f2 = orgcenter[:, 0].long() >= tgt_z-(stack_shape[0]//2)
    # f3 = orgcenter[:, 1].long() <= tgt_y+(stack_shape[1]//2)
    # f4 = orgcenter[:, 1].long() >= tgt_y-(stack_shape[1]//2)
    # f5 = orgcenter[:, 2].long() <= tgt_x+(stack_shape[2]//2)
    # f6 = orgcenter[:, 2].long() >= tgt_x-(stack_shape[2]//2)
    # mask = f1 & f2 & f3 & f4 & f5 & f6
    # nis_z = (nis_z[mask] / z_ratio).long()
    # nis_y = nis_y[mask].long()
    # nis_x = nis_x[mask].long()
    # nis_coor = torch.stack([nis_z, nis_y, nis_x], 1) # X x 3 x Vol 
    # nis_onfg = nis_coor.sum(1)!=0 # X x Vol 
    # shapez = nis_z[nis_onfg].max() - nis_z[nis_onfg].min()+1
    # z_range = [nis_z[nis_onfg].min(), nis_z[nis_onfg].max()]
    # y_range = [nis_y[nis_onfg].min(), nis_y[nis_onfg].max()]
    # x_range = [nis_x[nis_onfg].min(), nis_x[nis_onfg].max()]
    # nis_z[nis_onfg] = nis_z[nis_onfg] - nis_z[nis_onfg].min()
    # nis_y[nis_onfg] = nis_y[nis_onfg] - nis_y[nis_onfg].min()
    # nis_x[nis_onfg] = nis_x[nis_onfg] - nis_x[nis_onfg].min()
    img_3d = []
    for z in trange(z_range[0], z_range[1], desc='Loading 2D images'):
        img = np.array(Image.open(f'{img_r}/{brain_tag}_{z+1:04d}_C1_topro_stitched.tif'))
        img_3d.append(img[y_range[0]:y_range[1]+1,x_range[0]:x_range[1]+1])
    
    nib.save(nib.Nifti1Image(np.stack(img_3d), np.eye(4)), f'{save_root}/{brain_tag}_{z_range[0]:04d}-{z_range[1]:04d}_{y_range[0]:04d}-{y_range[1]:04d}_{x_range[0]:04d}-{x_range[1]:04d}_nis_backimg.nii.gz')
    # mask_img = torch.zeros((shapez, y_range[1]-y_range[0]+1, x_range[1]-x_range[0]+1)).long()
    # print('Mask range:', mask_img.shape, 'NIS number:', len(nis_z))
    # mask_img[nis_z, nis_y, nis_x] = torch.stack([torch.arange(len(nis_z))+1 for _ in range(nis_z.shape[1])], 1)
    # mask_img[0,0,0] = 0
    nib.save(nib.Nifti1Image(mask_img.numpy().astype(np.uint16), np.eye(4)), f'{save_root}/{brain_tag}_{z_range[0]:04d}-{z_range[1]:04d}_{y_range[0]:04d}-{y_range[1]:04d}_{x_range[0]:04d}-{x_range[1]:04d}_nis_mask.nii.gz')

def coloc_mask(pair_tag = 'pair15', brain_tag = 'L73D766P9'):
    device = 'cuda:1'
    img_r = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    seg_root = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{pair_tag}/{brain_tag}'
    save_root = f'/cajal/ACMUSERS/ziquanw/Lightsheet/renders/{pair_tag}'
    stat_r = '/cajal/ACMUSERS/ziquanw/Lightsheet/statistics'
    coloc_fn = f'{stat_r}/{pair_tag}/{brain_tag}_NIS_colocContrastCalib_label.zip'
    seg_paths, sortks = listdir_sorted(seg_root, "NIScpp", ftail="seg.zip", sortkid=3)
    tgt_z = 707
    tgt_roi = 16001
    downsample_res = [25, 25, 25]
    img_res = [4, 0.75, 0.75]
    seg_res = [2.5, 0.75, 0.75]
    dratio = [s/d for s, d in zip(seg_res, downsample_res)]
    mask_fn = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/registered/%s_MASK_topro_25_all.nii' % (pair_tag, brain_tag, brain_tag)
    mask = load_atlas(mask_fn).to(device)
    orgcenter = torch.load(f'{stat_r}/{pair_tag}/{brain_tag}_nis_center.zip').to(device)
    coloc_label = torch.load(coloc_fn, map_location=device)
    label_names = list(coloc_label.keys())
    # with open(f'{save_root}/{brain_tag}_{tgt_z:04d}_coloc_mask_info.txt', 'w') as f:
    #     f.write(','.join(label_names)+':'+','.join([str(i+1) for i in range(len(label_names))]))
    center = orgcenter.clone()
    center[:, 0] = center[:, 0] * dratio[0]
    center[:, 1] = center[:, 1] * dratio[1]
    center[:, 2] = center[:, 2] * dratio[2]
    center = center.long().T
    is_tgt_roi = mask[center[0],center[1],center[2]] == tgt_roi    
    coloc_name = {
        'pn_mask': 'b+c-',
        'np_mask': 'b-c+',
        'pp_mask': 'b+c+',
        'nn_mask': 'b-c-'
    }
    for k in coloc_label:
        is_k = coloc_label[k][is_tgt_roi]
        tgt_roi_count = len(is_k)
        k_count = len(torch.where(is_k)[0])
        print(f"{k_count} / {tgt_roi_count} ({100*(k_count/tgt_roi_count):.2f}%) is {coloc_name[k]} in ROI id=={tgt_roi}")
    # exit()
    zstack = orgcenter[:, 0].round()
    zstack_uni = zstack.unique()
    imgzs, imgi = ((zstack_uni/(img_res[0]/seg_res[0])).long()+1).unique(return_inverse=True)
    imgzs, imgi = imgzs.tolist(), imgi.tolist()
    load_segp = None
    for si in range(len(sortks)):
        if sortks[si] <= imgi.index(tgt_z):
            load_segp = seg_paths[si]
        else:
            break
    stitch_remap_all = torch.load(f"{seg_root}/{brain_tag}_remap.zip")#.to(device) # old id to new id map
    assert imgi.index(tgt_z)-sortks[seg_paths.index(load_segp)] >= 0, f"{tgt_z}-{sortks[seg_paths.index(load_segp)]}={tgt_z-sortks[seg_paths.index(load_segp)]}"
    seg = torch.load(load_segp)[imgi.index(tgt_z)-sortks[seg_paths.index(load_segp)]].to(device)
    total_label = torch.load(f"{stat_r}/{pair_tag}/{brain_tag}_nis_index.zip").to(device)
    max_label = seg.max()
    min_label = seg[seg>0].min()
    print("seg.shape, max_label, min", seg.shape, max_label.cpu().detach().item(), min_label.cpu().detach().item())
    f1 = stitch_remap_all[0]<=max_label.item()
    f2 = stitch_remap_all[0]>=min_label.item()
    f = f1 & f2
    stitch_remap = stitch_remap_all[:, f].to(device)
    side = 12
    for si in trange(stitch_remap.shape[1], desc=f'Stitching tgt z {tgt_z}'):
        f = total_label==stitch_remap[1, si]
        _,y,x = orgcenter[f][0].round().long()
        sy, sx = torch.where(seg[y-side:y+side, x-side:x+side]==stitch_remap[0, si])
        sy = sy + y - side
        sx = sx + x - side
        seg[sy, sx] = stitch_remap[1, si]
    coloc_seg = torch.zeros_like(seg)
    nis_segid = seg.unique()
    for sid in tqdm(nis_segid, desc='Remap NIS with coloc label'):
        if sid == 0: continue
        f = total_label==sid
        _,y,x = orgcenter[f][0].round().long()
        sy, sx = torch.where(seg[y-side:y+side, x-side:x+side]==sid)
        sy = sy + y - side
        sx = sx + x - side
        for k in coloc_label:
            if coloc_label[k][f].item():
                coloc_seg[sy, sx] = label_names.index(k)+1
                break
    
    Image.fromarray(coloc_seg.detach().cpu().numpy().astype(np.uint16)).save(f'{save_root}/{brain_tag}_{tgt_z:04d}_coloc_mask.tif')

def mask2mesh():
    device = 'cuda:1'
    brain_tag = 'L73D766P9'
    pair_tag = 'pair15'
    stat_r = '/cajal/ACMUSERS/ziquanw/Lightsheet/statistics'
    seg_root = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{pair_tag}/{brain_tag}'
    save_root = f'/cajal/ACMUSERS/ziquanw/Lightsheet/renders/{pair_tag}'
    seg_paths, sortks = listdir_sorted(seg_root, "NIScpp", ftail="seg.zip", sortkid=3)
    seg_paths = [[p] for p in seg_paths]
    stitch_remap_all = torch.load(f"{seg_root}/{brain_tag}_remap.zip").to(device) # old id to new id map
    total_label = torch.load(f"{stat_r}/{pair_tag}/{brain_tag}_nis_index.zip").to(device)
    orgcenter = torch.load(f'{stat_r}/{pair_tag}/{brain_tag}_nis_center.zip').to(device)

    load_len = 15#len(seg_paths)
    # print("Load z range:", sortks[0], sortks[load_len])
    with Pool(min(load_len, 5)) as p:
        nis_stack = list(p.starmap(torch.load, tqdm(seg_paths[:load_len], desc='Load all NIS mask')))
    # nis_stack = []
    # for sp in tqdm(seg_paths[:load_len], desc='Load all NIS mask'):
    #     nis_stack.append(torch.load(sp[0]).cpu())
    # depth_list = torch.LongTensor([nis.shape[0] for nis in nis_stack]).cumsum(0)[:-1]
    print('Done, loaded all NIS mask')
    nis_stack = torch.cat(nis_stack)#.cpu()
    # group_num = 3
    # depths = []
    # for i in range(len(depth_list)):
    #     if i//group_num >= len(depths): depths.append([])
    #     depths[i//group_num].append(depth_list[i])
    # for depth in depths:
    #     strip_ind = []
    #     for d in depth:
    #         strip_ind.extend([i for i in range(d-1, d+8)])
    #     strip_ind = torch.LongTensor(strip_ind)
    #     nis_strip = nis_stack[strip_ind].to(device)
    #     max_label = nis_strip.max()
    #     min_label = nis_strip.min()
    #     print("nis_strip.shape, max_label, min", nis_strip.shape, max_label.cpu().detach().item(), min_label.cpu().detach().item())
    #     f1 = stitch_remap_all[0]<=max_label
    #     f2 = stitch_remap_all[0]>=min_label
    #     f = f1 & f2
    #     stitch_remap = stitch_remap_all[:, f]
    side = 12
    for si in trange(stitch_remap_all.shape[1], desc=f'Stitching'):
        f = total_label==stitch_remap_all[1, si]
        z, y, x = orgcenter[f][0].round().long()
        sz, sy, sx = torch.where(nis_stack[z-side:z+side, y-side:y+side, x-side:x+side]==stitch_remap_all[0, si].cpu())
        sz = sz + z.cpu() - side
        sy = sy + y.cpu() - side
        sx = sx + x.cpu() - side
        nis_stack[sz, sy, sx] = stitch_remap_all[1, si].cpu()
        # z, y, x = torch.where(nis_strip==stitch_remap[0, si])
        # nis_stack[strip_ind[z.cpu()], y.cpu(), x.cpu()] = stitch_remap[1, si].cpu()
        # stitch_nis = torch.zeros_like(nis_stack, dtype=torch.bool)
        # stitch_nis[strip_ind] = (nis_strip==stitch_remap[0, si]).detach().cpu()
        # nis_stack[stitch_nis] = stitch_remap[1, si].cpu()
    
    nis_mesh = vedo.Volume(nis_stack).isosurface()
    nis_mesh.write(f"{save_root}/NIS_mesh_{brain_tag}.ply")



def load_atlas(mask_fn):
    orig_roi_mask = torch.from_numpy(np.transpose(nib.load(mask_fn).get_fdata(), (2, 0, 1))[:, :, ::-1].copy())
    return orig_roi_mask

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

if __name__=='__main__':
    # mask2mesh()
    # coloc_mask()
        
    # pair_tag = 'pair10'
    # brain_tag = 'L64D804P3'
    # coloc_mask(pair_tag, brain_tag)
    # # # 
    # pair_tag = 'pair10'
    # brain_tag = 'L64D804P9'
    # coloc_mask(pair_tag, brain_tag)

    pair_tag = 'female'
    brain_tag = 'L106P3'
    P_tag = 'P14'
    img_r = f'/lichtman/Ian/Lightsheet/P14/stitched/{pair_tag}/{brain_tag}/stitched'
    nis_mask(pair_tag, brain_tag, img_r, P_tag)