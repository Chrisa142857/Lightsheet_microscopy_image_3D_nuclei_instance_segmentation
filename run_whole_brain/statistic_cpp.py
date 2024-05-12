import torch
from utils import imread
import os
import re
import numpy as np
from multiprocessing import Pool
from tqdm import trange, tqdm
from datetime import datetime
# from kmeans_pytorch import kmeans
import h5py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import nibabel as nib

def main(data_root, pair_tag, brain_tag, img_tags=["C2_"], P_tag='P4'):
    device = 'cuda:2'
    downsample_res = [25, 25, 25]
    seg_res = [2.5, 0.75, 0.75]
    seg_pad = (4, 20, 20) # hard coded in CPP
    # seg_root = f"/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{pair_tag}/{brain_tag}"
    # img_root = f"/{data_root}/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched"
    # save_root = f"/cajal/ACMUSERS/ziquanw/Lightsheet/statistics/{pair_tag}"
    seg_root = f"/cajal/ACMUSERS/ziquanw/Lightsheet/results/{P_tag}/{pair_tag}/{brain_tag}"
    img_root = f"{data_root}"
    save_root = f"/cajal/ACMUSERS/ziquanw/Lightsheet/statistics/{P_tag}/{pair_tag}"
    ftail = "seg_meta.zip"
    os.makedirs(save_root, exist_ok=True)
    img_paths_list = []
    for img_tag in img_tags:
        img_paths, _ = listdir_sorted(img_root, img_tag)
        img_paths_list.append(img_paths)
    
    seg_paths, sortks = listdir_sorted(seg_root, "NIScpp", ftail=ftail, sortkid=3)

    # instance_statistic = [[] for _ in range(len(img_tags))]
    if not os.path.exists(f"{save_root}/{brain_tag}_nis_{img_tag}intensity.zip") or True:
    #     seg_depths = [sortks[i+1]-sortks[i] for i in range(len(sortks)-1)]
    #     seg_depths += [torch.load(seg_paths[-1]).shape[0]]
    #     seg_whole_shape = [seg_depths[-1]+sortks[-1]]
    #     stack_paths_list = []
    #     for img_paths in img_paths_list:
    #         img_whole_shape = [len(img_paths_list[0])]
    #         scale_r = [i/s for s, i in zip(seg_whole_shape, img_whole_shape)]
    #         stack_paths = []
    #         for i in range(len(seg_paths)):
    #             zmin = sortks[i]
    #             load_path = []
    #             for j in range(seg_depths[i]):
    #                 j = j + zmin
    #                 img_path = img_paths[int(scale_r[0]*j)]
    #                 load_path.append([img_path])
    #             stack_paths.append(load_path)
    #         stack_paths_list.append(stack_paths)

        stitch_remap = torch.load(f"{seg_root}/{brain_tag}_remap.zip").to(device) # old id to new id map
        total_vol = []
        total_center = []
        total_label = []
        total_pt = []
        remap_src_loc = []
        remap_tgt_loc = []
        nis_num = 0
        pre_nis_num = None
        pre_label = None
        remove_nis = []
        for i in trange(len(seg_paths), desc=f"Start statistics for {pair_tag} {brain_tag} of {img_tags} intensity"):
            seg_path = seg_paths[i]
            zmin = sortks[i]
            # print(datetime.now(), f"zmin: {zmin}, zmax: {zmin+seg_depths[i]}")
            label_path = seg_path.replace(ftail, 'instance_label.zip')
            vol_path = seg_path.replace(ftail, 'instance_volume.zip')
            ct_path = seg_path.replace(ftail, 'instance_center.zip')
            pt_path = seg_path.replace(ftail, 'instance_coordinate.zip')
            ## QC: correct previous wrong results
            # seg = torch.load(seg_path)#.to(device)
            # z, y, x = torch.meshgrid([torch.arange(seg.shape[dimi]) for dimi in range(len(seg.shape))])
            # nis_mask = seg>0
            # label, vol = seg[nis_mask].unique(return_counts=True)
            # print(datetime.now(), f"Max volume: {vol.max().item()}")
            # torch.save(label, label_path)
            # torch.save(vol, vol_path)
            # splits = vol.cumsum(0).cpu()
            # assert vol.sum() == nis_mask.sum()
            # z, y, x = z[nis_mask], y[nis_mask], x[nis_mask]
            # sorted_nis = seg[nis_mask].argsort()
            # z, y, x = z[sorted_nis], y[sorted_nis], x[sorted_nis]
            # pt = torch.stack([z, y, x], -1)
            # torch.save(pt, pt_path)
            # pt = torch.tensor_split(pt.cpu(), splits)[:-1]
            # ct = torch.stack([(p.max(0)[0] + p.min(0)[0])/2 for p in pt])
            # torch.save(ct, ct_path)
            ## Load
            label = torch.load(label_path).to(device)
            vol = torch.load(vol_path)#.to(device)
            splits = vol.cumsum(0)#.cpu()
            pt = torch.load(pt_path).to(torch.int16)#.to(device)
            pt[:, 0] = pt[:, 0] + zmin
            pt = torch.tensor_split(pt, splits)[:-1]
            ct = torch.load(ct_path)#.to(device)
            ct[:, 0] = ct[:, 0] + zmin
            ######
            # assert z.min() >= 0 and z.max() < seg_depths[i], f"mask depth {seg_depths[i]}, while z.min(): {z.min()} and z.max(): {z.max()}"
            # img_stack_list = []
            # for tagi, stack_paths in enumerate(stack_paths_list):
            #     load_path = stack_paths[i]
            #     print(datetime.now(), f"Load images {img_tags[tagi]}")
            #     with Pool(30) as p:
            #         img_stack = list(p.starmap(imread, load_path))
            #     img_stack = torch.from_numpy(np.stack(img_stack).astype(np.int32)).to(device)
            #     img_stack_list.append(img_stack[z, y, x].cpu())
            
            # print(datetime.now(), f"Start statistics:")
            #### Apply stitch
            # for tagi, img_stack in enumerate(img_stack_list):
            #     instance_statistic[tagi] += list(torch.tensor_split(img_stack, splits)[:-1]).cpu()
            print(datetime.now(), f"Get all intensity")
            assert len(vol) == len(label), f"{len(vol)} != {len(label)}"
            total_pt.extend(pt)
            total_center.append(ct)
            total_vol.append(vol)
            label = label.to(device)
            total_label.append(label.cpu())
            assert len(label.unique()) == len(label), f"{len(label.unique())} != {len(label)}"
            # mask = stitch_remap[0]<=label.max()
            # mask = mask & (stitch_remap[0]>=label.min())
            # src_stitch_remap = stitch_remap[:, mask]
            # for j in trange(src_stitch_remap.shape[1], desc='Find stitch-needed NIS'):
            #     sloc = torch.where(src_stitch_remap[0, j] == label)[0]
            #     if len(sloc) == 1:
            #         tloc = torch.where(pre_label==stitch_remap[1, j])[0]
            #         if len(tloc) == 1:
            #             remap_src_loc.append(sloc.item()+nis_num)    
            #             remap_tgt_loc.append(tloc.item()+pre_nis_num)
                
            for j in trange(len(label), desc='Find stitch-needed NIS'):
                loc = torch.where(label[j] == stitch_remap[0])[0]
                if len(loc) == 1: 
                    tloc = torch.where(pre_label==stitch_remap[1, loc])[0]
                    if len(tloc) < 1:
                        remove_nis.append(j+nis_num)
                    else:
                        remap_src_loc.append(j+nis_num)
                        remap_tgt_loc.append(tloc.item()+pre_nis_num)
            pre_nis_num = nis_num
            nis_num += len(vol)
            pre_label = label
            
        print(datetime.now(), "Concat all NIS")
        total_center = torch.cat(total_center)
        total_vol = torch.cat(total_vol)
        total_label = torch.cat(total_label)
        assert total_center.shape[0] == total_label.shape[0] == total_vol.shape[0], f"{total_center.shape[0]} {total_label.shape[0]} {total_vol.shape[0]}"
        print(datetime.now(), f"Apply remap to center and volume, shape: {total_label.shape}")
        remain_loc = torch.ones_like(total_label).bool()
        remain_loc[remap_src_loc] = False
        remain_loc[remove_nis] = False
        total_center = total_center.float()
        total_center[remap_tgt_loc] = (total_center[remap_src_loc] + total_center[remap_tgt_loc])/2
        total_center = total_center.to(torch.int16)
        total_vol[remap_tgt_loc] = total_vol[remap_src_loc] + total_vol[remap_tgt_loc]
        pt_remove_loc = []
        cumsum_vol = total_vol.cumsum(0)
        for sloc in remove_nis:
            pt_remove_loc.append(torch.arange(cumsum_vol[sloc-1] if sloc > 0 else 0, cumsum_vol[sloc]))
        for loci in trange(len(remap_src_loc), desc="Apply remap to coordinates and intensity"):
            sloc, tloc = remap_src_loc[loci], remap_tgt_loc[loci]
            pt_remove_loc.append(torch.arange(cumsum_vol[sloc-1] if sloc > 0 else 0, cumsum_vol[sloc]))
            total_pt[tloc] = torch.cat([total_pt[sloc], total_pt[tloc]])
            
        pt_remove_loc = torch.cat(pt_remove_loc)
        print(datetime.now(), "Remove stitched NIS")
        total_label = total_label[remain_loc]
        total_center = total_center[remain_loc]
        total_vol = total_vol[remain_loc]
        total_pt = torch.cat(total_pt)
        pt_mask = torch.ones_like(total_pt[:, 0], dtype=bool)
        pt_mask[pt_remove_loc] = False
        total_pt = total_pt[pt_mask, :]
        # for tagi in range(len(img_tags)):
        #     instance_statistic[tagi] = torch.cat([instance_statistic[tagi][i] for i in torch.where(remain_loc)[0]])

        print(datetime.now(), f"Save")
        torch.save(total_label, f"{save_root}/{brain_tag}_nis_index.zip")
        torch.save(total_center, f"{save_root}/{brain_tag}_nis_center.zip")
        torch.save(total_vol, f"{save_root}/{brain_tag}_nis_volume.zip")
        with h5py.File(f"{save_root}/{brain_tag}_nis_coordinate.h5", 'w') as f:
            f.create_dataset('data', shape=total_pt.shape, dtype=int, data=total_pt.numpy(), chunks=True)
        # torch.save(total_pt[..., 0], f"{save_root}/{brain_tag}_nis_coordinate_d0.zip")
        # torch.save(total_pt[..., 1], f"{save_root}/{brain_tag}_nis_coordinate_d1.zip")
        # torch.save(total_pt[..., 2], f"{save_root}/{brain_tag}_nis_coordinate_d2.zip")
        # for tagi, img_tag in enumerate(img_tags):
        #     torch.save(instance_statistic[tagi], f"{save_root}/{brain_tag}_nis_{img_tag}intensity.zip")
    else:
        print(datetime.now(), f"Already applied remap, loading")
        total_center = torch.load(f"{save_root}/{brain_tag}_nis_center.zip")
        total_vol = torch.load(f"{save_root}/{brain_tag}_nis_volume.zip")
        # total_vol = torch.load(f"{save_root}/{brain_tag}_nis_volume.zip")
        for tagi, img_tag in enumerate(img_tags):
            # if img_tag != 'C1_':
            #     instance_statistic[tagi] = torch.load(f"{save_root}/{brain_tag}_nis_{img_tag}intensity.zip")
            instance_statistic[tagi] = torch.load(f"{save_root}/{brain_tag}_nis_{img_tag}intensity.zip")
    return 
    # num_clusters = 3
    dratio = [s/d for s, d in zip(seg_res, downsample_res)]
    sshape = torch.load(seg_paths[-1]).shape
    sshape = list(sshape)
    sshape[0] = sshape[0] + sortks[-1]
    ## For visualization purpose
    # dshape = [0,0,0]
    # dshape[0] = int(sshape[0] * dratio[0])
    # dshape[1] = int(sshape[1] * dratio[1])
    # dshape[2] = int(sshape[2] * dratio[2])
    # print(datetime.now(), f"Downsample space shape: {dshape}, ratio: {dratio}")
    # density, vol_avg, local_loc_list = downsample(total_center, total_vol, dratio, dshape, device)
    # new_header, affine_m = init_nib_header()
    # nib.save(nib.Nifti1Image(density.numpy().astype(np.float32), affine_m, header=new_header), f'{save_root}/NIS_density_{pair_tag}_{brain_tag}.nii')
    # nib.save(nib.Nifti1Image(vol_avg.numpy().astype(np.float32), affine_m, header=new_header), f'{save_root}/NIS_volavg_{pair_tag}_{brain_tag}.nii')
    # print(datetime.now(), f"Saved {pair_tag} {brain_tag} denstiy and vol-avg")
    ## For statistics purpose
    dratio = [r/2 for r in dratio]
    dshape = [0,0,0]
    dshape[0] = int(sshape[0] * dratio[0])
    dshape[1] = int(sshape[1] * dratio[1])
    dshape[2] = int(sshape[2] * dratio[2])
    # instance_statistic = [instance_statistic[i].sort(1,descending=True)[0][:,:5].to(device) for i in range(len(instance_statistic))]
    print(datetime.now(), f"Downsample space shape: {dshape}, ratio: {dratio}, for local intensity normalization")
    _, _, local_loc_list = downsample(total_center, total_vol, dratio, dshape, device, True)
    # nis_label = torch.zeros(len(instance_statistic[1])).long()
    # for li in trange(len(local_loc_list), desc="Kmeans each local cube"):
    #     local_loc  = local_loc_list[li]
    #     kmeans_x = []
    #     for tagi, img_tag in enumerate(img_tags):
    #         if img_tag != 'C1_':
    #             x = norm_tensor(instance_statistic[tagi][local_loc].float()).to(device)#.float().to(device)
    #             # x = instance_statistic[tagi][local_loc].float().to(device)
    #         else:
    #             x = torch.zeros(len(local_loc), 1).to(device)
    #         # print(x)
    #         kmeans_x.append(x.mean(1))
    #     # exit()
    #     kmeans_x = torch.stack(kmeans_x, 1)
    #     if li == 0: print(datetime.now(), f"Kmeans input shape: {kmeans_x.shape}")
    #     out = coloc_nis(kmeans_x, device, li, tag=img_tag, num_clusters=num_clusters, save_root=save_root)
    #     if li == 0: print(datetime.now(), f"Kmeans output bincount: {out.bincount()}")
    #     nis_label[local_loc] = out
    out, hist_fig = coloc_nis(device, instance_statistic, local_loc_list, img_tags)
    torch.save(out, f"{save_root}/{brain_tag}_nis_normed_mean_{''.join(img_tags)}intensity.zip")
    # torch.save(nis_label, f"{save_root}/{brain_tag}_nis_coloc_label.zip")
    # hist_fig.savefig(f"{save_root}/{brain_tag}_nis_c12_abs_diff_hist.png")
    # plt.close()
    print(datetime.now(), f"Saved coloc_out: {out.shape}")
    # print(datetime.now(), f"Saved kmeans coloc_label: {nis_label.shape}")

    # for img_tag in img_tags:
    #     x = instance_statistic[tagi]
    #     nis_label = torch.zeros(len(x)).long()
    #     for li in trange(len(local_loc_list), desc="Kmeans each local cube"):
    #         local_loc  = local_loc_list[li]
    #         nis_label[local_loc] = coloc_nis(x[local_loc].float(), device, li, num_clusters=2, save_root=save_root)
    #     torch.save(nis_label, f"{save_root}/{brain_tag}_nis_{img_tag}label.zip")
    #     print(datetime.now(), f"Saved kmeans {img_tag}label: {nis_label.shape}")

def norm_tensor(x):
    fg = x>0
    fgx = x[fg]
    x[fg] = (fgx-fgx.min())/(fgx.max()-fgx.min())
    return x

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




def downsample(center, vol, ratio, dshape, device, only_loc=False):
    min_nis = 100
    center = center.to(device)
    vol = vol.float().to(device)
    center[:,0] = center[:,0] * ratio[0]
    center[:,1] = center[:,1] * ratio[1]
    center[:,2] = center[:,2] * ratio[2]
    z = center[:, 0].clip(min=0, max=dshape[0]-0.9)
    y = center[:, 1].clip(min=0, max=dshape[1]-0.9)
    x = center[:, 2].clip(min=0, max=dshape[2]-0.9)
    # print(center.shape)
    loc = torch.arange(dshape[0]*dshape[1]*dshape[2]).view(dshape[0], dshape[1], dshape[2]).to(device) 
    loc = loc[(z.long(), y.long(), x.long())] # all nis location in the downsample space
    loc_count = loc.bincount() 
    loc_count = loc_count[loc_count!=0] 
    atlas_loc = loc.unique().to(device) # unique location in the downsample space
    ## volume avg & local intensity
    vol_avg = None
    if not only_loc:
        vol_avg = torch.zeros(dshape[0]*dshape[1]*dshape[2], dtype=torch.float64).to(device)
    nis_downsample_loc = []
    
    for loci in tqdm(atlas_loc, desc="Collect NIS property in local cube"): 
        where_loc = torch.where(loc==loci)[0]
        if not only_loc:
            vol_avg[loci] = vol[where_loc].mean()
        if len(nis_downsample_loc) > 0 and len(nis_downsample_loc[-1]) <= min_nis:
            nis_downsample_loc[-1] = torch.cat([nis_downsample_loc[-1], where_loc.cpu()])
        else:
            nis_downsample_loc.append(where_loc.cpu())
    if len(nis_downsample_loc[-1]) <= min_nis: nis_downsample_loc = nis_downsample_loc[:-1]
    if not only_loc:
        vol_avg = vol_avg.view(dshape[0], dshape[1], dshape[2]).cpu()
    ## density map
    density = None
    if not only_loc:
        density = torch.zeros(dshape[0]*dshape[1]*dshape[2], dtype=torch.float64).to(device)
        density[atlas_loc] = loc_count.double() / center.shape[0]
        density = density.view(dshape[0], dshape[1], dshape[2]).cpu()
    return density, vol_avg, nis_downsample_loc


def coloc_nis(device, instance_statistic, local_loc_list, img_tags):
    '''
     if channels = [a, b, c], then 
     label name = ['b+c+', 'b+c-', 'b-c+', 'b-c-'] with 'a' always +
     label id = [0, 1, 2, 3]
    '''
    labelr = [0.05, 0.15, 0.15, 0.65]
    tag1_big_list, absdis_list, max_c1_c2_list = [], [], []
    norm_mean_intensity = []
    nis_label_sorted = torch.zeros(len(instance_statistic[1])).long()
    for li in trange(len(local_loc_list), desc="Norm each local cube"):
        local_loc  = local_loc_list[li]
        xs = []
        for tagi, img_tag in enumerate(img_tags):
            # if img_tag != 'C1_':
            #     x = norm_tensor(instance_statistic[tagi][local_loc].float()).to(device)#.float().to(device)
            #     # x = instance_statistic[tagi][local_loc].float().to(device)
            # else:
            #     x = torch.zeros(len(local_loc), 1).to(device)
            x = norm_tensor(instance_statistic[tagi][local_loc].float()).to(device)#.float().to(device)
            # print(x)
            xs.append(x.mean(1))
        # exit()
        xs = torch.stack(xs, 1)
        norm_mean_intensity.append(xs)
        continue
        # if li == 0: print(datetime.now(), f"Coloc input shape: {xs.shape}")
        # out = coloc_nis(kmeans_x, device, li, tag=img_tag, num_clusters=num_clusters, save_root=save_root)
        max_c1_c2 = xs[:, 1:].max(1)[0]
        max_c1_c2_list.append(max_c1_c2)
        d = xs[:, 1] - xs[:, 2]
        tag1bigger = d>0
        tag1_big_list.append(tag1bigger)
        absd = d.abs()
        absdis_list.append(absd)
    return torch.cat(norm_mean_intensity), None
        # if li == 0: print(datetime.now(), f"Kmeans output bincount: {out.bincount()}")
        # nis_label[local_loc] = out
    local_loc_list = torch.cat(local_loc_list)
    max_c1_c2_list = torch.cat(max_c1_c2_list)
    tag1_big_list = torch.cat(tag1_big_list)
    absdis_list = torch.cat(absdis_list)
    assert absdis_list.min() >= 0, f"absdis_list.min(): {absdis_list.min()}, absdis_list.max(): {absdis_list.max()}, absdis_list.mean(): {absdis_list.mean()}"
    # Compute the histogram
    hist, bins = torch.histogram(absdis_list.cpu(), bins=100)
    # Plot the histogram
    fig = plt.figure()
    plt.bar(bins[:-1], hist)
    plt.xlabel('Absolute diff of channels')
    plt.ylabel('Frequency')
    plt.xlim([absdis_list.min().cpu().item(), absdis_list.max().cpu().item()])
    plt.ylim([hist.min().item(), hist.max().item()])
    print(datetime.now(), f"Coloc input shape of absolute diff between two channels: {absdis_list.shape}")
    nis_label= torch.zeros(len(local_loc_list)).long()
    dis_sortid = absdis_list.argsort(descending=True)
    sortid_c1_big = torch.where(tag1_big_list[dis_sortid])[0]
    sortid_c2_big = torch.where(torch.logical_not(tag1_big_list[dis_sortid]))[0]
    label1 = dis_sortid[sortid_c1_big[:int(len(nis_label)*labelr[1])]]
    label2 = dis_sortid[sortid_c2_big[:int(len(nis_label)*labelr[2])]]
    label_0or3 = dis_sortid[torch.cat([
        sortid_c1_big[int(len(nis_label)*labelr[1]):], 
        sortid_c2_big[int(len(nis_label)*labelr[2]):]
    ])]
    max_c1_c2_sorid = max_c1_c2_list[label_0or3].argsort(descending=True)
    label0 = label_0or3[max_c1_c2_sorid[:int(len(label_0or3)*(labelr[0]/(labelr[0]+labelr[3])))]]
    label3 = label_0or3[max_c1_c2_sorid[int(len(label_0or3)*(labelr[0]/(labelr[0]+labelr[3]))):]]
    nis_label[label0] = 0
    nis_label[label1] = 1
    nis_label[label2] = 2
    nis_label[label3] = 3
    nis_label_sorted[local_loc_list] = nis_label.cpu()
    return nis_label_sorted, fig

    d = x[:, 1] - x[:, 2]
    absd = d.abs()
    lowb = 2*max(absd.mean() - absd.std(), 0)
    label = torch.zeros_like(d).long()
    f1 = d > 0 
    f2 = absd > 2*lowb
    f = f1 & f2
    label[f] = 1
    f3 = d < 0 
    f = f3 & f2
    label[f] = 2
    return label.cpu()
    # kmeans
    cluster_ids, cluster_centers = kmeans(
        X=x, num_clusters=num_clusters, distance='cosine', device=torch.device(device)
    )
    # assert num_clusters == 2
    # ## Make sure 0 is background
    # if x[cluster_ids==0].mean() > x[cluster_ids==1].mean():
    #     new_x = x.clone()
    #     new_x[x==0] = 1
    #     new_x[x==1] = 0 
    #     x = new_x
    ## Sort cluster
    l = int(cluster_centers.shape[1]/3)
    c1 = cluster_centers[:, :l].mean(1).argmax()
    c2 = cluster_centers[:, l:2*l].mean(1).argmax()
    c3 = cluster_centers[:, 2*l:].mean(1).argmax()
    new_x = cluster_ids.clone()
    new_x[cluster_ids==c1] = 0
    new_x[cluster_ids==c2] = 1
    new_x[cluster_ids==c3] = 2
    # if loci % 1000 == 0:
    #     plt.figure()
    #     x_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3, n_jobs=30, verbose=0).fit_transform(torch.cat([cluster_centers, x]).numpy())
    #     x_ct = x_embedded[:num_clusters]
    #     x_embedded = x_embedded[num_clusters:]
    #     for cid in range(num_clusters):
    #         pt = x_embedded[cluster_ids==cid]
    #         plt.scatter(pt[:, 0], pt[:, 1], label=f"cluster-{cid}", s=1)
        
    #     for cid in range(num_clusters):
    #         plt.scatter(x_ct[cid, 0], x_ct[cid, 1], marker="*", label=f"cluster-center-{cid}", s=20)
    #     plt.legend()
    #     plt.savefig(f"{save_root}/{tag}kmeans_loc{loci}.png")
    return new_x

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



def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        device=torch.device('cpu')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    # print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    initial_state = initialize(X, num_clusters)
    max_iter = 1e+3
    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # # update tqdm meter
        # tqdm_meter.set_postfix(
        #     iteration=f'{iteration}',
        #     center_shift=f'{center_shift ** 2:0.6f}',
        #     tol=f'{tol:0.6f}'
        # )
        # tqdm_meter.update()
        if center_shift ** 2 < tol or iteration > max_iter:
            break

    return choice_cluster.cpu(), initial_state.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis




'''
    -rw-r--r-- 1 ziquanw acm 36M Dec 15 22:31 pair10/L64D804P3/L64D804P3_remap.zip **
    -rw-r--r-- 1 ziquanw acm 38M Dec 16 15:48 pair10/L64D804P9/L64D804P9_remap.zip **
    -rw-r--r-- 1 ziquanw acm 33M Dec 17 07:38 pair11/L66D764P3/L66D764P3_remap.zip
    -rw-r--r-- 1 ziquanw acm 41M Dec 18 07:00 pair11/L66D764P8/L66D764P8_remap.zip
    -rw-r--r-- 1 ziquanw acm 39M Dec 19 00:00 pair12/L66D764P5/L66D764P5_remap.zip
    -rw-r--r-- 1 ziquanw acm 34M Dec 19 15:14 pair12/L66D764P6/L66D764P6_remap.zip
    -rw-r--r-- 1 ziquanw acm 37M Dec  2 07:12 pair13/L69D764P6/L69D764P6_remap.zip **
    -rw-r--r-- 1 ziquanw acm 42M Dec  3 11:57 pair13/L69D764P9/L69D764P9_remap.zip **
    -rw-r--r-- 1 ziquanw acm 31M Dec 20 05:43 pair14/L73D766P5/L73D766P5_remap.zip
    -rw-r--r-- 1 ziquanw acm 35M Dec 20 21:39 pair14/L73D766P7/L73D766P7_remap.zip
    -rw-r--r-- 1 ziquanw acm 31M Dec 13 08:00 pair15/L73D766P4/L73D766P4_remap.zip **
    -rw-r--r-- 1 ziquanw acm 35M Dec 13 23:13 pair15/L73D766P9/L73D766P9_remap.zip **
    -rw-r--r-- 1 ziquanw acm 36M Dec 22 02:35 pair16/L74D769P4/L74D769P4_remap.zip
    -rw-r--r-- 1 ziquanw acm 39M Dec 22 19:17 pair16/L74D769P8/L74D769P8_remap.zip
    -rw-r--r-- 1 ziquanw acm 28M Dec 12 03:40 pair17/L77D764P2/L77D764P2_remap.zip **
    -rw-r--r-- 1 ziquanw acm 33M Dec 12 18:20 pair17/L77D764P9/L77D764P9_remap.zip **
    -rw-r--r-- 1 ziquanw acm 31M Jan  1 01:53 pair18/L77D764P4/L77D764P4_remap.zip
    -rw-r--r-- 1 ziquanw acm 38M Dec  4 02:41 pair19/L79D769P5/L79D769P5_remap.zip **
    -rw-r--r-- 1 ziquanw acm 44M Dec  5 00:39 pair19/L79D769P8/L79D769P8_remap.zip **
    -rw-r--r-- 1 ziquanw acm 40M Dec 23 11:51 pair20/L79D769P7/L79D769P7_remap.zip
    -rw-r--r-- 1 ziquanw acm 35M Dec 24 02:44 pair20/L79D769P9/L79D769P9_remap.zip
    -rw-r--r-- 1 ziquanw acm 40M Dec 24 19:55 pair21/L91D814P2/L91D814P2_remap.zip
    -rw-r--r-- 1 ziquanw acm 44M Dec 25 14:15 pair21/L91D814P6/L91D814P6_remap.zip
    -rw-r--r-- 1 ziquanw acm 41M Dec 26 07:48 pair22/L91D814P3/L91D814P3_remap.zip
    -rw-r--r-- 1 ziquanw acm 39M Dec 27 00:56 pair22/L91D814P4/L91D814P4_remap.zip
    -rw-r--r-- 1 ziquanw acm 41M Dec 27 21:05 pair3/L35D719P1/L35D719P1_remap.zip
    -rw-r--r-- 1 ziquanw acm 39M Dec 28 14:46 pair3/L35D719P4/L35D719P4_remap.zip
    -rw-r--r-- 1 ziquanw acm 39M Dec 11 04:08 pair5/L57D855P4/L57D855P4_remap.zip **
    -rw-r--r-- 1 ziquanw acm 41M Dec 15 05:42 pair5/L57D855P5/L57D855P5_remap.zip **
    -rw-r--r-- 1 ziquanw acm 59M Dec  5 20:57 pair6/L57D855P2/L57D855P2_remap.zip **
    -rw-r--r-- 1 ziquanw acm 31M Dec 10 00:51 pair6/L57D855P6/L57D855P6_remap.zip **
    -rw-r--r-- 1 ziquanw acm 42M Dec 29 09:46 pair8/L59D878P2/L59D878P2_remap.zip
    -rw-r--r-- 1 ziquanw acm 39M Dec 30 02:41 pair8/L59D878P5/L59D878P5_remap.zip
    -rw-r--r-- 1 ziquanw acm 37M Dec 30 19:14 pair9/L64D804P4/L64D804P4_remap.zip
    -rw-r--r-- 1 ziquanw acm 36M Dec 31 11:41 pair9/L64D804P6/L64D804P6_remap.zip

    cajal: pair17 18 5 6 4
    lichtman: all others

'''
if __name__=="__main__":
    img_tags = ['C1_', 'C2_','C3_']

#     pair_tag = 'pair6'
#     brain_tag = 'L57D855P6'
#     data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     # TODO: NIS used data in lichtman, but should to use cajal
#     pair_tag = 'pair6'
#     brain_tag = 'L57D855P2'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)


    pair_tag = 'pair5'
    brain_tag = 'L57D855P4'
    data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag+'restitchedLocalNorm', img_tags)
#     pair_tag = 'pair5'
#     brain_tag = 'L57D855P4'
#     data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair5'
#     brain_tag = 'L57D855P5'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)

# #########
#     pair_tag = 'pair19'
#     brain_tag = 'L79D769P8'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair19'
#     brain_tag = 'L79D769P5'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)


#     pair_tag = 'pair17'
#     brain_tag = 'L77D764P2'
#     data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)

#     pair_tag = 'pair17'
#     brain_tag = 'L77D764P9'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)


#     pair_tag = 'pair13'
#     brain_tag = 'L69D764P6'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair13'
#     brain_tag = 'L69D764P9'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)

#     pair_tag = 'pair15'
#     brain_tag = 'L73D766P4'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair15'
#     brain_tag = 'L73D766P9'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)


#     pair_tag = 'pair10'
#     brain_tag = 'L64D804P3'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair10'
#     brain_tag = 'L64D804P9'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)

#     pair_tag = 'pair11'
#     brain_tag = 'L66D764P3'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair11'
#     brain_tag = 'L66D764P8'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)

#     pair_tag = 'pair12'
#     brain_tag = 'L66D764P5'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair12'
#     brain_tag = 'L66D764P6'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
# ##########
#     pair_tag = 'pair14'
#     brain_tag = 'L73D766P5'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair14'
#     brain_tag = 'L73D766P7'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)


    # pair_tag = 'pair16'
    # brain_tag = 'L74D769P4'
    # data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag+'restitchedLocalNorm', img_tags)
    pair_tag = 'pair16'
    brain_tag = 'L74D769P8'
    data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag+'restitchedLocalNorm', img_tags)
#     pair_tag = 'pair16'
#     brain_tag = 'L74D769P4'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
    # pair_tag = 'pair16'
    # brain_tag = 'L74D769P4'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag+'LocalNorm', img_tags)
    # pair_tag = 'pair16'
    # brain_tag = 'L74D769P8'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag, img_tags)

#     pair_tag = 'pair18'
#     brain_tag = 'L77D764P4'
#     data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)

#     pair_tag = 'pair20'
#     brain_tag = 'L79D769P7'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair20'
#     brain_tag = 'L79D769P9'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)

#     pair_tag = 'pair21'
#     brain_tag = 'L91D814P2'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair21'
#     brain_tag = 'L91D814P6'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)

#     pair_tag = 'pair22'
#     brain_tag = 'L91D814P3'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair22'
#     brain_tag = 'L91D814P4'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)

#     pair_tag = 'pair3'
#     brain_tag = 'L35D719P1'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair3'
#     brain_tag = 'L35D719P4'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)


#     pair_tag = 'pair8'
#     brain_tag = 'L59D878P2'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair8'
#     brain_tag = 'L59D878P5'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)


#     pair_tag = 'pair9'
#     brain_tag = 'L64D804P4'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair9'
#     brain_tag = 'L64D804P6'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)

#     pair_tag = 'pair4'
#     brain_tag = 'L35D719P3'
#     data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
#     pair_tag = 'pair4'
#     brain_tag = 'L35D719P5'
#     data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)

#     pair_tag = 'pair6'
#     brain_tag = 'L57D855P1'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)

#     pair_tag = 'pair18'
#     brain_tag = 'L77D764P8'
#     data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag, img_tags)
################
    # brain_tag = 'L106P3'
    # pair_tag = 'female'
    # data_root = f'/lichtman/Ian/Lightsheet/P14/stitched/female/{brain_tag}/stitched/'
    # main(data_root, pair_tag, brain_tag+'LocalNorm', img_tags, 'P14')

    # brain_tag = 'L106P5'
    # pair_tag = 'female'
    # data_root = f'/lichtman/Ian/Lightsheet/P14/stitched/female/{brain_tag}/stitched/'
    # main(data_root, pair_tag, brain_tag, img_tags, 'P14')

    # brain_tag = 'L102P1'
    # pair_tag = 'female'
    # data_root = f'/lichtman/Ian/Lightsheet/P14/stitched/female/{brain_tag}/stitched/'
    # main(data_root, pair_tag, brain_tag, img_tags, 'P14')

    # brain_tag = 'L86P3'
    # pair_tag = 'female'
    # data_root = f'/lichtman/Ian/Lightsheet/P14/stitched/female/{brain_tag}/stitched/'
    # main(data_root, pair_tag, brain_tag, img_tags, 'P14')

    # brain_tag = 'L86P4'
    # pair_tag = 'female'
    # data_root = f'/lichtman/Ian/Lightsheet/P14/stitched/female/{brain_tag}/stitched/'
    # main(data_root, pair_tag, brain_tag, img_tags, 'P14')

    # brain_tag = 'L88P3'
    # pair_tag = 'female'
    # data_root = f'/lichtman/Ian/Lightsheet/P14/stitched/female/{brain_tag}/stitched/'
    # main(data_root, pair_tag, brain_tag, img_tags, 'P14')

    # brain_tag = 'L88P4'
    # pair_tag = 'female'
    # data_root = f'/lichtman/Ian/Lightsheet/P14/stitched/female/{brain_tag}/stitched/'
    # main(data_root, pair_tag, brain_tag, img_tags, 'P14')

    # brain_tag = 'L92P4'
    # pair_tag = 'female'
    # data_root = f'/lichtman/Ian/Lightsheet/P14/stitched/female/{brain_tag}/stitched/'
    # main(data_root, pair_tag, brain_tag, img_tags, 'P14')

    # brain_tag = 'L94P3'
    # pair_tag = 'female'
    # data_root = f'/lichtman/Ian/Lightsheet/P14/stitched/female/{brain_tag}/stitched/'
    # main(data_root, pair_tag, brain_tag, img_tags, 'P14')
