import torch, os, re
# from math import e
from datetime import datetime
import matplotlib.pyplot as plt
from multiprocessing import get_context
import nibabel as nib
import numpy
from run_whole_brain.utils import imread
# from brain_render import init_nib_header
from tqdm import trange, tqdm
# from tqdm.contrib.concurrent import process_map
# import pandas as pd
# import seaborn as sns
# from PIL import Image
from fast_pytorch_kmeans import KMeans

torch.manual_seed(142857)
def main(pair_tag, brain_tag, root='lichtman', P_tag='P4'):
    print(datetime.now(), pair_tag, brain_tag)
    stat_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/statistics/{P_tag}'
    save_fn = f'{stat_r}/{pair_tag}/{brain_tag}_NIS_colocContrastCalib_label.zip'
    img_r = f'/{root}/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'

    tgt_roi = 16001
    downsample_res = [25, 25, 25]
    img_res = [4, 0.75, 0.75]
    seg_res = [2.5, 0.75, 0.75]
    dratio = [s/d for s, d in zip(seg_res, downsample_res)]

    mask_fn = '/%s/Felix/Lightsheet/P4/%s/output_%s/registered/%s_MASK_topro_25_all.nii' % (root, pair_tag, brain_tag, brain_tag)

    orgcenter = torch.load(f'{stat_r}/{pair_tag}/{brain_tag}_nis_center.zip')
    mask = load_atlas(mask_fn)
    center = orgcenter.clone()
    center[:, 0] = (center[:, 0] * dratio[0]).clip(max=mask.shape[0]-1)
    center[:, 1] = (center[:, 1] * dratio[1]).clip(max=mask.shape[1]-1)
    center[:, 2] = (center[:, 2] * dratio[2]).clip(max=mask.shape[2]-1)
    center = center.long().T
    is_tgt_roi = mask[center[0],center[1],center[2]] == tgt_roi    
    assert is_tgt_roi.any(), f"is_tgt_roi all False, mask.shape=={mask.shape}"
    img_tags = ['C1_', 'C2_','C3_']
    if os.path.exists(f"{img_r}/{brain_tag}_0001_C2_brn2_stitched.tif"):
        img_names = ['topro', 'brn2','ctip2']
    elif os.path.exists(f"{img_r}/{brain_tag}_0001_C2_Brn2_stitched.tif"):
        img_names = ['topro', 'Brn2','Ctip2']
    else:
        print("No valid LS image, check LS image path")
        exit()
    instance_statistic = []
    for tagi, img_tag in enumerate(img_tags):
        instance_statistic.append(torch.load(f"{stat_r}/{pair_tag}/{brain_tag}_nis_{img_tag}intensity.zip"))

    data = torch.stack(instance_statistic, -1).float()
    print(datetime.now(), f"Loaded intensity, {data.shape}")

    ## Contrast Calibration ######################################################
    tgt_z = 565
    zstack = orgcenter[:, 0].round()
    zstack_uni = zstack.unique()
    imgzs, imgi = ((zstack_uni/(img_res[0]/seg_res[0])).long()+1).unique(return_inverse=True)
    imgzs, imgi = imgzs.tolist(), imgi.tolist()
    img_path_list = [[f"{img_r}/{brain_tag}_{imgz:04d}_{img_tags[i]}{img_names[i]}_stitched.tif" for imgz in imgzs] for i in range(len(img_tags))]
    imgmins = [0,0]
    bg_avg = [0 for _ in range(len(img_tags))]
    nis_avg = torch.zeros(len(data), len(img_tags))
    img_3d_downsampled = [0 for _ in range(len(img_tags))]
    img_sum = [0 for _ in range(len(img_tags))]
    for i in range(len(img_tags)):
        hist_fn = f'{stat_r}/{pair_tag}/{brain_tag}_{img_tags[i]}{img_names[i]}_hist.zip'
        img_stack = [None if imgi != tgt_z else load_img(img_path_list[i][imgi]) for imgi in range(len(img_path_list[i]))]
        H, W = img_stack[tgt_z].shape[:2]
        D = int(len(img_path_list[i]) * (img_res[0]/seg_res[0]))
        if os.path.exists(hist_fn):
            hist_zip = torch.load(hist_fn)
            hist = hist_zip['hist']
            bin_ranges = hist_zip['bin_ranges']
        else:
            img_3d = torch.zeros(len(img_path_list[i]), img_stack[tgt_z].shape[0], img_stack[tgt_z].shape[1])
            imgi_3d = 0
            with get_context("spawn").Pool(30) as p:
                for img in tqdm(p.imap(load_img, img_path_list[i]), total=len(img_path_list[i]), desc=f"{datetime.now()} Load raw {img_tags[i]}image"):
                    img_3d[imgi_3d] = img
                    imgi_3d += 1
            print(datetime.now(), f"Run {img_tags[i]}histogram")
            hist, bin_ranges = torch.histogram(img_3d.reshape(-1), bins=int((img_3d.max()-img_3d.min())/20))
            # print(datetime.now(), f"Done {img_tags[i]}histogram, save it")
            # torch.save({'hist': hist, 'bin_ranges': bin_ranges}, hist_fn)
            # del img_3d
            
        if i == 0:
        # if True:
            imgmin = bin_ranges.min()
            imgmax = bin_ranges.max()
        else:
            loghist = torch.log(hist)
            loghist[loghist==-torch.inf] = 0
            dhist = loghist[1:]-loghist[:-1]
            if i == 2:
                sm_num = 5
                while sm_num>0:
                    dhist = torch.nn.functional.avg_pool1d(dhist[None,:], kernel_size=7, stride=1).squeeze()
                    sm_num -= 1        
            raise_range, down_range, plain_range = find_cutoff(dhist)
            poolpad_size = int((len(hist) - len(dhist))/2)
            longest_down = 0
            for s, e in down_range:
                if e-s >= longest_down:
                    imgmin = bin_ranges[s+poolpad_size]
                    imgmax = bin_ranges[e+poolpad_size]
                    longest_down = e-s
            
            imgmins[i-1] = imgmin
            if i == 2: imgmin = imgmins[0]
        print(datetime.now(), f"{img_tags[i]}Intensity range {imgmin} ~ {imgmax}")
        all_intensity= ((bin_ranges[1:]+bin_ranges[:-1])/2).clip(min=imgmin, max=imgmax) # median of each bin range, then clip 
        all_intensity = (all_intensity - imgmin) / (imgmax-imgmin) # and normalize to 0~1
        img_sum = (hist.type(torch.float64)*all_intensity.type(torch.float64)).sum() # sum all 3D image
        # img_3d = img_3d[None, None, ...].clip(min=imgmin, max=imgmax)
        # img_3d = (img_3d - imgmin) / (imgmax-imgmin)
        # img_3d = torch.nn.functional.avg_pool3d(img_3d, kernel_size=100, stride=100).squeeze()
        # img_sum[i] = [img_3d.clip(min=imgmin, max=imgmax)*(100**3), [coor*100 for coor in torch.meshgrid([torch.arange(s) for s in img_3d.shape])]]
        
        nis_onfg = data[..., i] > 0
        nis_vol = nis_onfg.sum(1)
        nis_onbg = nis_vol==0
        data[..., i] = data[..., i].clip(min=imgmin, max=imgmax) * nis_onfg
        nis_sum = ((data[..., i][nis_onfg].type(torch.float64)-imgmin)/(imgmax-imgmin)).sum() # sum all 3D NIS
        bg_avg[i] = (img_sum - nis_sum) / (H*W*D - nis_onfg.sum()) # background avg intensity
        nis_avg[:, i] = (data[..., i].sum(1)-(imgmin*nis_vol))/(nis_vol*(imgmax-imgmin)) # sum(normed_intensity) / nis_vol
        nis_avg[nis_onbg, i] = 0
        print(bg_avg[i], nis_avg[:, i].mean(), nis_avg[:, i].max(), nis_avg[:, i].min())
        print(f"{datetime.now()} norm NIS {img_tags[i]}intensity then get avg")
        #### Save one slice to visualize 
        # for zi in range(len(zstack_uni)):
        #     if imgi[zi] == tgt_z:
        #         img = img_stack[imgi[zi]]
        #         img.clip_(min=imgmin, max=imgmax)
        #         saveimg = (img-img.min())/(img.max()-img.min())
        #         saveimg = (saveimg*10000).numpy().astype(numpy.uint16)
        #         Image.fromarray(saveimg).save(f'downloads/cell_type_channel_LS/{brain_tag}_{(tgt_z+1):04d}_{img_names[i]}_contrast_calibrated.tif')
    ## Done: Contrast Calibration ######################################################
    device = 'cuda:1'
    data = nis_avg
    ## Save calibrated nis avg intensity
    # for tagi, img_tag in enumerate(img_tags):
    #     torch.save(data[..., tagi], f"{stat_r}/{pair_tag}/{brain_tag}_nis_{img_tag}AvgContrast_calibrated.zip")
    #################################
    # pn_mask, np_mask, pp_mask, nn_mask = independent_kmeans_label(data, device, bg_avg)
    # pn_mask, np_mask, pp_mask, nn_mask = local_independent_coloc_label(data, device, bg_avg, img_sum, orgcenter)
    pn_mask, np_mask, pp_mask, nn_mask = independent_coloc_label(data, device, bg_avg)
    # pn_mask, np_mask, pp_mask, nn_mask = kmeans_label(data, device)

    # print(f"{data[pp_mask].shape[0]} / {data.shape[0]} ({(data[pp_mask].shape[0]/data.shape[0])*100:.2f}%) NIS is b+c+, {data[nn_mask].shape[0]} / {data.shape[0]} ({(data[nn_mask].shape[0]/data.shape[0])*100:.2f}%) is b-c- in Whole Brain")
    print(datetime.now(), "Saving coloc labels")
    torch.save({
        'pn_mask': pn_mask,
        'np_mask': np_mask,
        'pp_mask': pp_mask,
        'nn_mask': nn_mask
    }, save_fn)

    coloc_name = {
        'pn_mask': 'b+c-',
        'np_mask': 'b-c+',
        'pp_mask': 'b+c+',
        'nn_mask': 'b-c-'
    }
    coloc_label = {
        'pn_mask': pn_mask,
        'np_mask': np_mask,
        'pp_mask': pp_mask,
        'nn_mask': nn_mask
    }
    for k in coloc_label:
        is_k = coloc_label[k][is_tgt_roi]
        tgt_roi_count = len(is_k)
        k_count = len(torch.where(is_k)[0])
        ratio = 100*(k_count/tgt_roi_count) if tgt_roi_count!=0 else -1
        print(f"{k_count} / {tgt_roi_count} ({ratio:.2f}%) is {coloc_name[k]} in ROI id=={tgt_roi}")

    labels = torch.zeros_like(pn_mask).long()
    labels[pn_mask] = 1
    labels[np_mask] = 2
    labels[pp_mask] = 3
    labels[nn_mask] = 4
    vis_num = 10000

    print(datetime.now(), f"Random sampling {vis_num} NIS coloc labels for visualization")


    pn_mask = torch.where(pn_mask)[0][torch.randint(0, data[pn_mask].shape[0]-1, (int(vis_num/4),))]
    np_mask = torch.where(np_mask)[0][torch.randint(0, data[np_mask].shape[0]-1, (int(vis_num/4),))]
    pp_mask = torch.where(pp_mask)[0][torch.randint(0, data[pp_mask].shape[0]-1, (int(vis_num/4),))]
    nn_mask = torch.where(nn_mask)[0][torch.randint(0, data[nn_mask].shape[0]-1, (int(vis_num/4),))]

    pn = data[pn_mask].cpu()
    np = data[np_mask].cpu()
    pp = data[pp_mask].cpu()
    nn = data[nn_mask].cpu()
    print(datetime.now(), f"Random sampled {pn.shape[0]+np.shape[0]+pp.shape[0]+nn.shape[0]} for visualization")

    plt.scatter(pn[:, 1], pn[:, 2], color='red', label='b+c-', s=1)
    plt.scatter(np[:, 1], np[:, 2], color='blue', label='b-c+', s=1)
    plt.scatter(pp[:, 1], pp[:, 2], color='green', label='b+c+', s=1)
    plt.scatter(nn[:, 1], nn[:, 2], color='purple', label='b-c-', s=1)
    plt.legend()
    plt.xlabel('B intensity contrast')
    plt.ylabel('C intensity contrast')
    plt.savefig(save_fn.replace('.zip',f'{vis_num}scatter.png'))
    plt.close()
    
def kmeans_label(data, device):
    data = data.to(device)
    clustern = 3
    kmeans = KMeans(n_clusters=clustern, mode='cosine', verbose=1)
    # labels = kmeans.fit_predict(torch.cat([data[:, 1:2]-data[:, 2:3], data[:, :1]-data[:, 1:]], dim=1).abs()) #
    labels = kmeans.fit_predict(data) #
    print(labels.bincount())
    masks = []
    c1c2dis = []
    for i in range(clustern):
        cluster = data[labels == i]
        masks.append(labels == i)
        c1c2dis.append(cluster[:, 1].mean() - cluster[:, 2].mean())
    ind = torch.stack(c1c2dis).abs().argsort(descending=True)
    if c1c2dis[ind[0]] < 0:
        np_mask = masks[ind[0]]
        pn_mask = masks[ind[1]]
    else:
        pn_mask = masks[ind[0]]
        np_mask = masks[ind[1]]
    ppornn = data[masks[ind[2]]]
    kmeans = KMeans(n_clusters=2, mode='cosine', verbose=0)
    newdata = ppornn[:, 0].unsqueeze(1) - ppornn[:, 1:]
    labels = kmeans.fit_predict(newdata)
    # cluster0 = (ppornn[labels == 0, 1:] >= ppornn[labels == 1, 1:]).sum()
    # cluster1 = (ppornn[labels == 0, 1:] < ppornn[labels == 1, 1:]).sum()
    # cluster0 = ppornn[labels == 0, 1:].mean(0).sum()
    # cluster1 = ppornn[labels == 1, 1:].mean(0).sum()
    cluster0 = ppornn[labels==0].shape[0]
    cluster1 = ppornn[labels==1].shape[0]
    if cluster0 >= cluster1:
        nn_mask = masks[ind[2]].clone()
        nn_mask[torch.where(masks[ind[2]])[0][labels!=0]] = False
        pp_mask = masks[ind[2]].clone()
        pp_mask[torch.where(masks[ind[2]])[0][labels!=1]] = False
    else:
        pp_mask = masks[ind[2]].clone()
        pp_mask[torch.where(masks[ind[2]])[0][labels!=0]] = False
        nn_mask = masks[ind[2]].clone()
        nn_mask[torch.where(masks[ind[2]])[0][labels!=1]] = False
        
    return pn_mask.cpu(), np_mask.cpu(), pp_mask.cpu(), nn_mask.cpu()
        

def local_independent_coloc_label(data, device, bg_avg, img_sum, center):
    masks = []
    for i in range(1, len(img_sum)):
        masks.append(torch.zeros_like(center[:, 0]))
        all_intensity, coord = img_sum[i]
        all_intensity = all_intensity.reshape(-1)
        coord[0] = coord[0].reshape(-1)
        coord[1] = coord[1].reshape(-1)
        coord[2] = coord[2].reshape(-1)
        for j in range(len(coord[0])):
            f1 = center[:, 0]>=coord[0][i]
            f2 = center[:, 0]<=coord[0][i]+100
            f3 = center[:, 1]>=coord[1][i]
            f4 = center[:, 1]<=coord[1][i]+100
            f5 = center[:, 2]>=coord[2][i]
            f6 = center[:, 2]<=coord[2][i]+100
            local_mask = f1 & f2 & f3 & f4 & f5 & f6
            nis_onfg = data[local_mask, :, i] > 0
            nis_vol = nis_onfg.sum(1)
            nis_onbg = nis_vol==0
            nis_avg = data[local_mask, :, i].sum(1) / nis_vol
            nis_avg[nis_onbg] = 0
            bg_avg = (all_intensity[j] - data[local_mask, :, i].sum()) / ((100**3)-nis_vol.sum())
            masks[i][local_mask] = nis_avg > bg_avg
    return None
    # dis1 = (data[:, 1] - bg_avg[1]).abs().std()
    # dis2 = (data[:, 2] - bg_avg[2]).abs().std()
    # data = data.to(device)
    # p__mask = data[:, 1] > (bg_avg[1] + dis1/3)
    # n__mask = data[:, 1] <= (bg_avg[1] + dis1/3)
    # _p_mask = data[:, 2] > (bg_avg[2] + dis2/1.5)
    # _n_mask = data[:, 2] <= (bg_avg[2] + dis2/1.5)
    # pp_mask = torch.logical_and(p__mask, _p_mask)
    # pn_mask = torch.logical_and(p__mask, _n_mask)
    # np_mask = torch.logical_and(n__mask, _p_mask)
    # nn_mask = torch.logical_and(n__mask, _n_mask)
    # return pn_mask.cpu(), np_mask.cpu(), pp_mask.cpu(), nn_mask.cpu()


def independent_coloc_label(data, device, bg_avg):
    dis1 = (data[:, 1] - bg_avg[1]).abs().std()
    dis2 = (data[:, 2] - bg_avg[2]).abs().std()
    data = data.to(device)
    p__mask = data[:, 1] > (bg_avg[1] + dis1/3)
    n__mask = data[:, 1] <= (bg_avg[1] + dis1/3)
    _p_mask = data[:, 2] > (bg_avg[2] + dis2/1.5)
    _n_mask = data[:, 2] <= (bg_avg[2] + dis2/1.5)
    pp_mask = torch.logical_and(p__mask, _p_mask)
    pn_mask = torch.logical_and(p__mask, _n_mask)
    np_mask = torch.logical_and(n__mask, _p_mask)
    nn_mask = torch.logical_and(n__mask, _n_mask)
    return pn_mask.cpu(), np_mask.cpu(), pp_mask.cpu(), nn_mask.cpu()

def independent_kmeans_label(data, device, bg_avg):
    data = data.to(device)
    clustern = 2
    kmeans = KMeans(n_clusters=clustern, mode='cosine', verbose=1)
    dis1 = data[:, 1] - bg_avg[1]
    dis2 = data[:, 1] - bg_avg[1]
    label1 = kmeans.fit_predict(dis1[:, None])
    if dis1[label1==0].mean() > dis1[label1==1].mean():
        p__mask = label1==0
        n__mask = label1==1
    else:
        p__mask = label1==1
        n__mask = label1==0
    kmeans = KMeans(n_clusters=clustern, mode='cosine', verbose=1)
    label2 = kmeans.fit_predict(dis2[:, None])
    if dis2[label2==0].mean() > dis2[label2==1].mean():
        _p_mask = label2==0
        _n_mask = label2==1
    else:
        _p_mask = label2==1
        _n_mask = label2==0
    pp_mask = torch.logical_and(p__mask, _p_mask)
    pn_mask = torch.logical_and(p__mask, _n_mask)
    np_mask = torch.logical_and(n__mask, _p_mask)
    nn_mask = torch.logical_and(n__mask, _n_mask)
    return pn_mask.cpu(), np_mask.cpu(), pp_mask.cpu(), nn_mask.cpu()

def heuristic_label(data, device):
    data = data.to(device)
    pos = data[:, 1] > data[:, 2]
    neg = data[:, 1] < data[:, 2]
    abs_dis = (data[:,1]-data[:,2]).abs()
    separate_mask = torch.zeros_like(abs_dis).bool()
    separate_mask[neg] = abs_dis[neg] >= abs_dis.mean()
    separate_mask[pos] = abs_dis[pos] >= (abs_dis.mean())/3
    pn_mask = separate_mask&pos
    np_mask = separate_mask&neg
    print(f"{data[separate_mask].shape[0]} / {data.shape[0]} ({(data[separate_mask].shape[0]/data.shape[0])*100:.2f}%) NIS are b+c- or b-c+ in Whole Brain")
    print(f"{data[pn_mask].shape[0]} / {data.shape[0]} ({(data[pn_mask].shape[0]/data.shape[0])*100:.2f}%) NIS is b+c-, {data[np_mask].shape[0]} / {data.shape[0]} ({(data[np_mask].shape[0]/data.shape[0])*100:.2f}%) is b-c+ in Whole Brain")
    max_data = data[:, 1:].max(1)[0]
    f1 = (data[:,0]-max_data).abs() >= (data[:,0]-max_data).abs().mean()
    f2  = max_data > data[:,0]
    f3 = torch.logical_not(separate_mask)
    pp_mask = f1 & f2 & f3
    nn_mask = torch.logical_not(pp_mask) & torch.logical_not(separate_mask)
    return pn_mask.cpu(), np_mask.cpu(), pp_mask.cpu(), nn_mask.cpu()

def load_img(path):
    return torch.from_numpy(imread(path).astype(numpy.int32)).float()

def load_atlas(mask_fn):
    orig_roi_mask = torch.from_numpy(numpy.transpose(nib.load(mask_fn).get_fdata(), (2, 0, 1))[:, :, ::-1].copy())
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

def find_cutoff(data):
    r = data > 0
    d = data < 0
    p = data == 0
    r_range = []
    d_range = []
    p_range = []
    cur_s = None
    turned = True
    for i in range(len(d)):
        if cur_s == 'r' : 
            if not r[i]:
                r_range[-1][1] = i
                turned = True
        elif cur_s == 'd':
            if not d[i]:
                d_range[-1][1] = i
                turned = True
        elif cur_s == 'p':
            if not p[i]:
                p_range[-1][1] = i
                turned = True
        if turned:
            if d[i]:
                cur_s = 'd' 
                d_range.append([i, None])
            elif r[i]:
                cur_s = 'r' 
                r_range.append([i, None])
            elif p[i]:
                cur_s = 'p' 
                p_range.append([i, None])
            turned = False
    
    if cur_s == 'r' : 
        r_range[-1][1] = i
    elif cur_s == 'd':
        d_range[-1][1] = i
    elif cur_s == 'p':
        p_range[-1][1] = i
    return r_range, d_range, p_range

if __name__ == '__main__':
    # pair_tag = 'pair15'
    # brain_tag = 'L73D766P9'
    # main(pair_tag, brain_tag)
    # pair_tag = 'pair15'
    # brain_tag = 'L73D766P4'
    # main(pair_tag, brain_tag)

    # pair_tag = 'pair6'
    # brain_tag = 'L57D855P6'
    # main(pair_tag, brain_tag, 'cajal')
    # pair_tag = 'pair6'
    # brain_tag = 'L57D855P2'
    # main(pair_tag, brain_tag, 'cajal')


    # pair_tag = 'pair5'
    # brain_tag = 'L57D855P4'
    # main(pair_tag, brain_tag, 'cajal')
    # pair_tag = 'pair5'
    # brain_tag = 'L57D855P5'
    # main(pair_tag, brain_tag, 'cajal')


    # pair_tag = 'pair19'
    # brain_tag = 'L79D769P8'
    # main(pair_tag, brain_tag)
    # pair_tag = 'pair19'
    # brain_tag = 'L79D769P5'
    # main(pair_tag, brain_tag)


    # pair_tag = 'pair17'
    # brain_tag = 'L77D764P2'
    # main(pair_tag, brain_tag, 'cajal')
    
    # pair_tag = 'pair17'
    # brain_tag = 'L77D764P9'
    # main(pair_tag, brain_tag, 'cajal')


    
    # pair_tag = 'pair13'
    # brain_tag = 'L69D764P6'
    # main(pair_tag, brain_tag)
    
    # pair_tag = 'pair13'
    # brain_tag = 'L69D764P9'
    # main(pair_tag, brain_tag)
    
    pair_tag = 'pair10'
    brain_tag = 'L64D804P3'
    main(pair_tag, brain_tag)
    # # 
    pair_tag = 'pair10'
    brain_tag = 'L64D804P9'
    main(pair_tag, brain_tag)
    
    # # pair_tag = 'pair11'
    # # brain_tag = 'L66D764P3'
    # # main(pair_tag, brain_tag)
    # # pair_tag = 'pair11'
    # # brain_tag = 'L66D764P8'
    # # main(pair_tag, brain_tag)

    # # pair_tag = 'pair12'
    # # brain_tag = 'L66D764P5'
    # # main(pair_tag, brain_tag)
    # # pair_tag = 'pair12'
    # # brain_tag = 'L66D764P6'
    # # main(pair_tag, brain_tag)

    # pair_tag = 'pair14'
    # brain_tag = 'L73D766P5'
    # main(pair_tag, brain_tag)
    # pair_tag = 'pair14'
    # brain_tag = 'L73D766P7'
    # main(pair_tag, brain_tag)

    # pair_tag = 'pair16'
    # brain_tag = 'L74D769P4'
    # main(pair_tag, brain_tag)
    # pair_tag = 'pair16'
    # brain_tag = 'L74D769P8'
    # main(pair_tag, brain_tag)

    # pair_tag = 'pair18'
    # brain_tag = 'L77D764P4'
    # main(pair_tag, brain_tag, 'cajal')

    # pair_tag = 'pair20'
    # brain_tag = 'L79D769P7'
    # main(pair_tag, brain_tag)
    # pair_tag = 'pair20'
    # brain_tag = 'L79D769P9'
    # main(pair_tag, brain_tag)

    # pair_tag = 'pair21'
    # brain_tag = 'L91D814P2'
    # main(pair_tag, brain_tag)
    # pair_tag = 'pair21'
    # brain_tag = 'L91D814P6'
    # main(pair_tag, brain_tag)

    # pair_tag = 'pair22'
    # brain_tag = 'L91D814P3'
    # main(pair_tag, brain_tag)
    # pair_tag = 'pair22'
    # brain_tag = 'L91D814P4'
    # main(pair_tag, brain_tag)

    # pair_tag = 'pair3'
    # brain_tag = 'L35D719P1'
    # main(pair_tag, brain_tag)
    # pair_tag = 'pair3'
    # brain_tag = 'L35D719P4'
    # main(pair_tag, brain_tag)


    # pair_tag = 'pair8'
    # brain_tag = 'L59D878P2'
    # main(pair_tag, brain_tag)
    # pair_tag = 'pair8'
    # brain_tag = 'L59D878P5'
    # main(pair_tag, brain_tag)


    # pair_tag = 'pair9'
    # brain_tag = 'L64D804P4'
    # main(pair_tag, brain_tag)
    # pair_tag = 'pair9'
    # brain_tag = 'L64D804P6'
    # main(pair_tag, brain_tag)