import torch, os, re
from math import e
from datetime import datetime
import matplotlib.pyplot as plt
from multiprocessing import Pool, set_start_method, get_context
import nibabel as nib
import numpy
from run_whole_brain.utils import imread
from brain_render import init_nib_header
from tqdm import trange, tqdm
from tqdm.contrib.concurrent import process_map
import pandas as pd
import seaborn as sns
from PIL import Image
# set_start_method("spawn")


def main(pair_tag, brain_tag):
    stat_r = '/cajal/ACMUSERS/ziquanw/Lightsheet/statistics'
    # fn = f'{stat_r}/{pair_tag}/{brain_tag}_nis_normed_mean_C1_C2_C3_intensity.zip'
    save_fn = f'{stat_r}/{pair_tag}/{brain_tag}_NIS_colocContrastCalib_label.zip'
    img_r = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'

    # tgt = 16001
    # downsample_res = [25, 25, 25]
    img_res = [4, 0.75, 0.75]
    seg_res = [2.5, 0.75, 0.75]
    # dratio = [s/d for s, d in zip(seg_res, downsample_res)]

    # mask_fn = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/registered/%s_MASK_topro_25_all.nii' % (pair_tag, brain_tag, brain_tag)

    orgcenter = torch.load(f'{stat_r}/{pair_tag}/{brain_tag}_nis_center.zip')
    # mask = load_atlas(mask_fn)
    # center = orgcenter.clone()
    # print(mask.unique())
    # exit()
    # center[:, 0] = center[:, 0] * dratio[0]
    # center[:, 1] = center[:, 1] * dratio[1]
    # center[:, 2] = center[:, 2] * dratio[2]
    # center = center.long().T

    # is_tgt = mask[center[0],center[1],center[2]] == tgt
    # print(datetime.now(), f"Loaded mask, tgt num: {len(torch.where(is_tgt)[0])}")

    # data = torch.load(fn)

    img_tags = ['C1_', 'C2_','C3_']
    img_names = ['topro', 'brn2','ctip2']
    instance_statistic = []
    for tagi, img_tag in enumerate(img_tags):
        instance_statistic.append(torch.load(f"{stat_r}/{pair_tag}/{brain_tag}_nis_{img_tag}intensity.zip"))

    data = torch.stack(instance_statistic, -1).float()
    # data = data[is_tgt[:len(data)]]
    print(datetime.now(), f"Loaded intensity, {data.shape}")

    ## Contrast Calibration ######################################################
    device = 'cuda:1'
    # dist_org = {"Z": [], "Channel": [], "Intensity": []}
    # dist_norm = {"Z": [], "Channel": [], "Intensity": []}
    tgt_z = 565
    zstack = orgcenter[:, 0].round()
    zstack_uni = zstack.unique()
    imgzs, imgi = ((zstack_uni/(img_res[0]/seg_res[0])).long()+1).unique(return_inverse=True)
    imgzs, imgi = imgzs.tolist(), imgi.tolist()
    img_path_list = [[f"{img_r}/{brain_tag}_{imgz:04d}_{img_tags[i]}{img_names[i]}_stitched.tif" for imgz in imgzs] for i in range(len(img_tags))]
    # img_stacks = [[] for i in range(len(img_tags))]
    # save_imgs = [None for i in range(len(img_tags))]
    # img_range = [[] for i in range(len(img_tags))]
    for i in range(len(img_tags)):
        hist_fn = f'{stat_r}/{pair_tag}/{brain_tag}_{img_tags[i]}{img_names[i]}_hist.zip'
        if os.path.exists(hist_fn):
            hist_zip = torch.load(hist_fn)
            hist = hist_zip['hist']
            bin_ranges = hist_zip['bin_ranges']
            img_stack = [None if imgi != tgt_z else load_img(img_path_list[i][imgi]) for imgi in range(len(img_path_list[i]))]
        else:
            # img_stack = []
            # for path in tqdm(img_path_list[i], desc=f"{datetime.now()} Load raw {img_tags[i]}image"):
            #     img_stack.append(load_img(path[0]))
            # img_stack = process_map(load_img, img_path_list[i])
            with get_context("spawn").Pool(30) as p:
                img_stack = list(tqdm(p.imap(load_img, img_path_list[i]), total=len(img_path_list[i]), desc=f"{datetime.now()} Load raw {img_tags[i]}image"))
            print(datetime.now(), f"Stack 2D {img_tags[i]}LS images")
            img_stack = torch.stack(img_stack)
            print(datetime.now(), f"Run {img_tags[i]}histogram")
            hist, bin_ranges = torch.histogram(img_stack.reshape(-1), bins=int((img_stack.max()-img_stack.min())/20))
            print(datetime.now(), f"Done {img_tags[i]}histogram, save it")
            torch.save({'hist': hist, 'bin_ranges': bin_ranges}, hist_fn)
        loghist = torch.log(hist)
        loghist[loghist==-torch.inf] = 0
        dhist = loghist[1:]-loghist[:-1]
        # sm_num = 1
        # while sm_num>0:
        #     dhist = torch.nn.functional.avg_pool1d(dhist[None,:], kernel_size=5, stride=1).squeeze()
        #     sm_num -= 1        
        raise_range, down_range, plain_range = find_cutoff(dhist)
        poolpad_size = int((len(hist) - len(dhist))/2)
        longest_down = 3
        for s, e in down_range:
            if e-s >= longest_down:
                imgmin = bin_ranges[s+poolpad_size]
                imgmax = bin_ranges[e+poolpad_size]
                longest_down = e-s
        # img_range[i]
        # else:
        #     imgmin = img_stack.min()
        #     imgmin = img_stack.max()
        for zi in trange(len(zstack_uni), desc=f"{datetime.now()} norm NIS {img_tags[i]}intensity"):
            z = zstack_uni[zi]
            zloc = torch.where(zstack==z)[0]
            fg = torch.where(data[zloc, :, i] > 0)
            zloc = zloc[fg[0]]
            # dist_org['Z'].extend([imgi[zi] for _ in range(len(data[zloc, fg[1], i]))])
            # dist_org['Channel'].extend([img_tags[i] for _ in range(len(data[zloc, fg[1], i]))])
            # dist_org['Intensity'].extend(data[zloc, fg[1], i].tolist())
            
            ## TODO: get imgmin imgmax by histogram
            ## Default:
            # if img_tags[i] == 'C2_':
            #     imgmin, imgmax = 120, 600
            # elif img_tags[i] == 'C3_':
            #     imgmin, imgmax = 120, 24000
            ########################################
            # img = torch.from_numpy(img.astype(numpy.int32)).float()
            # hist, bin_ranges = torch.histogram(img.reshape(-1), bins=int((img.max()-img.min())/20))
            # loghist = torch.log(hist)
            # loghist[loghist==-torch.inf] = 0
            # dhist = loghist[1:]-loghist[:-1]
            # dhist = torch.nn.functional.avg_pool1d(dhist[None,:], kernel_size=5, stride=1).squeeze()
            # if len(dhist.shape) > 0:
            #     if dhist.shape[0] > 1:
            #         # sm_num = 10
            #         # while sm_num>0:
            #         #     dhist = torch.nn.functional.avg_pool1d(dhist[None,:], kernel_size=5, stride=1).squeeze()
            #         #     sm_num -= 1
            #         raise_range, down_range, plain_range = find_cutoff(dhist)
            #         poolpad_size = int((len(hist) - len(dhist))/2)
            #         longest_down = 3
            #         for s, e in down_range:
            #             if e-s >= longest_down:
            #                 imgmin = bin_ranges[s+poolpad_size]
            #                 imgmax = bin_ranges[e+poolpad_size]
            #                 longest_down = e-s
            ########################################
            x = data[zloc, fg[1], i]
            # if img_tags[i] != 'C1_':
            x.clip_(min=imgmin, max=imgmax)
            # else:
                # imgmin, imgmax = img.min(), img.max()
            if imgi[zi] == tgt_z:
                img = img_stack[imgi[zi]]
                img.clip_(min=imgmin, max=imgmax)
                saveimg = (img-img.min())/(img.max()-img.min())
                saveimg = (saveimg*10000).numpy().astype(numpy.uint16)
                Image.fromarray(saveimg).save(f'downloads/{brain_tag}_{(tgt_z+1):04d}_{img_names[i]}_contrast_calibrated.tif')
            # dist_norm['Z'].extend([imgi[zi] for _ in range(len(data[zloc, fg[1], i]))])
            # dist_norm['Channel'].extend([img_tags[i] for _ in range(len(data[zloc, fg[1], i]))])
            data[zloc, fg[1], i] = (x-imgmin)/(imgmax-imgmin)
            # dist_norm['Intensity'].extend(data[zloc, fg[1], i].tolist())

    # dist_org = pd.DataFrame(dist_org)
    # dist_norm = pd.DataFrame(dist_norm)

    # plt.figure(figsize=(250,10))
    # ax = sns.boxplot(data=dist_norm, x='Z', y='Intensity', hue='Channel')
    # ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    # plt.tight_layout()
    # plt.savefig('temp_norm.png')
    # plt.close()
            
    ## Contrast Calibration ######################################################

    # data = norm_tensor(data).mean(1)
    data = data.mean(1)
    for tagi, img_tag in enumerate(img_tags):
        torch.save(data[..., tagi], f"{stat_r}/{pair_tag}/{brain_tag}_nis_{img_tag}AvgContrast_calibrated.zip")
    data = data.to(device)
    pos = data[:, 1] > data[:, 2]
    neg = data[:, 1] < data[:, 2]

    separate_mask = (data[:,1]-data[:,2]).abs() >= (data[:,1]-data[:,2]).abs().mean()
    pn_mask = separate_mask&pos
    np_mask = separate_mask&neg

    print(datetime.now(), f"{data[separate_mask].shape[0]} / {data.shape[0]} ({(data[separate_mask].shape[0]/data.shape[0])*100:.2f}%) NIS are b+c- or b-c+")
    print(datetime.now(), f"{data[pn_mask].shape[0]} / {data.shape[0]} ({(data[pn_mask].shape[0]/data.shape[0])*100:.2f}%) NIS is b+c-, {data[np_mask].shape[0]} / {data.shape[0]} ({(data[np_mask].shape[0]/data.shape[0])*100:.2f}%) is b-c+")

    max_data = data[:, 1:].max(1)[0]
    f1 = (data[:,0]-max_data).abs() >= (data[:,0]-max_data).abs().mean()
    f2  = max_data > data[:,0]
    f3 = torch.logical_not(separate_mask)
    pp_mask = f1 & f2 & f3
    nn_mask = torch.logical_not(pp_mask) & torch.logical_not(separate_mask)

    print(datetime.now(), f"{data[pp_mask].shape[0]} / {data.shape[0]} ({(data[pp_mask].shape[0]/data.shape[0])*100:.2f}%) NIS is b+c+, {data[nn_mask].shape[0]} / {data.shape[0]} ({(data[nn_mask].shape[0]/data.shape[0])*100:.2f}%) is b-c-")
    print(datetime.now(), "Saving coloc labels")
    torch.save({
        'pn_mask': pn_mask,
        'np_mask': np_mask,
        'pp_mask': pp_mask,
        'nn_mask': nn_mask
    }, save_fn)
    labels = torch.zeros_like(pn_mask).long()
    labels[pn_mask] = 1
    labels[np_mask] = 2
    labels[pp_mask] = 3
    labels[nn_mask] = 4
    # vis_chunk('lichtman', orgcenter[is_tgt], labels, save_fn)
    # exit()
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
    plt.xlabel('B intensity')
    plt.ylabel('C intensity')
    plt.savefig(save_fn.replace('.zip',f'{vis_num}scatter.png'))
    # plt.savefig(f'{stat_r}/{pair_tag}/{brain_tag}_nis_coloc_label_{vis_num}scatter.png')

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

def vis_chunk(data_root, center, label, save_fn):
    dratio = [s/i for i, s in zip(img_res, seg_res)]
    new_header, affine_m = init_nib_header()
    center = center.T
    center[0] = center[0] * dratio[0]
    center[1] = center[1] * dratio[1]
    center[2] = center[2] * dratio[2]
    print(center.shape, center[0].min(), center[0].max(), center[1].min(), center[1].max(), center[2].min(), center[2].max())
    x, y, z = [3000, 3300], [3000, 3300], [350, 410]
    f1 = center[0] > z[0]
    f2 = center[1] > x[0]
    f3 = center[2] > y[0]
    f4 = center[0] < z[1]
    f5 = center[1] < x[1]
    f6 = center[2] < y[1]
    f = f1 & f2 & f3 & f4 & f5 & f6
    center = center[:, f].long()
    print(center.shape, label[f].unique())
    label_img = torch.zeros(z[1]-z[0], x[1]-x[0], y[1]-y[0]).long()
    label_img[center[0], center[1], center[2]] = label[f].cpu()
    img_root = f"/{data_root}/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched"
    img_paths, _ = listdir_sorted(img_root, 'C1')  
    with Pool(z[1]-z[0]) as p:
        img_stack = list(p.starmap(imread, [[p] for p in img_paths[z[0]:z[1]]]))
    img_stack = torch.from_numpy(numpy.stack(img_stack).astype(numpy.int32))[:, x[0]:x[1], y[0]:y[1]]
    nib.save(nib.Nifti1Image(label_img.numpy().astype(numpy.float64), affine_m, header=new_header), f'{save_fn[:-4]}_chunk_label.nii')
    nib.save(nib.Nifti1Image(img_stack.numpy().astype(numpy.float64), affine_m, header=new_header), f'{save_fn[:-4]}_chunk_img.nii')

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

def norm_tensor(x):
    fg = x>0
    fgx = x[fg]
    x[fg] = (fgx-fgx.min())/(fgx.max()-fgx.min())
    return x

if __name__ == '__main__':
    pair_tag = 'pair15'
    brain_tag = 'L73D766P9'
    main(pair_tag, brain_tag)