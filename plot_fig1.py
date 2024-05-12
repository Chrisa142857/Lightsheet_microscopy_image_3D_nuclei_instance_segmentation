import torch, os
import torch.nn.functional as F
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
import nibabel as nib
import h5py
import random
import SimpleITK as sitk
import vedo

np.random.seed(142857)
STAT_ROOT = '/cajal/ACMUSERS/ziquanw/Lightsheet/statistics/P4'
LOC = (np.random.randint(1,1000)/1200, 0.5, 0.5)
size = (4, 10000, 10000)


def rotate_batch_tensor(x, vectors):
    x = x.unsqueeze(1).float()
    # x [N x W x H x D] vectors [N x 3]
    a = vectors[:, 0]
    b = vectors[:, 1] # N 
    c = vectors[:, 2]
    ab = torch.sqrt(a**2 + b**2) # N
    ac = torch.sqrt(a**2 + c**2)
    bc = torch.sqrt(c**2 + b**2)
    cosa = b/ab # N
    sina = -a/ab
    cosb = c/bc
    sinb = -b/bc
    cosc = c/ac
    sinc = -a/ac
    rot_mat = torch.stack((torch.stack([cosa*cosb, cosa*sinb*sinc-sina*cosc, cosa*sinb*cosc + sina*sinc], dim=1),
                           torch.stack([sina*cosb, sina*sinb*sinc + cosa*cosc, sina*sinb*cosc - cosa*sinc], dim=1),
                           torch.stack([-sinb, cosb*sinc, cosb*cosc], dim=1)), dim=1) # N x 3 x 3
    zeros = torch.zeros(rot_mat.size(0), vectors.shape[-1]).unsqueeze(2) # N x 3 x 1
    aff_mat = torch.cat((rot_mat, zeros), 2) # N x 3 x 4
    grid = F.affine_grid(aff_mat, x.size(), align_corners=True)
    x = F.grid_sample(x, grid, align_corners=True)
    return x

def get_one_vol_frame(pair_tag, brain_tag, nis_id):
    # print(datetime.now(), f"Loading {pair_tag} {brain_tag}")
    # nis_max_num = 3000
    stat_root = f"{STAT_ROOT}/{pair_tag}"
    total_vol = torch.load(f"{stat_root}/{brain_tag}_nis_volume.zip", map_location='cpu')
    splits = total_vol.cumsum(0)
    coord_id = []
    pt_splits = total_vol[nis_id].cumsum(0)
    for i in nis_id:
        coord_id.append(np.arange(splits[i]-total_vol[i], splits[i]))
        # pt_splits.append()
    coord_id = np.concatenate(coord_id)
    print(datetime.now(), f"Start get coordinate {pair_tag} {brain_tag}")
    coordinate = h5py.File(f"{stat_root}/{brain_tag}_nis_coordinate.h5", 'r')['data']
    pts = torch.from_numpy(coordinate[coord_id].astype(np.int16))#.reshape(nis_max_num, vol, 3)
    pts = torch.nn.utils.rnn.pad_sequence(torch.tensor_split(pts, pt_splits)[:-1], batch_first=True, padding_value=-1) # N x vol x 3
    print(datetime.now(), "Done get coordinate, Start build frames")
    # ptmin = pts.min(1)[0] # N x 3
    pts_pad_mask = pts==-1
    # print(pts_pad_mask[...,0].sum(1).shape)
    ptmin = pts.sort(1)[0][torch.arange(len(pts)), pts_pad_mask[...,0].sum(1)]
    # print(ptmin[:10], ptmin.shape)
    pts = pts - ptmin.unsqueeze(1)
    pts[pts_pad_mask] = -1
    ptmax = pts.max(1)[0] # N x 3
    # print(ptmax)
    # ptmin = pts.min(1)[0] # N x 3
    # assert (ptmin==0).all()
    ptmid = ptmax//2 # N x 3
    frame_whd = ptmax.max(0)[0] # 3
    frame_mid = (frame_whd//2).unsqueeze(0) # 1 x 3
    mid_remain = frame_mid - ptmid # N x 3
    assert (mid_remain>=0).all()
    pts = pts + mid_remain.unsqueeze(1) # N x vol x 3
    pts[pts_pad_mask] = -1
    frame_whd = frame_whd + 1
    # print(frame_whd,pts.shape)
    # exit()
    frames = torch.zeros([len(pts)]+frame_whd.tolist(), dtype=bool)
    frame_id = torch.arange(len(pts)).unsqueeze(0).repeat(pts.shape[1],1).T#.reshape(-1)
    frames[frame_id, pts[..., 0].long(), pts[..., 1].long(), pts[..., 2].long()] = True
    frames[:, -1, -1, -1] = False
    # frames = frames.long()
    print(datetime.now(), "Done build frames, Start get principle axis")
    filter_label = sitk.LabelShapeStatisticsImageFilter()
    filter_label.SetComputeFeretDiameter(True)
    pa_list = [] # principle axis list
    all_pa = []
    pts = pts - frame_mid.unsqueeze(0) # N x vol x 3

    for i in trange(len(frames), desc='Get principle axises'):
        frame = frames[i].long()
        filter_label.Execute(sitk.GetImageFromArray(frame.numpy()))
        pa = torch.FloatTensor(filter_label.GetPrincipalAxes(1)) # 9
        distances_pa1 = pts[i].float() @ pa[:3] # pa1: N
        pa1_start = frame_mid.squeeze() + (distances_pa1.min() * pa[:3]) # 3
        pa1_end = frame_mid.squeeze() + (distances_pa1.max() * pa[:3]) # 3
        pa1_vec = pa1_end - pa1_start # 3
        pa_list.append(pa1_vec)
        distances_pa2 = pts[i].float() @ pa[3:6] # pa1: N
        pa2_start = frame_mid.squeeze() + (distances_pa2.min() * pa[3:6]) # 3
        pa2_end = frame_mid.squeeze() + (distances_pa2.max() * pa[3:6]) # 3
        distances_pa3 = pts[i].float() @ pa[6:] # pa1: N
        pa3_start = frame_mid.squeeze() + (distances_pa3.min() * pa[6:]) # 3
        pa3_end = frame_mid.squeeze() + (distances_pa3.max() * pa[6:]) # 3
        all_pa.append(torch.stack([
            pa1_end - pa1_start, 
            pa2_end - pa2_start, 
            pa3_end - pa3_start
        ]))
    pa_list = torch.stack(pa_list) # N x 3
    all_pa = torch.stack(all_pa) # N x 3 x 3
    print(datetime.now(), "Done get principle axis, Start rotate frames", frames.shape)
    frames = rotate_batch_tensor(frames, pa_list)
    avg_frame = frames.mean(0)[0]
    print(datetime.now(), "Done rotate frames, return avg", avg_frame.shape)
    return avg_frame, all_pa
    # frame_loc = torch.arange(frame_whd.cumprod(0)[-1]).view(frame_whd.tolist())
    # loc = frame_loc[pts[..., 0].long(), pts[..., 1].long(), pts[..., 2].long()].reshape(-1)
    # loc_count = loc.bincount() 
    # loc_count = loc_count[loc_count!=0]
    # frame = torch.zeros(frame_whd.cumprod(0)[-1], dtype=torch.float64)
    # frame[loc.unique()] = loc_count.double() #/ center.shape[0]
    # frame = frame.view(frame_whd.tolist())
    # # print(frame.max(), frame.min())
    # return frame

def get_one_stack_vol(d, pair_tag, brain_tag, locp=None, key='', size=None):
    print(datetime.now(), f"Loading {pair_tag} {brain_tag}")
    stat_root = f"{STAT_ROOT}/{pair_tag}"
    total_center = torch.load(f"{stat_root}/{brain_tag}_nis_center.zip", map_location='cpu')
    total_vol = torch.load(f"{stat_root}/{brain_tag}_nis_volume.zip", map_location='cpu')
    loc = [0,0,0]
    valid_center = total_center
    for i in range(len(locp)):
        minc = valid_center.min(0)[0][i]
        maxc = valid_center.max(0)[0][i]
        loc[i] = locp[i] * (maxc - minc) + minc
        valid_center = valid_center[valid_center[:, i]==minc]

    print(datetime.now(), f"Loaded {total_vol.shape} NIS")
    if loc is not None:
        f1 = total_center[:, 0] >= loc[0]-size[0]/2
        f2 = total_center[:, 1] >= loc[1]-size[1]/2
        f3 = total_center[:, 2] >= loc[2]-size[2]/2
        f4 = total_center[:, 0] <= loc[0]+size[0]/2
        f5 = total_center[:, 1] <= loc[1]+size[1]/2
        f6 = total_center[:, 2] <= loc[2]+size[2]/2
        stack_vol = total_vol[f1 & f2 & f3 & f4 & f5 & f6]
        stack_center = total_center[f1 & f2 & f3 & f4 & f5 & f6]
    else:
        stack_vol = total_vol
        stack_center = total_center
    stack_vol = stack_vol[stack_vol<=1000]
    d['vol'] += stack_vol.tolist()
    d['brain_tag'] += np.array(brain_tag+key).repeat(len(stack_vol)).tolist()
    stack_range = np.array([
        [loc[0]-size[0]/2],
        [loc[1]-size[1]/2],
        [loc[2]-size[2]/2],
        [loc[0]+size[0]/2],
        [loc[1]+size[1]/2],
        [loc[2]+size[2]/2]
    ])
    if d['stack_range'] is None:
        d['stack_range'] = stack_range.repeat(len(stack_vol),1).T
    else:
        d['stack_range'] = np.concatenate([d['stack_range'], stack_range.repeat(len(stack_vol),1).T])
    print(datetime.now(), f"Get {len(stack_vol)} NIS")
    if 'density' in d and stack_center.numel() > 0:
        down_size = 256
        stack_center = stack_center[:, 1:]//down_size
        stack_center = stack_center.long()
        dshape = (stack_center[:, 0].max() + 1, stack_center[:, 1].max() + 1)
        loc = torch.arange(dshape[0]*dshape[1]).view(dshape[0], dshape[1])
        loc = loc[stack_center[:, 0], stack_center[:, 1]]
        loc_count = loc.bincount() 
        loc_count = loc_count[loc_count!=0] 
        d['density'] += loc_count.tolist()
        d['brain_tag_density'] += np.array(brain_tag+key).repeat(len(loc_count)).tolist()
        if d['stack_range_density'] is None:
            d['stack_range_density'] = stack_range.repeat(len(loc_count),1).T
        else:
            d['stack_range_density'] = np.concatenate([d['stack_range_density'], stack_range.repeat(len(loc_count),1).T])
    return d

def get_pair_nis(pair_tag1, brain_tag1, pair_tag2, brain_tag2, data={'vol': [], 'brain_tag': []}):
    
    if brain_tag1 == brain_tag2:
        # LOC1 = (0.5, 0.3, 0.3)
        # LOC2 = (0.3, 0.5, 0.3)
        # LOC1 = list(np.random.randint(1,1000,(3,))/1500)
        # LOC2 = list(np.random.randint(1,1000,(3,))/1500)
        # size = (64, 128, 128)
        # LOC1 = (np.random.randint(1,1000)/1200, 0.5, 0.5)
        # LOC2 = (np.random.randint(1,1000)/1200, 0.5, 0.5)
        # LOC1 = LOC
        LOC2 = (LOC[0]+0.1, LOC[1], LOC[2])
        # size = (4, 10000, 10000)
        # data = get_one_stack_vol(data, pair_tag1, brain_tag1, LOC1, f'', size)
        data = get_one_stack_vol(data, pair_tag2, brain_tag2, LOC2, f'loc2', size)
    else:
        # LOC = (0.5, 0.3, 0.3)
        # LOC = (np.random.randint(1,1000)/1200, 0.5, 0.5)
        data = get_one_stack_vol(data, pair_tag1, brain_tag1, LOC, f'', size)
        data = get_one_stack_vol(data, pair_tag2, brain_tag2, LOC, f'', size)

    return data

def plot_one_hist(data, pair_tag1, brain_tag1, pair_tag2, brain_tag2, key='vol'): 
    tail = '' if key == 'vol' else '_density'
    # data = pd.DataFrame(data)
    if brain_tag1 != brain_tag2:
        mask = (data[f'brain_tag{tail}']==brain_tag1) | (data[f'brain_tag{tail}']==brain_tag2)
    else:
        mask = (data[f'brain_tag{tail}']==brain_tag1) | (data[f'brain_tag{tail}']==brain_tag2+'loc2')
    data = data[mask]
    if len(data[key]) == 0: return
    colors = [(255, 0, 0), (0, 119, 0)]
    # colors = [[c/255 for c in color] for color in colors]
    colors = {d: [c/255 for c in color] for d, color in zip(data[f'brain_tag{tail}'].unique(), colors)}
    # print(data)
    ##### Hist plot
    plt.figure(figsize=(5,4))
    ax = sns.histplot(data, x=key, hue=f'brain_tag{tail}', kde=True, multiple="dodge", shrink=.8, linewidth=0, bins=32, palette=colors, alpha=0.4,
                  line_kws={'lw': 1.5, 'ls': '--'}, kde_kws={'gridsize': 5000})
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="y",direction="in")
    ax.tick_params(axis="x",direction="in")
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_xlim([0, 25])
    try:
        ax.get_legend().remove()
    except:
        pass
    plt.tight_layout()
    plt.savefig(f'stats/plots/{key}_{pair_tag1}-{pair_tag2}-{brain_tag1}-{brain_tag2}_hist.svg')
    plt.savefig(f'stats/plots/{key}_{pair_tag1}-{pair_tag2}-{brain_tag1}-{brain_tag2}_hist.png')
    plt.close()
    ###########################
    ##### violin plot
    plt.figure(figsize=(1,2.5))
    ax = sns.violinplot(data, y=key, x=f'brain_tag{tail}', palette=colors, linewidth=0)
    plt.setp(ax.collections, alpha=.7)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(f'stats/plots/{key}_{pair_tag1}-{pair_tag2}-{brain_tag1}-{brain_tag2}_viol.svg')
    plt.savefig(f'stats/plots/{key}_{pair_tag1}-{pair_tag2}-{brain_tag1}-{brain_tag2}_viol.png')
    plt.close()
    ###########################

def plot_one_kde(data, pair_tag1, brain_tag1, pair_tag2, brain_tag2, kdeax, key='vol'):
    tail = '' if key == 'vol' else '_density'
    # data = pd.DataFrame(data)
    if brain_tag1 != brain_tag2:
        mask = (data[f'brain_tag{tail}']==brain_tag1) | (data[f'brain_tag{tail}']==brain_tag2)
    else:
        mask = (data[f'brain_tag{tail}']==brain_tag1) | (data[f'brain_tag{tail}']==brain_tag2+'loc2')
    data = data[mask]
    if len(data[key]) == 0: return
    colors = [(255, 0, 0), (0, 119, 0)]
    # colors = [[c/255 for c in color] for color in colors]
    colors = {d: [c/255 for c in color] for d, color in zip(data[f'brain_tag{tail}'].unique(), colors)}
    # print(data)
    ##### kde plot
    sns.kdeplot(data, x=key, hue=f'brain_tag{tail}', palette=colors, legend=False, ax=kdeax, linestyle='--', linewidth=2.5, gridsize=5000)
    kdeax.set_title(f'{brain_tag1}-{brain_tag2}', fontsize=8)

def get_pair_brain(gender_filter):
    pair_tags = [p for p in os.listdir(STAT_ROOT) if p.startswith('pair')]
    brain_label = pd.read_csv('downloads/brain_gene_label.csv')
    brain_label = brain_label[[os.path.exists(f"{STAT_ROOT}/pair{p}/{b}_nis_center.zip") for p, b in zip(brain_label['Pair'], brain_label['Brain'])]]
    valid_brain = None
    collapsed_brain = ['L57D855P1', 'L77D764P8', 'L59D878P1']
    for collapsed in collapsed_brain:
        if valid_brain is None:
            valid_brain = (brain_label['Brain'] != collapsed) & ('LocalNorm' not in brain_label['Brain'])
        else:
            valid_brain &= (brain_label['Brain'] != collapsed) & ('LocalNorm' not in brain_label['Brain'])
    brain_label = brain_label[valid_brain]
    brains = pd.concat([brain_label[brain_label['Pair'] == int(pair_tag.replace('pair', ''))] for pair_tag in pair_tags])
    if gender_filter != 'All':
        pair_tags = list(brains[brains['Gender']==gender_filter]['Pair'])
        brain_tags = list(brains[brains['Gender']==gender_filter]['Brain'])
    else:
        pair_tags = list(brains['Pair'])
        brain_tags = list(brains['Brain'])
    return pair_tags, brain_tags

def save_allstack():
    pair_tags, brain_tags = get_pair_brain('All')
    data = {'vol': [], 'density': [], 'brain_tag': [], 'brain_tag_density': [], 'stack_range': None, 'stack_range_density': None}
    
    for j in trange(len(pair_tags)):
        pair_tag = f'pair{pair_tags[j]}'
        brain_tag = brain_tags[j]
        data = get_one_stack_vol(data, pair_tag, brain_tag, LOC, f'', size)
        # data = get_pair_nis(pair_tag1, brain_tag1, pair_tag2, brain_tag2, data)
    LOC2 = (LOC[0]+0.1, LOC[1], LOC[2])

    for j in trange(len(pair_tags)):
        pair_tag = f'pair{pair_tags[j]}'
        brain_tag = brain_tags[j]
        data = get_one_stack_vol(data, pair_tag, brain_tag, LOC2, f'loc2', size)

    data = {k: np.array(data[k]) for k in data}
    def dftosave(data):
        dftosave = {}
        for k in data:
            if len(data[k].shape) == 2:
                for d in range(data[k].shape[1]):
                    dftosave[f'{k}:dim{d}'] = data[k][:, d]
            else:
                dftosave[k] = data[k]
        return pd.DataFrame(dftosave)
    dftosave({k:data[k] for k in data if 'density' not in k}).to_pickle(f'stats/All-All_brain_random_stack_vol.pkl')
    dftosave({k:data[k] for k in data if 'density' in k}).to_pickle(f'stats/All-All_brain_random_stack_density.pkl')

def plot_all_kde(pair_tags1, brain_tags1, pair_tags2, brain_tags2, plot_key = 'density', kde_row_first=True):
    nrows = len(pair_tags1)
    ncols = len(pair_tags2)
    fig, kdeaxes = plt.subplots(nrows, ncols, figsize=(nrows*2, ncols*2))
    for rowi in range(nrows):
        for colj in range(ncols):
            if kde_row_first:
                i = rowi
                j = colj
            else:
                j = rowi
                i = colj
            kdeaxes[i,j].set_xticklabels([])
            kdeaxes[i,j].set_yticklabels([])
            kdeaxes[i,j].set_xticks([])
            kdeaxes[i,j].set_yticks([])
            kdeaxes[i,j].set_xlabel('')
            kdeaxes[i,j].set_ylabel('')
            if j<i: kdeaxes[i,j].axis('off')
    df = pd.read_pickle(f'stats/{filter1}-{filter2}_brain_random_stack_{plot_key}.pkl')
    for i in trange(len(pair_tags2)):
        pair_tag1 = f'pair{pair_tags1[i]}'
        brain_tag1 = brain_tags1[i]
        for j in trange(i, len(pair_tags2)):
            pair_tag2 = f'pair{pair_tags2[j]}'
            brain_tag2 = brain_tags2[j]
            plot_one_kde(df, pair_tag1, brain_tag1, pair_tag2, brain_tag2, kdeaxes[i,j] if kde_row_first else kdeaxes[j,i], plot_key)
    fig.savefig(f'stats/plots/{plot_key}_{filter1}-{filter2}_brain_kdeplot.png')
    fig.savefig(f'stats/plots/{plot_key}_{filter1}-{filter2}_brain_kdeplot.svg')
    plt.close(fig)

def save_brain_size_vs_time():
    pair_tags, brain_tags = get_pair_brain('All')
    data = {'brain_size': [], 'comp_time': [], 'brain_tag': []}
    for pair, brain in tqdm(zip(pair_tags, brain_tags), total=len(brain_tags)):
        density_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/renders/P4/pair{pair}/NIS_density_pair{pair}_{brain}.nii'
        density_map = nib.load(density_path).get_fdata()
        bsize = len(np.where(density_map>0)[0])
        log_path = f'downloads/cpp_logs/{brain}.log'
        if not os.path.exists(log_path): continue
        with open(log_path, 'r') as logf:
            logs = logf.read().split('\n')
        if len(logs) < 2: continue
        if logs[-2] != 'ok': 
            continue
        start_time = datetime.strptime(logs[2], '%a %b %d %H:%M:%S %Y')
        end_time = datetime.strptime(logs[-4], '%a %b %d %H:%M:%S %Y')
        data['brain_size'].append(bsize)
        data['brain_tag'].append(brain)
        data['comp_time'].append((end_time-start_time).seconds / 3600)
    pd.DataFrame(data).to_pickle(f'stats/All_brain_size_vs_time.pkl')

def plot_brain_size_vs_time():
    data = pd.read_pickle(f'stats/All_brain_size_vs_time.pkl')
    data['brain_size'] = ((data['brain_size'] * ((0.025)**3))/7).astype(int) * 7
    print(data)
    plt.figure(figsize=(5,3))
    ax = sns.lineplot(data=data, x='brain_size', y='comp_time')
    ax.set(xlabel='Brain size (mm$^3$)', ylabel='Computational time (hr)')
    plt.savefig('stats/brain_size_vs_time.png')
    plt.savefig('stats/brain_size_vs_time.svg')
    plt.close()

def plot_avg_nis():
    # vol = 200
    pair_tags, brain_tags = get_pair_brain('All')
    for vol in [200, 400, 600]:
        for pair_tag, brain_tag in zip(pair_tags, brain_tags):
            pair_tag = f'pair{pair_tag}'
            nis_max_num = 3000
            stat_root = f"{STAT_ROOT}/{pair_tag}"
            total_vol = torch.load(f"{stat_root}/{brain_tag}_nis_volume.zip", map_location='cpu')
            nis_id = torch.where(total_vol==vol)[0].tolist()
            random.shuffle(nis_id)
            nis_id = nis_id[:nis_max_num]
            nis_id.sort()
            frame, all_pa = get_one_vol_frame(pair_tag, brain_tag, nis_id)
            nibimg = nib.Nifti1Image(frame.numpy(), np.eye(4))
            nib.save(nibimg, f'stats/avg_nis/{pair_tag}_{brain_tag}_vol={vol}.nii.gz')
            torch.save(all_pa, f'stats/avg_nis/{pair_tag}_{brain_tag}_vol={vol}_axis.zip')
            # exit()
        # volume = vedo.Volume(frame.numpy())
        # volume.cmap(['white','b','g','r']).mode(1)
        # # volume.add_scalarbar()
        # vedo.show(volume, __doc__, axes=1).close()
        # # plt.savefig('temp.png')
        # exit()
            
def unique_return_index(A):
    unique, idx, counts = torch.unique(A, dim=1, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return unique, first_indicies

def plot_density_vs_volume():
    pair_tags, brain_tags = get_pair_brain('All')
    if not os.path.exists(f'stats/density_vs_volume_plot.zip'):
        data_list = {}
    else:
        data_list = torch.load(f'stats/density_vs_volume_plot.zip')
    resolution = (0.75*0.75*2.5)
    vol_range = [100, 600]
    for pair_tag, brain_tag in tqdm(zip(pair_tags, brain_tags), total=len(pair_tags), desc='Preload maps'):
        pair_tag = f'pair{pair_tag}'
        if pair_tag not in data_list: data_list[pair_tag] = {'Density': [], 'Volume': [], 'Brain': []}
        if brain_tag in data_list[pair_tag]['Brain']: continue
        render_root = f"/cajal/ACMUSERS/ziquanw/Lightsheet/renders/P4/{pair_tag}"
        density = nib.load(f"{render_root}/NIS_density_{pair_tag}_{brain_tag}.nii").get_fdata()
        avgvol = nib.load(f"{render_root}/NIS_volavg_{pair_tag}_{brain_tag}.nii").get_fdata()
        mask = (density > 0) & (avgvol >= vol_range[0]) & (avgvol <= vol_range[1])
        density = torch.from_numpy(density[mask])#.long()
        avgvol = torch.from_numpy(avgvol[mask])*resolution#.long()
        if density.shape[0] == 0: continue
        # precision = 10
        # density = (density*precision).long().float() / precision
        # avgvol = (avgvol*precision).long().float() / precision
        # _, first_indicies = unique_return_index(torch.stack([density, avgvol], dim=0).long())
        # density = density[first_indicies]
        # avgvol = avgvol[first_indicies]
        density, avgvol = torch.stack([density, avgvol], dim=0).unique(dim=1)
        print(f"{pair_tag}, {brain_tag}", density.shape, avgvol.shape)
        data_list[pair_tag]['Density'].append(density.reshape(-1))
        data_list[pair_tag]['Volume'].append(avgvol.reshape(-1))
        data_list[pair_tag]['Brain'].extend([brain_tag for _ in range(len(density.reshape(-1)))])
    torch.save(data_list, f'stats/density_vs_volume_plot.zip')
    cmaps = [(255/255, 0, 0), (0, 119/255, 0)]
    # pair_tags1, brain_tags1 = get_pair_brain('Male')
    # pair_tags2, brain_tags2 = get_pair_brain('Female')
    # pairs = [(f'pair{p1}',f'pair{p2}',b1,b2) for p1, b1 in zip(pair_tags1, brain_tags1) for p2, b2 in zip(pair_tags2, brain_tags2) if b1 != b2 and b2=='L66D764P5']
    pair_tags, brain_tags = get_pair_brain('All')
    pairs = [(f'pair{p1}',f'pair{p2}',b1,b2) for p1, b1 in zip(pair_tags, brain_tags) for p2, b2 in zip(pair_tags, brain_tags) if b1 != b2 and p1 == p2]
    for p1, p2, b1, b2 in tqdm(pairs,desc='Plotting'):
        if p1 not in data_list or p2 not in data_list: continue
        if isinstance(data_list[p1]['Density'], list):
            data_list[p1]['Density'] = np.concatenate(data_list[p1]['Density'])
            data_list[p1]['Volume'] = np.concatenate(data_list[p1]['Volume'])
        if isinstance(data_list[p2]['Density'], list):
            data_list[p2]['Density'] = np.concatenate(data_list[p2]['Density'])
            data_list[p2]['Volume'] = np.concatenate(data_list[p2]['Volume'])
        data1 = pd.DataFrame(data_list[p1])
        data1 = data1[data1['Brain']==b1]
        data2 = pd.DataFrame(data_list[p2])
        data2 = data2[data2['Brain']==b2]
        data = pd.concat([data1, data2])
        # print(data)
        if len(data['Brain'].unique()) == 1: continue
        platte = {}
        for bi, brain in enumerate(data['Brain'].unique()):
            platte[brain] = cmaps[bi]

        print(data)
        g = sns.jointplot(
            data=data,
            x="Density", y="Volume", hue="Brain", kind="kde", hue_order=list(data['Brain'].unique())[::-1],
            fill=True, zorder=0, alpha=0.3, palette=platte, thresh=0.05#, levels=100
        )
        # g.plot_joint(sns.rugplot, data=data.iloc[dot_list], lw=1, alpha=.005)#, height=-.15, clip_on=False
        # g.plot_joint(sns.scatterplot, data=data.iloc[dot_list], s=1)#, height=-.15, clip_on=False
        plt.savefig(f'stats/density_vs_volume_plot/{b1}-{b2}.png')
        plt.savefig(f'stats/density_vs_volume_plot/{b1}-{b2}.svg')
        # exit()


def stats_pa_by_p4_atlas():
    pair_tags, brain_tags = get_pair_brain('All')
    resolution = [2.5, 0.75, 0.75]
    atlas_res = [25, 25, 25]
    eps = 1e-3
    if os.path.exists('stats/stats_pa_by_p4_atlas.zip'):
        data_list = torch.load('stats/stats_pa_by_p4_atlas.zip')
    else:
        data_list = {}
    for pair_tag, brain_tag in tqdm(zip(pair_tags, brain_tags), total=len(pair_tags), desc='Preload'):
        pair_tag = f'pair{pair_tag}'
        fn = f'/cajal/ACMUSERS/ziquanw/Lightsheet/statistics/P4/{pair_tag}/{brain_tag}_nis_pa.zip'
        atlas_fn = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/registered/{brain_tag}_MASK_topro_25_all.nii'
        if not os.path.exists(fn) or not os.path.exists(atlas_fn): continue
        if pair_tag not in data_list: data_list[pair_tag] = {'PA1/PA2': [], 'PA1/PA3': [], 'rid': [], 'Brain': [], 'vol': [], 'density': []}
        if brain_tag in data_list[pair_tag]['Brain']: continue
        atlas = np.transpose(nib.load(atlas_fn).get_fdata(), (2,0,1))
        nis_center = torch.load(fn.replace('pa.zip', 'center.zip'), map_location='cpu')
        nis_vol = torch.load(fn.replace('pa.zip', 'volume.zip'), map_location='cpu')
        density = nib.load(f'/cajal/ACMUSERS/ziquanw/Lightsheet/renders/P4/{pair_tag}/NIS_density_{pair_tag}_{brain_tag}.nii').get_fdata()
        data_dict = torch.load(fn)
        center = nis_center[data_dict['NIS id']]
        data = data_dict['PA'] # N x 3 x 3
        data[..., 0] = data[..., 0] * resolution[0]
        data[..., 1] = data[..., 1] * resolution[1]
        data[..., 2] = data[..., 2] * resolution[2]
        pa_length = torch.sqrt((data**2).sum(2)).sort(dim=1, descending=True)[0] # N x 3
        data1 = pa_length[:, 0]/pa_length[:, 1].clip(min=eps) # N
        data2 = pa_length[:, 0]/pa_length[:, 2].clip(min=eps) # N
        data_list[pair_tag]['PA1/PA2'].append(data1)
        data_list[pair_tag]['PA1/PA3'].append(data2)
        data_list[pair_tag]['Brain'].extend([brain_tag for _ in range(len(data1))])
        # data_list[pair_tag]['num'].append(len(data1))
        data_list[pair_tag]['vol'].append(nis_vol[data_dict['NIS id']])
        data_list[pair_tag]['rid'].append(atlas[
            (center[:, 0]*resolution[0]/atlas_res[0]).long().clip(max=atlas.shape[0]-1),
            (center[:, 1]*resolution[1]/atlas_res[1]).long().clip(max=atlas.shape[1]-1),
            (center[:, 2]*resolution[2]/atlas_res[2]).long().clip(max=atlas.shape[2]-1)
        ])
        data_list[pair_tag]['density'].append(density[
            (center[:, 0]*resolution[0]/atlas_res[0]).long().clip(max=density.shape[0]-1),
            (center[:, 1]*resolution[1]/atlas_res[1]).long().clip(max=density.shape[1]-1),
            (center[:, 2]*resolution[2]/atlas_res[2]).long().clip(max=density.shape[2]-1)
        ])
    torch.save(data_list, 'stats/stats_pa_by_p4_atlas.zip')

    import pingouin as pg
    import sys
    within_key = sys.argv[1]#'Gene'
    brain_csv = pd.read_csv('downloads/brain_gene_label.csv')
    blabel = {row['Brain']:row[within_key] for index, row in brain_csv[['Brain', within_key]].iterrows()}
    glabel = {row['Brain']:row["Gene"] for index, row in brain_csv[['Brain', 'Gene']].iterrows()}
    # all_data = {g: [] for g in list(set(list(blabel.values())))}
    # all_data = {}
    dv_key = sys.argv[2]
    # dv_key = 'density'#'vol'# 'PA1/PA3'，'density'
    outlier_r = 1
    for pair_tag in data_list:
        data_list[pair_tag]['rid'] = torch.from_numpy(np.concatenate(data_list[pair_tag]['rid']))
        data_list[pair_tag]['density'] = torch.from_numpy(np.concatenate(data_list[pair_tag]['density']))
        data_list[pair_tag]['vol'] = torch.cat(data_list[pair_tag]['vol'])
        data_list[pair_tag]['PA1/PA2'] = torch.cat(data_list[pair_tag]['PA1/PA2'])
        data_list[pair_tag]['PA1/PA3'] = torch.cat(data_list[pair_tag]['PA1/PA3'])
        data_list[pair_tag][within_key] = [blabel[b] for b in data_list[pair_tag]['Brain']]
        data_list[pair_tag]['pair'] = [pair_tag for b in data_list[pair_tag]['Brain']]
    tgt_rids = data_list[list(data_list.keys())[0]]['rid'].unique().tolist()[1:]
    fig, axes = plt.subplots(1,len(tgt_rids), figsize=(len(tgt_rids)*2, 5))
    for ri, tgt_rid in enumerate(tgt_rids):
        ax = axes[ri]
        mean_data = {'PA1/PA2': [], 'PA1/PA3': [], 'brain': [], 'pair':[], within_key: [], 'vol': [], 'density': []}
        for pair_tag in data_list:
            data = pd.DataFrame(data_list[pair_tag])
            data = data[data['rid']==tgt_rid]
            
            for b in data['Brain'].unique():
                mean_data['PA1/PA2'].append(np.mean(data[data['Brain']==b]['PA1/PA2']))
                mean_data['PA1/PA3'].append(np.mean(data[data['Brain']==b]['PA1/PA3']))
                mean_data['vol'].append(np.mean(data[data['Brain']==b]['vol']))
                mean_data['density'].append(np.mean(data[data['Brain']==b]['density']))
                mean_data['brain'].append(b)
                mean_data['pair'].append(pair_tag)
                mean_data[within_key].append(blabel[b])
        mean_data = pd.DataFrame(mean_data)
        new_data = []
        for k in mean_data[within_key].unique():
            data = np.array(mean_data[mean_data[within_key]==k][dv_key])
            mean = np.mean(data)
            std = np.std(data)
            mask = (data>=(mean-outlier_r*std)) & (data<=(mean+outlier_r*std))
            new_data.append(mean_data[mean_data[within_key]==k][mask])
        if len(new_data) == 0:continue
        new_data = pd.concat(new_data)
        mean_data = new_data
        if tgt_rid == 16001:
            sep_fig, sep_ax = plt.subplots(1,1,figsize=(2,5))
        Groups = {}
        if within_key == 'Gene':
            for p in mean_data['pair'].unique():
                if len(mean_data[mean_data['pair']==p]) == 1: continue
                for k in mean_data[mean_data['pair']==p][within_key]:
                    if k not in Groups: Groups[k] = []
                    Groups[k].append(mean_data[(mean_data['pair']==p)&(mean_data[within_key]==k)][dv_key].item())
            if len(Groups) < 2: continue
            test_result = pg.ttest(Groups['WT'], Groups['HET'], paired=True)
            ax = pg.plot_paired(data=mean_data, dv=dv_key, within=within_key, subject='pair', boxplot=True, ax=ax)#, colors=[(255/255, 0, 0), (0, 119/255, 0)]
            ax.set_title(f'RID: {tgt_rid}\np = {test_result["p-val"].item():.3f} (n={len(Groups["WT"])*2})')
            if tgt_rid == 16001:
                sep_ax = pg.plot_paired(data=mean_data, dv=dv_key, within=within_key, subject='pair', boxplot=True, ax=sep_ax)#, colors=[(255/255, 0, 0), (0, 119/255, 0)]
                sep_ax.set_title(f'RID: {tgt_rid}\np = {test_result["p-val"].item():.3f} (n={len(Groups["WT"])*2})')
                sep_fig.tight_layout()
                sep_fig.savefig(f"stats/stats_pa_by_p4_atlas_isocortex_{within_key.lower()}_{dv_key.replace('/', '-')}.png")
                sep_fig.savefig(f"stats/stats_pa_by_p4_atlas_isocortex_{within_key.lower()}_{dv_key.replace('/', '-')}.svg")
        else:
            Groups['Female'] = list(mean_data[mean_data[within_key]=='Female'][dv_key])
            Groups['Male'] = list(mean_data[mean_data[within_key]=='Male'][dv_key])
            if len(Groups['Female'])<=0 or len(Groups['Male'])<=0: continue
            test_result = pg.ttest(Groups['Female'], Groups['Male'], paired=False)
            ax = sns.boxplot(data=mean_data, x=within_key, y=dv_key, ax=ax, orient='v', zorder=0, color="#137", boxprops={"facecolor": (0,0,0,0)})
            glist = np.array([glabel[b] for b in mean_data['brain']])
            ax = sns.scatterplot(data=mean_data[glist=='WT'], x=within_key, y=dv_key, ax=ax, color='blue')
            ax = sns.scatterplot(data=mean_data[glist=='HET'], x=within_key, y=dv_key, ax=ax, color='red')
            ax.set_title(f'RID: {tgt_rid}\np = {test_result["p-val"].item():.3f} (n={len(Groups["Female"])+len(Groups["Male"])})')
            if tgt_rid == 16001:
                sep_ax = sns.boxplot(data=mean_data, x=within_key, y=dv_key, ax=sep_ax, orient='v', zorder=0, color="#137", boxprops={"facecolor": (0,0,0,0)})
                sep_ax = sns.scatterplot(data=mean_data[glist=='WT'], x=within_key, y=dv_key, ax=sep_ax, color='blue')
                sep_ax = sns.scatterplot(data=mean_data[glist=='HET'], x=within_key, y=dv_key, ax=sep_ax, color='red')
                sep_ax.set_title(f'RID: {tgt_rid}\np = {test_result["p-val"].item():.3f} (n={len(Groups["Female"])+len(Groups["Male"])})')
                sep_fig.tight_layout()
                sep_fig.savefig(f"stats/stats_pa_by_p4_atlas_isocortex_{within_key.lower()}_{dv_key.replace('/', '-')}.png")
                sep_fig.savefig(f"stats/stats_pa_by_p4_atlas_isocortex_{within_key.lower()}_{dv_key.replace('/', '-')}.svg")
        # print(f'P value (n={len(Groups["WT"])*2})', test_result['p-val'])
        
        # if within_key == 'Gene':
        # else:
    plt.tight_layout()
    plt.savefig(f"stats/stats_pa_by_p4_atlas_{within_key.lower()}_{dv_key.replace('/', '-')}.png")
    plt.savefig(f"stats/stats_pa_by_p4_atlas_{within_key.lower()}_{dv_key.replace('/', '-')}.svg")
    

def plot_avg_nis_by_pa():
    pair_tags, brain_tags = get_pair_brain('All')
    resolution = [2.5, 0.75, 0.75]
    if os.path.exists('stats/avg_nis_by_pa.zip'):
        data_list = torch.load('stats/avg_nis_by_pa.zip')
    else:
        data_list = {}
        for pair_tag, brain_tag in tqdm(zip(pair_tags, brain_tags), total=len(pair_tags), desc='Preload'):
            pair_tag = f'pair{pair_tag}'
            fn = f'/cajal/ACMUSERS/ziquanw/Lightsheet/statistics/P4/{pair_tag}/{brain_tag}_nis_pa.zip'
            if not os.path.exists(fn): continue
            if pair_tag not in data_list: data_list[pair_tag] = {'avg frames': [], 'nis center': [], 'thr': [], 'nis num': [], 'Brain': []}
            if brain_tag in data_list[pair_tag]['Brain']: continue
            nis_center = torch.load(fn.replace('pa.zip', 'center.zip'))
            data_dict = torch.load(fn)
            data = data_dict['PA'] # N x 3 x 3
            data[..., 0] = data[..., 0] * resolution[0]
            data[..., 1] = data[..., 1] * resolution[1]
            data[..., 2] = data[..., 2] * resolution[2]
            pa_length = torch.sqrt((data**2).sum(2)).sort(dim=1, descending=True)[0]
            data1 = pa_length[:, 0]/pa_length[:, 1]
            data2 = pa_length[:, 0]/pa_length[:, 2]
            prec = 10
            data1 = ((data1 * prec).long().float() / prec)
            data2 = ((data2 * prec).long().float() / prec)
            for data1th in range(10,25):
                data1th /= prec
                mask = (data2 == 2.5) & (data1 == data1th)
                nis_id = data_dict['NIS id'][mask].tolist()
                nis_id.sort()
                if len(nis_id) == 0: continue
                avg_frames, _ = get_one_vol_frame(pair_tag, brain_tag, nis_id)
                data_list[pair_tag]['avg frames'].append(avg_frames)
                data_list[pair_tag]['nis center'].append(nis_center[nis_id])
                data_list[pair_tag]['nis num'].append(len(nis_id))
                data_list[pair_tag]['thr'].append(torch.FloatTensor([data1th, 2.5]))
                data_list[pair_tag]['Brain'].append(brain_tag)
            # data_list[pair_tag]['avg frames'] = torch.stack(data_list[pair_tag]['avg frames'])
        for pair_tag in data_list:
            data_list[pair_tag]['nis center'] = torch.cat(data_list[pair_tag]['nis center'])
            data_list[pair_tag]['nis num'] = torch.LongTensor(data_list[pair_tag]['nis num'])
            data_list[pair_tag]['thr'] = torch.stack(data_list[pair_tag]['thr'])
        torch.save(data_list, 'stats/avg_nis_by_pa.zip')
    import vedo
    # atlas_res = [25, 25, 25]
    for pair_tag in data_list:
        # for brain_tag in data_list[pair_tag]['Brain'].unique():
        for i, avg_frame in enumerate(data_list[pair_tag]['avg frames']):
            brain_tag = data_list[pair_tag]['Brain'][i]
            thr = [str(t) for t in list(data_list[pair_tag]['thr'][i])]
            # atlas = nib.load(f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/registered/{brain_tag}_MASK_topro_25_all.nii').get_fdata()
            # vp = vedo.Plotter(offscreen=True)
            # vp += vedo.Volume(avg_frame)
            # vp.show(interactive=False, zoom=2.5, camera={'pos':(1,2,3), 'viewAngle': 0.5})
            # vedo.screenshot(f'{pair_tag}_{brain_tag}_{"-".join(thr)}.png')
            # vedo.screenshot(f'{pair_tag}_{brain_tag}_{"-".join(thr)}.svg')
            # vp.close()
            # exit()
            nib.save(nib.Nifti1Image(avg_frame.numpy(), np.eye(4)), f'stats/avg_nis_bypa/{pair_tag}_{brain_tag}_{"-".join(thr)}.nii')
            

def plot_pa1_vs_pa2():
    pair_tags, brain_tags = get_pair_brain('All')
    data_list = {}
    resolution = [2.5, 0.75, 0.75]
    for pair_tag, brain_tag in tqdm(zip(pair_tags, brain_tags), total=len(pair_tags), desc='Preload'):
        pair_tag = f'pair{pair_tag}'
        fn = f'/cajal/ACMUSERS/ziquanw/Lightsheet/statistics/P4/{pair_tag}/{brain_tag}_nis_pa.zip'
        if not os.path.exists(fn): continue
        if pair_tag not in data_list: data_list[pair_tag] = {'PA1/PA2': [], 'PA1/PA3': [], 'Brain': []}
        if brain_tag in data_list[pair_tag]['Brain']: continue
        data = torch.load(fn)['PA'] # N x 3 x 3
        data[..., 0] = data[..., 0] * resolution[0]
        data[..., 1] = data[..., 1] * resolution[1]
        data[..., 2] = data[..., 2] * resolution[2]
        pa_length = torch.sqrt((data**2).sum(2)).sort(dim=1, descending=True)[0]
        data1 = pa_length[:, 0]/pa_length[:, 1]
        data2 = pa_length[:, 0]/pa_length[:, 2]
        mask = (data1<=5) & (data2<=5)
        data1 = data1[mask]
        data2 = data2[mask]
        data_list[pair_tag]['PA1/PA2'].append(data1)
        data_list[pair_tag]['PA1/PA3'].append(data2)
        data_list[pair_tag]['Brain'].extend([brain_tag for _ in range(len(data1))])
    torch.save(data_list, f'stats/pa1_vs_pa2_plot.zip')
    cmaps = [(255/255, 0, 0), (0, 119/255, 0)]
    pair_tags1, brain_tags1 = get_pair_brain('Male')
    pair_tags2, brain_tags2 = get_pair_brain('Female')
    pairs = [(f'pair{p1}',f'pair{p2}',b1,b2) for p1, b1 in zip(pair_tags1, brain_tags1) for p2, b2 in zip(pair_tags2, brain_tags2) if b1 != b2]
    # pair_tags, brain_tags = get_pair_brain('All')
    # pairs = [(f'pair{p1}',f'pair{p2}',b1,b2) for p1, b1 in zip(pair_tags, brain_tags) for p2, b2 in zip(pair_tags, brain_tags) if b1 != b2 and p1 == p2]
    for p1, p2, b1, b2 in tqdm(pairs,desc='Plotting'):
        if p1 not in data_list or p2 not in data_list: continue
    # for pair_tag in tqdm(data_list,desc='Plotting'):
        if isinstance(data_list[p1]['PA1/PA2'], list):
            data_list[p1]['PA1/PA2'] = torch.cat(data_list[p1]['PA1/PA2'])#.tolist()
            data_list[p1]['PA1/PA3'] = torch.cat(data_list[p1]['PA1/PA3'])#.tolist()
        if isinstance(data_list[p2]['PA1/PA2'], list):
            data_list[p2]['PA1/PA2'] = torch.cat(data_list[p2]['PA1/PA2'])#.tolist()
            data_list[p2]['PA1/PA3'] = torch.cat(data_list[p2]['PA1/PA3'])#.tolist()
        
        data1 = pd.DataFrame(data_list[p1])
        data1 = data1[data1['Brain']==b1]
        data2 = pd.DataFrame(data_list[p2])
        data2 = data2[data2['Brain']==b2]
        data = pd.concat([data1, data2])
        # data = pd.DataFrame(data_list[pair_tag])

        if len(data['Brain'].unique()) == 1: continue
        platte = {}
        for bi, brain in enumerate(data['Brain'].unique()):
            platte[brain] = cmaps[bi]
            
        print(data)
        g = sns.jointplot(
            data=data,
            x="PA1/PA2", y="PA1/PA3", hue="Brain", hue_order=list(data['Brain'].unique())[::-1],
            kind="kde", fill=True,
            # kind="scatter", s=3,
            alpha=0.3, palette=platte,
            zorder=0, thresh=0.01#, levels=100, 
        )
        # g.plot_joint(sns.scatterplot, data=data, s=1, alpha=0.1)
        # sns.kdeplot(data=data,
        #     x="PA1/PA2", y="PA1/PA3", hue="Brain"
        #     , palette=platte,)
        
        # plt.savefig(f'stats/pa1_vs_pa2_plot/{pair_tag}.png')
        # plt.savefig(f'stats/pa1_vs_pa2_plot/{pair_tag}.svg')
        plt.savefig(f'stats/pa1_vs_pa2_plot/{b1}-{b2}.png')
        plt.savefig(f'stats/pa1_vs_pa2_plot/{b1}-{b2}.svg')


if __name__=="__main__":
    filter1 = 'All'
    filter2 = 'All'
    ####
    stats_pa_by_p4_atlas()
    #####
    # plot_avg_nis_by_pa()
    #####
    # plot_pa1_vs_pa2()
    #####
    # plot_density_vs_volume()
    ####
    # plot_avg_nis()
    #####
    # save_brain_size_vs_time()
    # plot_brain_size_vs_time()
    #####
    # save_allstack()
    #####
    # pair_tags1, brain_tags1 = get_pair_brain(filter1)
    # pair_tags2, brain_tags2 = get_pair_brain(filter2)
    # plot_all_kde(pair_tags1, brain_tags1, pair_tags2, brain_tags2, 'density')
    # plot_all_kde(pair_tags1, brain_tags1, pair_tags2, brain_tags2, 'vol', False)
    # #####
    # # plot_key = 'vol'
    # # plot_one_hist(pd.read_pickle(f'stats/{filter1}-{filter2}_brain_random_stack_{plot_key}.pkl'), 'pair18', 'L77D764P4', 'pair18', 'L77D764P4', plot_key)
    # # plot_key = 'density'
    # # plot_one_hist(pd.read_pickle(f'stats/{filter1}-{filter2}_brain_random_stack_{plot_key}.pkl'), 'pair18', 'L77D764P4', 'pair18', 'L77D764P4', plot_key)
    # # plot_key = 'vol'
    # # plot_one_hist(pd.read_pickle(f'stats/{filter1}-{filter2}_brain_random_stack_{plot_key}.pkl'), 'pair12', 'L66D764P5', 'pair21', 'L91D814P6', plot_key)
    # # plot_key = 'density'
    # # plot_one_hist(pd.read_pickle(f'stats/{filter1}-{filter2}_brain_random_stack_{plot_key}.pkl'), 'pair12', 'L66D764P5', 'pair21', 'L91D814P6', plot_key)
    # # plot_key = 'vol'
    # # plot_one_hist(pd.read_pickle(f'stats/{filter1}-{filter2}_brain_random_stack_{plot_key}.pkl'), 'pair16', 'L74D769P4', 'pair16', 'L74D769P8', plot_key)
    # # plot_key = 'density'
    # # plot_one_hist(pd.read_pickle(f'stats/{filter1}-{filter2}_brain_random_stack_{plot_key}.pkl'), 'pair16', 'L74D769P4', 'pair16', 'L74D769P8', plot_key)
    # plot_key = 'vol'
    # plot_one_hist(pd.read_pickle(f'stats/{filter1}-{filter2}_brain_random_stack_{plot_key}.pkl'), 'pair11', 'L66D764P3', 'pair11', 'L66D764P8', plot_key)
    # plot_key = 'density'
    # plot_one_hist(pd.read_pickle(f'stats/{filter1}-{filter2}_brain_random_stack_{plot_key}.pkl'), 'pair11', 'L66D764P3', 'pair11', 'L66D764P8', plot_key)
     