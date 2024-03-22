import torch, os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import trange

np.random.seed(142857)
STAT_ROOT = '/cajal/ACMUSERS/ziquanw/Lightsheet/statistics'

def main(pair_tag1, brain_tag1, pair_tag2, brain_tag2, kdeax, not_plot=False, data={'vol': [], 'brain_tag': []}):
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
        stack_vol = stack_vol[stack_vol<=25]
        d['vol'] += stack_vol.tolist()
        d['brain_tag'] += [brain_tag+key for _ in range(len(stack_vol))]
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
            d['brain_tag_density'] += [brain_tag+key for _ in range(len(loc_count))]
        return d
    
    if brain_tag1 == brain_tag2:
        # LOC1 = (0.5, 0.3, 0.3)
        # LOC2 = (0.3, 0.5, 0.3)
        # LOC1 = list(np.random.randint(1,1000,(3,))/1500)
        # LOC2 = list(np.random.randint(1,1000,(3,))/1500)
        # size = (64, 128, 128)
        LOC1 = (np.random.randint(1,1000)/1200, 0.5, 0.5)
        LOC2 = (np.random.randint(1,1000)/1200, 0.5, 0.5)
        size = (4, 10000, 10000)
        data = get_one_stack_vol(data, pair_tag1, brain_tag1, LOC1, f'{LOC1}', size)
        data = get_one_stack_vol(data, pair_tag2, brain_tag2, LOC2, f'{LOC2}', size)
    else:
        # LOC = (0.5, 0.3, 0.3)
        size = (4, 10000, 10000)
        LOC = (np.random.randint(1,1000)/1200, 0.5, 0.5)
        data = get_one_stack_vol(data, pair_tag1, brain_tag1, LOC, f'{LOC}', size)
        data = get_one_stack_vol(data, pair_tag2, brain_tag2, LOC, f'{LOC}', size)
    if not_plot: 
        return data
    data = pd.DataFrame(data)
    if len(data['vol']) == 0: return
    colors = [(255, 0, 0), (0, 119, 0)]
    # colors = [[c/255 for c in color] for color in colors]
    colors = {d: [c/255 for c in color] for d, color in zip(data['brain_tag'].unique(), colors)}
    # print(data)
    ##### Hist plot
    plt.figure(figsize=(5,4))
    ax = sns.histplot(data, x='vol', hue='brain_tag', kde=True, multiple="dodge", shrink=.8, linewidth=0, bins=32, palette=colors, alpha=0.4,
                  line_kws={'lw': 1.5, 'ls': '--'}, kde_kws={'gridsize': 5000})
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="y",direction="in")
    ax.tick_params(axis="x",direction="in")
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim([0, 25])
    try:
        ax.get_legend().remove()
    except:
        pass
    plt.tight_layout()
    plt.savefig(f'stats/plots/{pair_tag1}-{pair_tag2}-{brain_tag1}-{brain_tag2}_hist.svg')
    plt.savefig(f'stats/plots/{pair_tag1}-{pair_tag2}-{brain_tag1}-{brain_tag2}_hist.png')
    plt.close()
    ###########################
    ##### violin plot
    plt.figure(figsize=(1,2.5))
    ax = sns.violinplot(data, y='vol', x='brain_tag', palette=colors, linewidth=0)
    plt.setp(ax.collections, alpha=.7)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(f'stats/plots/{pair_tag1}-{pair_tag2}-{brain_tag1}-{brain_tag2}_viol.svg')
    plt.savefig(f'stats/plots/{pair_tag1}-{pair_tag2}-{brain_tag1}-{brain_tag2}_viol.png')
    plt.close()
    ###########################
    ##### kde plot
    sns.kdeplot(data, x='vol', hue='brain_tag', palette=colors, legend=False, ax=kdeax, linestyle='--', linewidth=1.5, gridsize=5000)

def get_pair_brain(gender_filter):
    pair_tags = [p for p in os.listdir(STAT_ROOT) if p.startswith('pair')]
    brain_label = pd.read_csv('downloads/brain_gene_label.csv')
    brain_label = brain_label[[os.path.exists(f"{STAT_ROOT}/pair{p}/{b}_nis_center.zip") for p, b in zip(brain_label['Pair'], brain_label['Brain'])]]
    valid_brain = None
    collapsed_brain = ['L57D855P1', 'L77D764P8', 'L59D878P1']
    for collapsed in collapsed_brain:
        if valid_brain is None:
            valid_brain = brain_label['Brain'] != collapsed
        else:
            valid_brain &= brain_label['Brain'] != collapsed
    brain_label = brain_label[valid_brain]
    brains = pd.concat([brain_label[brain_label['Pair'] == int(pair_tag.replace('pair', ''))] for pair_tag in pair_tags])
    if gender_filter != 'All':
        pair_tags = list(brains[brains['Gender']==gender_filter]['Pair'])
        brain_tags = list(brains[brains['Gender']==gender_filter]['Brain'])
    else:
        pair_tags = list(brains['Pair'])
        brain_tags = list(brains['Brain'])
    return pair_tags, brain_tags

if __name__=="__main__":
    filter1 = 'All'
    filter2 = 'All'
    pair_tags1, brain_tags1 = get_pair_brain(filter1)
    pair_tags2, brain_tags2 = get_pair_brain(filter2)
    nrows = len(pair_tags1)
    ncols = len(pair_tags2)
    # fig, kdeaxes = plt.subplots(nrows, ncols, figsize=(nrows, ncols))
    # for i in range(nrows):
    #     for j in range(ncols):
    #         kdeaxes[i,j].set_xticklabels([])
    #         kdeaxes[i,j].set_yticklabels([])
    #         kdeaxes[i,j].set_xticks([])
    #         kdeaxes[i,j].set_yticks([])
    #         kdeaxes[i,j].set_xlabel('')
    #         kdeaxes[i,j].set_ylabel('')
    #         if j<i: kdeaxes[i,j].axis('off')
    data = {'vol': [], 'density': [], 'brain_tag': [], 'brain_tag_density': []}
    for i in trange(len(pair_tags1)):
        pair_tag1 = f'pair{pair_tags1[i]}'
        brain_tag1 = brain_tags1[i]
        for j in trange(i, len(pair_tags2)):
            pair_tag2 = f'pair{pair_tags2[j]}'
            brain_tag2 = brain_tags2[j]
            # main(pair_tag1, brain_tag1, pair_tag2, brain_tag2, kdeaxes[i,j])
            data = main(pair_tag1, brain_tag1, pair_tag2, brain_tag2, None, True, data)

    # fig.savefig(f'stats/plots/{filter1}-{filter2}_brain_kdeplot.png')
    # fig.savefig(f'stats/plots/{filter1}-{filter2}_brain_kdeplot.svg')
    # plt.close(fig)
    pd.DataFrame({k:data[k] for k in ['vol', 'brain_tag']}).to_pickle(f'stats/{filter1}-{filter2}_brain_random_stack_nisvol.pkl')
    pd.DataFrame({k:data[k] for k in ['density', 'brain_tag_density']}).to_pickle(f'stats/{filter1}-{filter2}_brain_random_stack_density.pkl')