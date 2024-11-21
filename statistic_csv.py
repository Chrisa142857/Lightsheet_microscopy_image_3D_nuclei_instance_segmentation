import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json, torch, scipy, random, nrrd


def main(data_csv, label_csv, datatag):
    label_ids, label_names, region_level, collect_id = get_filtered_label_id('Isocortex')
    i = 0
    for label_id, label_name, rlvl in zip(label_ids, label_names, region_level):
        if i != 0: collect_id=None
        barplot(data_csv, label_csv, datatag+f"_L{rlvl}", label_id, label_name, collect_id=collect_id)
        i+=1

def collect_region(data, collect_id, datatag):
    output = {'a': ['' for _ in range(len(data['brain_tag']))], 'brain_tag': list(data['brain_tag'])}
    for out_rid in collect_id:
        out = []
        for org_rid in collect_id[out_rid]:
            # if '-' in list(data[f'ROI{org_rid}']) or 0 in list(data[f'ROI{org_rid}']): continue
            # out.append(list(data[f'ROI{org_rid}']))
        # axis = 0
        # if len(out) == 0: continue
        # if datatag in ['volavg', 'density']:
        #     out = np.array(out).mean(axis)
        # else:
        #     out = np.array(out).sum(axis)
            out.append([float(d) if d != '-' and d != 0 else '-' for d in data[f'ROI{org_rid}']])
        if datatag in ['volavg', 'density']:
            out = [np.mean([o[i] for o in out if o[i]!='-']) for i in range(len(data['brain_tag']))]
        else:
            out = [np.sum([o[i] for o in out if o[i]!='-']) for i in range(len(data['brain_tag']))]
        
        output[f'ROI{out_rid}'] = list(out)
    return pd.DataFrame(output)

def barplot(data_csv, label_csv, datatag, ridlist, namelist, collect_id=None, xkey='ROI ID', huekeys=['Gene', 'Gender']):
    data = pd.read_csv(data_csv)
    if collect_id is not None: 
        data = collect_region(data, collect_id, datatag)
    label = pd.read_csv(label_csv)
    male_brain = label['Brain'][label['Gender']=='Male']
    female_brain = label['Brain'][label['Gender']=='Female']
    wt_brain = label['Brain'][label['Gene']=='WT']
    het_brain = label['Brain'][label['Gene']=='HET']
    gender_label = []
    gene_label = []
    atlas_list = []
    pair_list = []
    for bi, b in enumerate(data['brain_tag']):
        pair_list.append(label['Pair'][label['Brain']==b].item())
        if len(np.where(male_brain==b)[0]) == 1: 
            gender_label.append('Male')
        elif len(np.where(female_brain==b)[0]) == 1: 
            gender_label.append('Female')
        else:
            gender_label.append('Female')
        if len(np.where(wt_brain==b)[0]) == 1: 
            gene_label.append('WT')
        elif len(np.where(het_brain==b)[0]) == 1: 
            gene_label.append('HET')
        else:
            gene_label.append('WT')
        if 'density' in datatag:
            atlas_list.append(load_atlas(b).to('cuda:1'))
    
    Male_n = len([1 for l in gender_label if l == 'Male'])
    Female_n = len([1 for l in gender_label if l == 'Female'])
    WT_n = len([1 for l in gene_label if l == 'WT'])
    HET_n = len([1 for l in gene_label if l == 'HET'])
    N = {
        'Male' : Male_n,
        'Female' : Female_n,
        'WT' : WT_n,
        'HET' : HET_n
    }
    d = {"Gender": [], "Gene": [], datatag: [], "ROI name": [], "ROI ID": [], "pair": []}
    outlier_tags = {}
    for c in list(data.columns.values)[2:]:
        rid = int(c.replace('ROI', ''))
        x = data[c]
        valid_id = [i for i in range(len(x)) if ['L77D764P2', 'L77D764P4']]
        cur_gene_label = gene_label
        cur_gender_label = gender_label 
        cur_pair_list = pair_list
        if '-' in list(x) or 0 in list(x): 
            cur_gene_label = [gene_label[i] for i in range(len(x)) if x[i] != '-' and x[i] != 0]
            cur_gender_label = [gender_label[i] for i in range(len(x)) if x[i] != '-' and x[i] != 0]
            cur_pair_list = [pair_list[i] for i in range(len(x)) if x[i] != '-' and x[i] != 0]
            valid_id = [i for i in range(len(x)) if x[i] != '-' and x[i] != 0]
            x = [xi for xi in list(x) if xi != '-' and xi != 0]
        x = np.array([float(xi) for xi in x])
        # if 'density' in datatag:
        #     vox_num = np.array([len(torch.where(atlas_list[i] == rid)[0]) for i in valid_id])
        #     x = x/vox_num
        # remove_outlier = np.abs(scipy.stats.zscore(x)) < 1
        f1 = x > 1
        f2 = np.abs(scipy.stats.zscore(x)) < 3
        remove_outlier = f1 & f2
        cur_gene_label = [cur_gene_label[ri] for ri, r in enumerate(remove_outlier) if r]
        cur_gender_label = [cur_gender_label[ri] for ri, r in enumerate(remove_outlier) if r]
        cur_pair_list = [cur_pair_list[ri] for ri, r in enumerate(remove_outlier) if r]
        outlier_tags[rid] = [data['brain_tag'][valid_id[ri]] for ri, r in enumerate(remove_outlier) if not r]
        valid_id = [valid_id[ri] for ri, r in enumerate(remove_outlier) if r]
        x = x[remove_outlier]
        if rid not in ridlist: continue    
        d["Gene"] += cur_gene_label
        d["Gender"] += cur_gender_label
        d[datatag].append(x)
        rname = namelist[ridlist.index(rid)]
        d["ROI name"] += [rname for _ in range(len(x))]
        d["ROI ID"] += [rid for _ in range(len(x))]
        d["pair"] += cur_pair_list
        N[rid] = len(x)
        
    d[datatag] = np.concatenate(d[datatag])
    d = pd.DataFrame(d)
    print(d)
    cnts = dict(d[xkey].value_counts())
    key = list(cnts.keys())
    hue_pair = {
        'Gene': ['WT', 'HET'],
        'Gender': ['Male', 'Female']
    }

    for huekey in huekeys:
        plt.figure(figsize=(max(0.25*len(key), 10), 10))
        g = sns.boxplot(data=d, hue=huekey, y=datatag, x=xkey, width=.5, orient="v", order=key)
        xticks = []
        sign_keys = []
        sign_p = []
        sign_alist = []
        sign_blist = []
        for i in range(len(key)):
            a = list(d.loc[(d[huekey]==hue_pair[huekey][0]) & (d["ROI ID"]==key[i]), datatag])
            b = list(d.loc[(d[huekey]==hue_pair[huekey][1]) & (d["ROI ID"]==key[i]), datatag])
            if huekey == 'Gene':
                a_pairtag = list(d.loc[(d[huekey]==hue_pair[huekey][0]) & (d["ROI ID"]==key[i]), 'pair'])
                b_pairtag = list(d.loc[(d[huekey]==hue_pair[huekey][1]) & (d["ROI ID"]==key[i]), 'pair'])
                _, a_ind, b_ind = np.intersect1d(a_pairtag, b_pairtag, return_indices=True)
                a = [a[i] for i in a_ind]
                b = [b[i] for i in b_ind]
            else:
                minlen = min(len(a), len(b))
                random.shuffle(a)
                random.shuffle(b)
                a = a[:minlen]
                b = b[:minlen]
            p = ttest_pvalue(a, b)
            if p <= 0.05:
                sign_alist.append(a)
                sign_blist.append(b)
                sign_keys.append(key[i])
                sign_p.append(p)
            
            xticks.append(f"{namelist[ridlist.index(key[i])]}(n={len(a)*2},p={p:.3f})")
        # xticks = [f"{namelist[ridlist.index(key[i])]}(n={N[key[i]]},p={ttest_out.pvalue:.3f})" for i in range(len(key))]
        g.set_xticklabels(xticks, rotation = 90)#
        plt.tight_layout()
        plt.savefig(f'{save_r}/hist_roi_{datatag}_{huekey}.png')
        plt.close()

        if len(sign_keys) > 0:
            plt.figure(figsize=(max(0.25*len(sign_keys), 5), 5))
            g = sns.boxplot(data=d, hue=huekey, y=datatag, x=xkey, width=.5, orient="v", order=sign_keys)
            xticks = []
            for i in range(len(sign_keys)):
                print("outlier_tags", set(outlier_tags[sign_keys[i]]))
                a = sign_alist[i]
                b = sign_blist[i]
                # a = list(d.loc[(d[huekey]==hue_pair[huekey][0]) & (d["ROI ID"]==sign_keys[i]), datatag])
                # b = list(d.loc[(d[huekey]==hue_pair[huekey][1]) & (d["ROI ID"]==sign_keys[i]), datatag])
                xticks.append(f"{namelist[ridlist.index(sign_keys[i])]}(n={len(a)*2},p={sign_p[i]:.3f})")
            g.set_xticklabels(xticks, rotation = 90 if huekey != 'Gene' else 30)#
            plt.tight_layout()
            plt.savefig(f'{save_r}/hist_roi_{datatag}_{huekey}_p0.05.png')
            plt.close()

    # plt.figure(figsize=(15, 10))
    # # g = sns.boxplot(data=d, x="Gender", y=datatag, hue="ROI id", width=.5)
    # g = sns.boxplot(data=d, hue="Gender", y=datatag, x="ROI id", width=.5)
    # cnts = dict(d['Gender'].value_counts())
    # key = list(cnts.keys())
    # # g.set_xticklabels([f"{key[i]}\n(n={N[key[i]]})" for i in range(len(key))])
    # plt.savefig(f'hist_roi_{datatag}_gender.png')
    # # print(x)
    # # print(label)
    # # return list(set(d["ROI name"]))

def load_atlas(brain_tag):
    ann_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/register_to_rotation_corrected/allen_atlas/{brain_tag}_annotation_resampled.nrrd'
    ann = torch.from_numpy(nrrd.read(ann_path)[0].astype(np.int32))
    ann[ann==1000] = 0
    return ann

def ttest_pvalue(a, b):
    # ttest_out = scipy.stats.ttest_ind(a[:minlen], b[:minlen], random_state=142857)#, nan_policy='omit'
    ttest_out = scipy.stats.ttest_rel(a, b)#, nan_policy='omit'
    return ttest_out.pvalue

def get_filtered_label_id(filtern=None):
    org_labeln = pd.read_csv('downloads/allen_atlas_ccfv3_label_table.csv')
    labeln = org_labeln.copy()
    remap = json.load(open('downloads/allen_atlas_ccfv3_id_remap.json', 'r'))
    remap = {v: k for k, v in remap.items()}
    for i in range(len(labeln['structure ID'])):
        if labeln['structure ID'][i] not in remap:
            labeln.at[i, 'structure ID'] = -1
        else:
            labeln.at[i, 'structure ID'] = int(remap[labeln['structure ID'][i]])
    if filtern is not None:
        rowi = list(labeln['abbreviation']).index(filtern)
        filter_rlvl = labeln['depth in tree'][rowi]
        collect_rlvl = filter_rlvl+1
        sid_path = labeln['structure_id_path'][rowi]
        filetered_label_id = [labeln['structure ID'][i] for i in range(len(labeln['structure_id_path'])) if sid_path in labeln['structure_id_path'][i] and labeln['structure ID'][i] != -1]
        filetered_label_name = [labeln['abbreviation'][i] for i in range(len(labeln['structure_id_path'])) if sid_path in labeln['structure_id_path'][i] and labeln['structure ID'][i] != -1]
        filetered_sid_path = [labeln['structure_id_path'][i] for i in range(len(labeln['structure_id_path'])) if sid_path in labeln['structure_id_path'][i] and labeln['structure ID'][i] != -1]
        region_level = [labeln['depth in tree'][i] for i in range(len(labeln['structure_id_path'])) if sid_path in labeln['structure_id_path'][i] and labeln['structure ID'][i] != -1]
        collect_remap = {
            org_labeln['structure ID'][i]: [filetered_label_id[j] for j in range(len(filetered_label_id)) if labeln['structure_id_path'][i] in filetered_sid_path[j]]
        for i in range(len(labeln['structure_id_path'])) if sid_path in labeln['structure_id_path'][i] and labeln['depth in tree'][i] == collect_rlvl}
        label_ids = [[org_labeln['structure ID'][i] for i in range(len(labeln['structure_id_path'])) if sid_path in labeln['structure_id_path'][i] and labeln['depth in tree'][i] == collect_rlvl]]
        label_names = [[org_labeln['abbreviation'][i] for i in range(len(labeln['structure_id_path'])) if sid_path in labeln['structure_id_path'][i] and labeln['depth in tree'][i] == collect_rlvl]]
    else:
        filetered_label_id = list(labeln['structure ID'])
        filetered_label_name = list(labeln['abbreviation'])
        region_level = list(labeln['depth in tree'])
        label_ids = []
        label_names = []
        collect_remap = None
    region_level = torch.LongTensor(region_level)
    for rlvl in region_level.unique():
        loc = torch.where(region_level==rlvl)[0].tolist()
        label_ids.append([filetered_label_id[i] for i in loc] + [filetered_label_id[i]+1000 for i in loc])
        label_names.append([filetered_label_name[i] for i in loc] + [filetered_label_name[i] for i in loc])
    region_level = region_level.unique().tolist()
    if collect_remap is not None: region_level = [collect_rlvl] + region_level
    print(f"Do statistic under Region levels {region_level}")
    return label_ids, label_names, region_level, collect_remap


if __name__ == "__main__":
    global save_r
    save_r = 'stats/remove_zero'
    stat_tag = '30brain'
    # for datatag in ['density']:
    for datatag in ['density', 'cell-counting', 'volavg']:
        labelp = "downloads/brain_gene_label.csv"
        datap = f"downloads/statistic_{stat_tag}_{datatag}.csv"
        ridlist = main(datap, labelp, f"{stat_tag}_{datatag}_hueROI")