import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

gtag = 'P14'
btag = 'L106P3'
path1 = '/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/P14/female/L106P3/NIS_tranform/L106P3_tform_coarse.json'
path2 = '/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/P14/female/L106P3/NIS_tranform/L106P3_tform_refine.json'
seg_shape = torch.load('/cajal/ACMUSERS/ziquanw/Lightsheet/results/P14/female/231028_L106P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_14-48-05/UltraII[00 x 02]/L106P3_NIScpp_results_zmin958_seg_meta.zip')

# path1 = '/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/pair21/L91D814P6/NIS_tranform/L91D814P6_tform_coarse.json'
# path2 = '/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/pair21/L91D814P6/NIS_tranform/L91D814P6_tform_refine.json'
# seg_shape = torch.load('/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/pair21/220927_L91D814P6_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-26-45/UltraII[03 x 01]/L91D814P6_NIScpp_results_zmin958_seg_meta.zip')

with open(path1, 'r') as jfile:
    coarse_tform = json.load(jfile)

print(coarse_tform)

with open(path2, 'r') as jfile:
    refine_tform = json.load(jfile)

fig, axes = plt.subplots(5, 4, figsize=(20,12.5), sharex=True, sharey=True)
tform_xy_max = [0.05*seg_shape[1], 0.05*seg_shape[2]]
pre_refine_lt = None
for k in refine_tform[0]:
    data = {'transform': [], 'dim': [], 'slice': []}
    for zi in range(len(refine_tform)):
        
        _tx, _ty = refine_tform[zi][k]
        if (abs(_tx) > tform_xy_max[0] or abs(_ty) > tform_xy_max[1]):
            if pre_refine_lt is not None:
                _tx, _ty = pre_refine_lt
            else:
                for zii in range(zi, len(refine_tform)):
                    if k not in refine_tform[zii]: continue
                    _tx, _ty = refine_tform[zii][k]
                    if abs(_tx) <= refine_tform[0] and abs(_ty) <= refine_tform[1]:
                        break
        else:
            _tx, _ty = refine_tform[zi][k]
        data['transform'].append(_tx)
        data['dim'].append('X')
        data['slice'].append(zi+1)
        data['transform'].append(_ty)
        data['dim'].append('Y')
        data['slice'].append(zi+1)
        pre_refine_lt = [_tx, _ty]
    data = pd.DataFrame(data)    
    i, j = k.split('-')
    i, j = int(i), int(j)
    print(k)
    print(data)
    # plt.figure(figsize=(5,2.5))
    sns.lineplot(data=data, x='slice', y='transform', hue='dim', ax=axes[j, i])
plt.tight_layout()
plt.savefig(f'refine_tform_{gtag}_{btag}.svg')
plt.savefig(f'refine_tform_{gtag}_{btag}.png')
plt.close()
        
