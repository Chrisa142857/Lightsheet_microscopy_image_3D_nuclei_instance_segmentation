import torch, os
from tqdm import trange
from datetime import datetime

res_r = '/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4'
for pair_tag in os.listdir(res_r)[:8]:
    if not pair_tag.startswith('pair'): continue
    for brain_tag in os.listdir(f'{res_r}/{pair_tag}'):
        for seg_name in os.listdir(f'{res_r}/{pair_tag}/{brain_tag}'):
            if not seg_name.endswith('_seg.zip'): continue
            seg_path = f'{res_r}/{pair_tag}/{brain_tag}/{seg_name}'
            # seg_path='/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/pair10/L64D804P3/L64D804P3_NIScpp_results_zmin718_seg.zip'
            label_path = seg_path.replace('seg.zip', 'instance_label.zip')
            vol_path = seg_path.replace('seg.zip', 'instance_volume.zip')
            # ct_path = seg_path.replace('seg.zip', 'instance_center.zip')
            # pt_path = seg_path.replace('seg.zip', 'contour.zip')
            seg=torch.load(seg_path)
            # vol = torch.load(vol_path)
            # label = torch.load(label_path)
            # ct = torch.load(ct_path)
            # pt = torch.load(pt_path)
            # vols = []
            # instance_masks = []
            # chunk_shape = [20, 60, 60]
            # half_c = [int(s/2) for s in chunk_shape]
            print(f'{datetime.now()}: On {seg_name}')
            # indecies = [[],[],[]]
            # labels = []
            # splits = []
            seg_bincount = seg[seg>0].reshape(-1).bincount()
            print(datetime.now(), 'Geting vols', seg_bincount.shape)
            vols = seg_bincount
            print(datetime.now(), 'Saving vols', vols.shape)
            torch.save(vols, vol_path)
            label = seg[seg>0].unique()
            torch.save(label, label_path)
            # instance_masks = torch.cat(instance_masks)#.bool()
            # torch.save(instance_masks, seg_path.replace('seg.zip', 'instance_mask.zip'))
            # print(vols)