import h5py, os, json
import nibabel as nib
import numpy as np
import torch
from datetime import datetime


def main():
    _r = '/lichtman/ziquanw/Lightsheet/results/P4'
    mask_r = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/registered/%s_MASK_topro_25_all.nii'
    data_list = []
    for r, d, fs in os.walk(_r):
        data_list.extend([os.path.join(r, f) for f in fs if f.endswith('_remap.json')])
        # if len(fs) > 0: break
    out = []
    for remap_fn in data_list:
        brain_tag = os.path.dirname(remap_fn).split('/')[-1]
        pair_tag = os.path.dirname(os.path.dirname(remap_fn)).split('/')[-1]
        out.append({
            'brain': brain_tag,
            'pair': pair_tag,
            'roi_count': {}
        })
        print(datetime.now(), "Statistic brain", pair_tag, brain_tag)
        mask_fn = mask_r % (pair_tag, brain_tag, brain_tag)
        nis_fn = remap_fn.replace('_remap.json', '_NIS_results.h5')
        assert os.path.exists(nis_fn), f'{brain_tag} has not complete NIS'
        assert os.path.exists(mask_fn), f'{brain_tag} has no RoI mask'
        roi_mask = torch.from_numpy(np.transpose(nib.load(mask_fn).get_fdata(), (2, 0, 1))[:, :, ::-1].copy()).cuda()
        nis = h5py.File(nis_fn, 'r')
        brain_shape = nis['nuclei_segmentation'].shape
        nis_labels = nis['instance_label'][:]
        nis_centers = nis['instance_center'][:]
        nis_labels = torch.from_numpy(nis_labels).cuda()
        nis_centers = torch.from_numpy(nis_centers).cuda()
        print(datetime.now(), "Total nuclei number", nis_labels.shape)
        with open(remap_fn, 'r') as jsonf:
            remaps = json.load(jsonf)
        remap_dict = {}
        replicated = torch.zeros_like(nis_labels, dtype=bool)
        print(datetime.now(), 'Remove replicated nuclei')
        for remap in remaps:
            dic = remap['remap']
            for oldid in dic:
                assert oldid not in remap_dict, remap['stitching_slices']
                remap_dict[oldid] = dic[oldid]
                replicated = replicated & (nis_labels == oldid)
        nis_labels = nis_labels[~replicated]
        nis_centers = nis_centers[~replicated]
        
        ratio = [b/r for b, r in zip(brain_shape, roi_mask.shape)]
        nis_centers[:, 0] = nis_centers[:, 0] / ratio[0]
        nis_centers[:, 1] = nis_centers[:, 1] / ratio[1]
        nis_centers[:, 2] = nis_centers[:, 2] / ratio[2]
        
        print(datetime.now(), f'\n RoI mask shape: {roi_mask.shape}, \n Nuclei center maximum location: {nis_centers.max(0)[0]}')
        nis_centers[:, 0] = torch.clip(nis_centers[:, 0], min=0, max=roi_mask.shape[0]-1)
        nis_centers[:, 1] = torch.clip(nis_centers[:, 1], min=0, max=roi_mask.shape[1]-1)
        nis_centers[:, 2] = torch.clip(nis_centers[:, 2], min=0, max=roi_mask.shape[2]-1)
        roi_label = roi_mask[tuple(nis_centers.long().T)]
        roi_id = roi_label.unique()
        print(datetime.now(), f'Statistic nuclei count from {len(roi_id)-1} regions')
        for i in roi_id:
            if i == 0: continue
            out[-1]['roi_count'][i] = sum(roi_label == i).item()
    
    csv = 'pair ID,brain ID,'
    roi_ids = torch.cat([o['roi_id'] for o in out]).unique()
    for i in roi_ids:
        csv += 'count.ROI_%d,'
    csv = csv[:-1] + '\n'
    for o in out:
        line = '%s,%s,' % (o['pair'], o['brain'])
        for i in roi_ids:
            line += '%d,' % o['roi_count'][i]
        line = line[:-1] + '\n'
        csv += line
    
    with open('statistics.csv', 'w') as f:
        f.write(csv)
    
if __name__ == '__main__': main()