import h5py, os, json
import nibabel as nib
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm, trange
from multiprocessing import Pool

def find_label(ds, i, tgt, loworhigh=0):
    x = ds[i, 3]
    condition = x < tgt if loworhigh == 0 else x > tgt
    if condition:
        i = i * 2 if loworhigh == 0 else i / 2
        i = find_label(ds, i, tgt, loworhigh)
    return i


def main():
    std_resolution = (2.5, .75, .75)
    in_resolution = (4, .75, .75)
    scale_r = [i/s for i, s in zip(in_resolution, std_resolution)]
    vol_scale = scale_r[0]*scale_r[1]*scale_r[2]
    _r = '/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4'
    mask_r = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/registered/%s_MASK_topro_25_all.nii'
    save_roi_id = 16001
    data_list = []
    for r, d, fs in os.walk(_r):
        data_list.extend([os.path.join(r, f) for f in fs if f.endswith('_remap.json') and 'pair' in r])
        # if len(fs) > 0: break
    out = []
    for remap_fn in data_list:
        
        # try:
        # if 'pair3' not in remap_fn and 'pair20' not in remap_fn: continue
        # if 'pair14' not in remap_fn or 'L73D766P5' not in remap_fn: continue
        if 'L79D769P8' not in remap_fn: continue
        brain_tag = os.path.dirname(remap_fn).split('/')[-1]
        pair_tag = os.path.dirname(os.path.dirname(remap_fn)).split('/')[-1]
        print(datetime.now(), "Statistic brain", pair_tag, brain_tag)
        mask_fn = mask_r % (pair_tag, brain_tag, brain_tag)
        nis_fn = remap_fn.replace('_remap.json', '_NIS_results.h5')
        assert os.path.exists(nis_fn), f'{brain_tag} has not complete NIS'
        assert os.path.exists(mask_fn), f'{brain_tag} has no RoI mask'
        orig_roi_mask = torch.from_numpy(np.transpose(nib.load(mask_fn).get_fdata(), (2, 0, 1)).copy()).to('cuda:1') #[:, :, ::-1]
        # print(orig_roi_mask.shape)
        # exit()
        roi_mask = orig_roi_mask.clone()
        assert (roi_mask==orig_roi_mask).all()
        roi_remap = {}
        for newi, i in enumerate(roi_mask.unique()):
            if i == save_roi_id: 
                remap_save_roi_id = newi
            roi_remap[newi] = i
            roi_mask[roi_mask == i] = newi
        roi_mask = roi_mask.long()
        nis = h5py.File(nis_fn, 'r')
        brain_shape = nis['nuclei_segmentation'].shape
        nis_labels = nis['instance_label'][:]
        nis_centers = nis['instance_center'][:]
        print(datetime.now(),"Loaded centers", nis_centers.dtype)
        nis_volumes = nis['instance_volume'][:]
        # nis_volumes = nis['coordinate'][:, 3]
        nis_labels = torch.from_numpy(nis_labels).to('cuda:1')
        nis_volumes = torch.from_numpy(nis_volumes).to('cuda:1')#.bincount()[1:]
        # torch.save(nis_volumes, os.path.join(_r, 'NIS_VOL_%s_%s.pth' % (pair_tag, brain_tag)))
        # print(datetime.now(),"Saved volumes, max volume=", nis_volumes.max())
        nis_centers = torch.from_numpy(nis_centers).to('cuda:1')
        print(datetime.now(), "Total nuclei number", nis_labels.shape, nis_centers.dtype)
        with open(remap_fn, 'r') as jsonf:
            remaps = json.load(jsonf)
        # remap_dict = {}
        replicated = torch.zeros_like(nis_labels, dtype=bool)
        target = torch.zeros_like(nis_labels, dtype=bool)
        print(datetime.now(), f"Remove {sum([len(remap['remap'].keys()) for remap in remaps])} replicated nuclei") 
        for ri, remap in enumerate(remaps):
            # if ri == 2: break
            dic = remap['remap']
            with Pool(8) as p:
                keys = list(p.map(int, dic.keys()))
            keys = torch.LongTensor(keys).to(nis_labels.device)
            values = torch.LongTensor(list(dic.values())).to(nis_labels.device)
            interval = 250
            for di in trange(0, len(keys), interval):
                cur_replicated = (nis_labels == keys[di:di+interval, None]).any(0)
                cur_target = (nis_labels == values[di:di+interval, None]).any(0)
                replicated = replicated | cur_replicated
                target = target | cur_target
            # for oldid in tqdm(dic):
            #     # replicated = replicated & (nis_labels == oldid)
            #     # target = target & (nis_labels == dic[oldid])
            #     replicated[nis_labels == oldid] = True
            #     target[nis_labels == dic[oldid]] = True
        nis_centers[target] = (nis_centers[target] + nis_centers[replicated]) / 2
        nis_labels = nis_labels[~replicated]
        nis_volumes = nis_volumes[~replicated]
        nis_centers = nis_centers[~replicated]
        ratio = [b/r for b, r in zip(brain_shape, roi_mask.shape)]
        center_pt = nis_centers.clone()
        center_pt[:, 0] = center_pt[:, 0] / brain_shape[0]
        center_pt[:, 1] = center_pt[:, 1] / brain_shape[1]
        center_pt[:, 2] = center_pt[:, 2] / brain_shape[2]
        nis_centers[:, 0] = nis_centers[:, 0] / ratio[0]
        nis_centers[:, 1] = nis_centers[:, 1] / ratio[1]
        nis_centers[:, 2] = nis_centers[:, 2] / ratio[2]
        # print(datetime.now(), f'\n RoI mask shape: {roi_mask.shape}, \n Nuclei center maximum location: {nis_centers.max(0)[0]}')
        nis_centers[:, 0] = torch.clip(nis_centers[:, 0], min=0, max=roi_mask.shape[0]-0.1)
        nis_centers[:, 1] = torch.clip(nis_centers[:, 1], min=0, max=roi_mask.shape[1]-0.1)
        nis_centers[:, 2] = torch.clip(nis_centers[:, 2], min=0, max=roi_mask.shape[2]-0.1)

        roi_label = roi_mask[tuple(nis_centers.long().T)]
        roi_id = roi_label.unique()
        center_loc = roi_label==remap_save_roi_id
        # if 0 in roi_id:
        is_fg = roi_label!=0
        center_loc = is_fg & center_loc
            # roi_label = roi_label[is_fg]
        # print('Len', len(torch.where(center_loc)))
        # print('Len', center_pt.shape)
        # exit()
        print(datetime.now(), f'Statistic {sum(is_fg)} nuclei in {len(roi_id)-1} different regions')
        bincount = roi_label[is_fg].bincount()
        # assert bincount.sum() == len(roi_label), f'bincount error, where bincount.sum()={bincount.sum()} and len(roi_label)={len(roi_label)}'
        o = {
            'brain': brain_tag,
            'pair': pair_tag,
            'roi_count': {},
            'roi_avg_volume': {},
        }
        for i, c in enumerate(bincount):
            if i == 0 or c == 0: continue
            assert i in roi_id
            org_i = roi_remap[i]
            o['roi_count'][org_i.item()] = c.item()
            cur_loc = roi_label == i
            cur_loc = is_fg & cur_loc
            vols = nis_volumes[cur_loc] / vol_scale
            # cur_label = nis_labels[cur_loc]
            # labelmin = cur_label.min()
            # lowi = find_label(nis['coordinate'], labelmin, labelmin, loworhigh=0)
            # highi = find_label(nis['coordinate'], nis_labels.max() - cur_label.max(), nis_labels.max() - cur_label.max(), loworhigh=1)
            # vols = (torch.from_numpy(nis['coordinate'][lowi:highi, 3]) - labelmin).bincount()
            # vols = vols[vols!=0] / vol_scale
            # print(datetime.now(),"Loaded coordinates")
            o['roi_avg_volume'][org_i.item()] = vols.mean().item()
            if remap_save_roi_id == i:
                save_vols = vols.float()
        
        out.append(o)
        center_pt = center_pt[center_loc, :]
        center_pt_head = 'center X,center Y,center Z,Volume\n'
        center_pt = torch.cat([center_pt, save_vols.unsqueeze(1)], -1)
        center_csv = str(center_pt.tolist())
        center_csv = center_csv[2:-2].replace('], [','\n')
        with open(os.path.join(_r, 'NIS_%s_%s_RoI%d.csv' % (pair_tag, brain_tag, save_roi_id)), 'w') as f:
            f.write(center_pt_head + center_csv)
        # except:
        #     continue
    exit()
    csv = 'pair ID,brain ID,'
    roi_ids = []
    for o in out:
        roi_ids += list(o['roi_count'].keys())
    roi_ids = torch.LongTensor(roi_ids).unique()
    for i in roi_ids:
        csv += 'count.ROI_%d,volume.avg.ROI_%d,' % (i, i)
    csv = csv[:-1] + '\n'
    for o in out:
        line = '%s,%s,' % (o['pair'], o['brain'])
        for i in roi_ids:
            i = i.item()
            if i in o['roi_count']:
                line += '%d,%f,' % (o['roi_count'][i], o['roi_avg_volume'][i])
            else:
                line += '-,-,'
        line = line[:-1] + '\n'
        csv += line
    
    with open(os.path.join(_r, 'statistics.csv'), 'w') as f:
        f.write(csv)

if __name__ == '__main__': main()