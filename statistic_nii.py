import nibabel as nib
import matplotlib.pyplot as plt
import torch, os
import nrrd
import numpy as np
from tqdm import tqdm

def main(pair_tag, brain_tag, rid_list=[], datatag='density', img_tag='_C2', dataroot='cajal'):
    '''
    output
        "{pair_tag},{brain_tag},[{rid-cell-couting}...]\n"
    '''
    datatag_n = 'density' if datatag in ['density', 'cell-counting'] else datatag
    map_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/register_to_rotation_corrected/{datatag_n}_map/{brain_tag}.nii'
    ann_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/register_to_rotation_corrected/allen_atlas/{brain_tag}_annotation_resampled.nrrd'
    # map_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/renders/{pair_tag}/NIS_density{img_tag}_{pair_tag}_{brain_tag}.nii'
    # ann_path = f'/{dataroot}/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/registered/{brain_tag}_MASK_topro_25_all.nii'
    rep = torch.from_numpy(nib.load(map_path).get_fdata())
    
    if datatag in ['density', 'cell-counting']:
        rep = torch.round(rep)    
    elif datatag == 'volavg':
        seg_res = [2.5, 0.75, 0.75] 
        unit = 1
        for sr in seg_res:
            unit *= sr
        rep = rep * unit
    # ann = torch.from_numpy(nib.load(ann_path).get_fdata())
    # ann = ann.permute(2, 0, 1)
    ann = torch.from_numpy(nrrd.read(ann_path)[0].astype(np.int32))
    ann[ann==1000] = 0
    # print(ann.shape, rep.shape)
    # plt.matshow(rep[100])
    # plt.savefig('temp1.png')
    # plt.close()
    # plt.matshow(ann[100])
    # plt.savefig('temp2.png')
    # exit()
    mshape = [min(s1, s2) for s1, s2 in zip(rep.shape, ann.shape)]
    rep = rep[:mshape[0],:mshape[1],:mshape[2]].cuda()
    ann = ann[:mshape[0],:mshape[1],:mshape[2]].cuda()
    out = [pair_tag,brain_tag]
    out += ['-' for _ in rid_list]
    for rid in ann.unique():
        if rid == 0: continue
        if rid not in rid_list:
            rid_list.append(rid)
            out.append('-')
        loc = ann==rid
        if datatag == 'cell-counting':
            out[rid_list.index(rid)+2] = f'{int(rep[loc].sum().item())}'
        elif datatag == 'density':
            out[rid_list.index(rid)+2] = f'{int(rep[loc].sum().item())/len(torch.where(loc)[0])}'
        elif datatag == 'volavg':
            out[rid_list.index(rid)+2] = f'{rep[loc].mean().item()}'
    return ','.join(out)+'\n', rid_list

if __name__ == "__main__":
    for datatag in ['cell-counting', 'density', 'volavg']:
        rid_list = []
        datatag_n = 'density' if datatag in ['density', 'cell-counting'] else datatag
        registered_dir = f'/cajal/ACMUSERS/ziquanw/Lightsheet/register_to_rotation_corrected/{datatag_n}_map'
        csv_body = ''
        for brain_tag in tqdm(os.listdir(registered_dir)):
            brain_tag = brain_tag[:-4]
            out, rid_list = main('', brain_tag, rid_list, datatag=datatag)
            csv_body += out
        print(f"Total {len(rid_list)} regions")
        csv_head = 'pair_tag,brain_tag'
        for rid in rid_list:
            csv_head += f',ROI{rid}'
        csv = csv_head + '\n' + csv_body
        with open(f"downloads/statistic_{len(os.listdir(registered_dir))}brain_{datatag}.csv", 'w') as f:
            f.write(csv)
    

    # pair_range = [6, 5, 17, 13, 15, 10, 19]
    # roots = ['cajal', 'cajal', 'cajal', 'lichtman', 'lichtman', 'lichtman', 'lichtman']
    # # pair_range = [6]
    # img_tags = ['', '_C2', '_C3']
    # for img_tag in img_tags:
    #     csv_body = ''
    #     for dataroot, pn in zip(roots, pair_range):
    #         btags = os.listdir(f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/pair{pn}')
    #         assert len(btags) == 2, btags
    #         out, rid_list = main(f"pair{pn}", btags[0], img_tag=img_tag, dataroot=dataroot)
    #         csv_body += out
    #         out, rid_list = main(f"pair{pn}", btags[1], rid_list=rid_list, img_tag=img_tag, dataroot=dataroot)
    #         csv_body += out

    #     csv_head = 'pair_tag,brain_tag'
    #     for rid in rid_list:
    #         csv_head += f',ROI{rid}-cell-counting'
    #     csv = csv_head + '\n' + csv_body
    #     with open(f"downloads/statistic_pair{min(pair_range)}-{max(pair_range)}{img_tag}_counting.csv", 'w') as f:
    #         f.write(csv)
    