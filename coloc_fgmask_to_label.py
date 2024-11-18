import torch, os, json, copy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from datetime import datetime
from torch_scatter import scatter_sum
import pandas as pd

zratio = 2.5/4

def main():
    device = 'cuda:2'
    fgmask_to_label(ptag='pair4', btag='220904_L35D719P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_21-49-38', device=device)
    # exit()
    ptag='pair20'
    btag='220912_L79D769P7_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_11-27-41'
    print("=== fgmask_to_label ", ptag, btag.split("_")[1], "===")
    fgmask_to_label(ptag, btag, device)
    ptag='pair20'
    btag='220917_L79D769P9_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-06-03'
    print("=== fgmask_to_label ", ptag, btag.split("_")[1], "===")
    fgmask_to_label(ptag, btag, device)
    ptag='pair4'
    btag='220902_L35D719P3_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-41-36'
    fgmask_to_label(ptag, btag, device)
    ptag='pair21'
    btag='220923_L91D814P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-24-18'
    print("=== fgmask_to_label ", ptag, btag.split("_")[1], "===")
    fgmask_to_label(ptag, btag, device)
    ptag='pair21'
    btag='220927_L91D814P6_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-26-45'
    print("=== fgmask_to_label ", ptag, btag.split("_")[1], "===")
    fgmask_to_label(ptag, btag, device)
    ptag='pair10'
    btag='220624_L64D804P3_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_16-44-01'
    print("=== fgmask_to_label ", ptag, btag.split("_")[1], "===")
    fgmask_to_label(ptag, btag, device)
    ptag='pair10'
    btag='220625_L64D804P9_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_10-50-29'
    print("=== fgmask_to_label ", ptag, btag.split("_")[1], "===")
    fgmask_to_label(ptag, btag, device)
    ptag='pair15'
    btag='220729_L73D766P4_topro_brn2_ctip2_4x_50sw_11hdf_0_108na_4z_20ov_17-07-38'
    print("=== fgmask_to_label ", ptag, btag.split("_")[1], "===")
    fgmask_to_label(ptag, btag, device)
    ptag='pair15'
    btag='220730_L73D766P9_topro_brn2_ctip2_4x_50sw_11hdf_0_108na_4z_20ov_10-58-21'
    print("=== fgmask_to_label ", ptag, btag.split("_")[1], "===")
    fgmask_to_label(ptag, btag, device)
    ptag='pair16'
    btag='220805_L74D769P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_15-51-36'
    print("=== fgmask_to_label ", ptag, btag.split("_")[1], "===")
    fgmask_to_label(ptag, btag, device)
    ptag='pair16'
    btag='220807_L74D769P8_OUT_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_11-51-47'
    fgmask_to_label(ptag, btag, device)
    ptag='pair16'
    btag='220807_L74D769P8_OUT_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_11-51-47'
    print("=== fgmask_to_label ", ptag, btag.split("_")[1], "===")
    fgmask_to_label(ptag, btag, device)
    ptag='pair17'
    btag='220819_L77D764P2_OUT_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_16-54-05'
    print("=== fgmask_to_label ", ptag, btag.split("_")[1], "===")
    fgmask_to_label(ptag, btag, device)
    ptag='pair17'
    btag='220820_L77D764P9_OUT_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_17-00-41'
    print("=== fgmask_to_label ", ptag, btag.split("_")[1], "===")
    fgmask_to_label(ptag, btag, device)

def fgmask_to_label(ptag, btag, device):
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}/{btag}'
    root = result_path + '/UltraII[%02d x %02d]'
    fgmask_root = f'/cajal/ACMUSERS/ziquanw/Lightsheet/colocalization/P4/{ptag}/{btag.split("_")[1]}' + '/UltraII[%02d x %02d]'
    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path)])
    ncol, nrow = tile_loc.max(0)+1
    stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    stack_names = sort_stackname(stack_names)
    
    nis_c1_count = []
    nis_c2_count = []
    tile_splits = []
    lrange = 100000
    for i in range(ncol):
        for j in range(nrow):
            k = f'{i}-{j}'
            # print(datetime.now(), k)
            i, j = k.split('-')
            i, j = int(i), int(j)
            if not os.path.exists(f"{fgmask_root % (i, j)}/{btag.split('_')[1]}_C00_fgmask.nii.gz"): 
                print("No fg mask", f"{fgmask_root % (i, j)}/{btag.split('_')[1]}_C00_fgmask.nii.gz")
                continue
            nis_c1_sum_ls = []
            nis_c2_sum_ls = []
            stack_splits = []
            vols = []
            coords = []
            for stack_name in stack_names:
                if not os.path.exists(f"{root % (i, j)}/{stack_name}"): continue
                zstart = int(stack_name.split('zmin')[1].split('_')[0])
                zstart = int(zstart*zratio)
                coordfn = f"{root % (i, j)}/{stack_name.replace('instance_center', 'instance_coordinate')}"
                coloc_labelfn = f"{root % (i, j)}/{stack_name.replace('instance_center', 'instance_coloc')}"
                volfn = f"{root % (i, j)}/{stack_name.replace('instance_center', 'instance_volume')}"
                # if not os.path.exists(coloc_labelfn) and os.path.exists(coordfn):
                if os.path.exists(coordfn):
                    print(f'{datetime.now()}: Load NIS of tile [{i}, {j}] {stack_name}')
                    vol = torch.load(volfn).long().to(device)
                    coord = torch.load(coordfn)#.to(device)

                    # if (vol == 0).any():
                    #     print('Exist NIS has zero volume')
                    #     valid_nis = vol!=0
                    #     vol_cumsum = vol.cumsum(0)
                    #     vol_cumsum = [0] + vol_cumsum.tolist()
                    #     valid_nis = torch.where(valid_nis)[0]
                    #     valid_coord = []
                    #     for i in valid_nis:
                    #         valid_coord.append(torch.arange(vol_cumsum[i], vol_cumsum[i+1]))
                    #         assert valid_coord[-1].shape[0] == vol[i], f'{i}: {valid_coord[-1].shape[0]} != {vol[i]}, {vol_cumsum[i-1]}, {vol_cumsum[i]}, {vol_cumsum[i+1]}'

                    #     valid_coord = torch.cat(valid_coord)
                    #     vol = vol[valid_nis]
                    #     coord = coord[valid_coord]

                    coord[:, 0] = coord[:, 0]*zratio + zstart
                    coord = torch.round(coord).long()
                    vols.append(vol)
                    coords.append(coord)

            if len(vols)>0:
                print(f'{datetime.now()}: Get coloc fg label of NIS of tile [{i}, {j}]')
                c1_fgmask = nib.load(f"{fgmask_root % (i, j)}/{btag.split('_')[1]}_C00_fgmask.nii.gz").get_fdata()
                c2_fgmask = nib.load(f"{fgmask_root % (i, j)}/{btag.split('_')[1]}_C02_fgmask.nii.gz").get_fdata()
                c1_fgmask = (torch.from_numpy(c1_fgmask) > 0)#.to(device)
                c2_fgmask = (torch.from_numpy(c2_fgmask) > 0)#.to(device)
                for vol, coord, stack_name in zip(vols, coords, stack_names):
                    print(f'{datetime.now()}: Count channel fg of NIS of tile [{i}, {j}] {stack_name}')
                    label = torch.arange(len(vol)).to(device)
                    label = torch.repeat_interleave(label, vol)
                    # for i in range(0, len(coord), lrange):
                    nis_c1_mask = c1_fgmask[coord[:, 0], coord[:, 1], coord[:, 2]].to(device)
                    nis_c1_sum_ls.append(scatter_sum(nis_c1_mask.float(), label).cpu().long())
                    nis_c2_mask = c2_fgmask[coord[:, 0], coord[:, 1], coord[:, 2]].to(device)
                    nis_c2_sum_ls.append(scatter_sum(nis_c2_mask.float(), label).cpu().long())
                    stack_splits.append(len(vol))
                nis_c1_sum_ls = torch.cat(nis_c1_sum_ls)
                nis_c2_sum_ls = torch.cat(nis_c2_sum_ls)
                data = pd.DataFrame({
                    'NIS fg-count': nis_c1_sum_ls.tolist() + nis_c2_sum_ls.tolist(),
                    'Channel': np.concatenate([np.repeat(np.array(['C00']), len(nis_c1_sum_ls)), np.repeat(np.array(['C02']), len(nis_c2_sum_ls))])
                })
                ax = sns.displot(data=data, x='NIS fg-count', hue='Channel', stat="probability", log_scale=False)
                for a in ax.axes.flat:
                    a.set_yscale("log")
                plt.tight_layout()
                plt.savefig(f'{btag.split("_")[1]}_tile{i}-{j}.png')
                plt.close()

    #         nis_c1_count.append(nis_c1_sum_ls)
    #         nis_c2_count.append(nis_c2_sum_ls)
    #         tile_splits.append(stack_splits)
    # nis_c1_count = torch.cat(nis_c1_count)
    # nis_c2_count = torch.cat(nis_c2_count)

            
                
def sort_stackname(stack_names):
    stack_z = []
    for stack_name in stack_names:
        stack_z.append(int(stack_name.split('zmin')[1].split('_')[0]))

    argsort = np.argsort(stack_z)
    return [stack_names[i] for i in argsort]

if __name__ == '__main__': main()