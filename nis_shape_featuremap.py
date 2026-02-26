import torch, os
# import h5py
import numpy as np
import SimpleITK as sitk
from datetime import datetime
from tqdm import trange, tqdm
import random

gtag = 'P4'
STAT_ROOT = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/{gtag}_morphology'
r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/{gtag}'
pbtag_ls = []
for ptag in os.listdir(r):
    for btag in os.listdir(f'{r}/{ptag}'):
        pbtag_ls.append([ptag, btag])
pbtag_ls = list(reversed(pbtag_ls))

def main():
    img_tags = ['C1_', 'C2_','C3_']
    pair_tag = 'pair21'
    brain_tag = 'L91D814P6'
    brainmap_pa(pair_tag, brain_tag)


def brainmap_pa(pair_tag, brain_tag):
    device = 'cpu'
    for ptag, btag in pbtag_ls[4:]:
        if btag.split('_')[1] == brain_tag: break
    print(ptag, btag, brain_tag)
    # print(datetime.now(), f"Loading {pair_tag} {brain_tag}")
    root, result_root, stack_names, tile_lt_loc, seg_shape, whole_brain_shape = brain_shape_tile_location(pair_tag, btag)
    print(result_root)

    centers = []
    features = []

    for ijkey in tile_lt_loc:
        i, j = ijkey.split('-')
        _i, _j = int(i), int(j)
        tx, ty = tile_lt_loc[ijkey]
        tx, ty = int(tx), int(ty)
        
        pafn = f"{result_root % (_i, _j)}/{btag.split('_')[1]}_pa.zip"
        data = torch.load(pafn)
        pa = data['PA']
        center = data['center']
        center[:, 1] = center[:, 1] + tx
        center[:, 2] = center[:, 2] + ty

        all_pa_list = []
        nis_id_list = []
        center_list = []
        for stack_name in stack_names:
            # savefn = f"{result_root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_pa')}"
            if not os.path.exists(f"{result_root % (_i, _j)}/{stack_name}"): continue
            zstart = int(stack_name.split('zmin')[1].split('_')[0])
            zstart = int(zstart*zratio)
            
            centerfn = f"{result_root % (_i, _j)}/{stack_name}"
            center = torch.load(centerfn).long()#.to(device)
            center[:, 0] = center[:, 0] * zratio + zstart

            labelfn = f"{result_root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_label')}"
            label = torch.load(labelfn).long().to(device)

            volfn = f"{result_root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_volume')}"
            total_vol = torch.load(volfn).long().to(device)

            nisfn = f"{result_root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_coordinate')}"
            coordinate = torch.load(nisfn).long().to(device)
            coordinate[:, 0] = coordinate[:, 0] * zratio + zstart

            splits = total_vol.cumsum(0)
            for vol in tqdm(total_vol.unique(), desc='Looping unique volume'):
                if vol <= 300 or vol >= 800: continue
                nis_id = torch.where(total_vol==vol)[0]#.tolist()
                nis_label = label[nis_id]
                
                stitched_mask = []
                for lrangei in range(0, len(nis_label), lrange):
                    lo, _ = torch.where(nis_label[lrangei:lrangei+lrange, None] == zstitch_remap[0, None, :])
                    if len(lo) > 0:
                        stitched_mask.append(lo+lrangei)
                if len(stitched_mask) > 0:
                    stitched_mask = torch.cat(stitched_mask)
                    # print(stitched_mask.shape)
                    nis_id = nis_id[stitched_mask]
                
                nis_id = nis_id.tolist()
                nis_max_num = len(nis_id) // 100
                if nis_max_num == 0: continue
                random.shuffle(nis_id)
                nis_id = nis_id[:nis_max_num]
                nis_id.sort()
                nis_id_list.extend(label[nis_id].cpu().tolist())
                center_list.append(center[nis_id].cpu())
                coord_id = []
                for i in nis_id:
                    coord_id.append(torch.arange(splits[i]-total_vol[i], splits[i]))
                coord_id = torch.concat(coord_id)
                # print(datetime.now(), f"Start get coordinate {pair_tag} {brain_tag}")
                pts = coordinate[coord_id].reshape(len(nis_id), vol, 3)
                # print(datetime.now(), "Done get coordinate, Start build frames")
                ptmin = pts.min(1)[0] # N x 3
                pts = pts - ptmin.unsqueeze(1)
                ptmax = pts.max(1)[0] # N x 3
                ptmin = pts.min(1)[0] # N x 3
                assert (ptmin==0).all()
                ptmid = ptmax//2 # N x 3
                frame_whd = ptmax.max(0)[0] # 3
                frame_mid = (frame_whd//2).unsqueeze(0) # 1 x 3
                mid_remain = frame_mid - ptmid # N x 3
                assert (mid_remain>=0).all()
                pts = pts + mid_remain.unsqueeze(1) # N x vol x 3
                # print(frame_whd)
                frame_whd = frame_whd + 1
                frames = torch.zeros([len(nis_id)]+frame_whd.tolist(), dtype=bool)
                frame_id = torch.arange(len(nis_id)).unsqueeze(0).repeat(vol,1).T#.reshape(-1)
                frames[frame_id, pts[..., 0].long().cpu(), pts[..., 1].long().cpu(), pts[..., 2].long().cpu()] = True
                # frames = frames.long()
                # print(datetime.now(), "Done build frames, Start get principle axis")
                filter_label = sitk.LabelShapeStatisticsImageFilter()
                filter_label.SetComputeFeretDiameter(True)
                # pa_list = [] # principle axis list
                all_pa = []
                pts = pts - frame_mid.unsqueeze(0) # N x vol x 3

                # for i in trange(len(frames), desc='Get principle axises'):
                for i in range(len(frames)):
                    frame = frames[i].long()
                    filter_label.Execute(sitk.GetImageFromArray(frame.cpu().numpy()))
                    pa = torch.FloatTensor(filter_label.GetPrincipalAxes(1)).to(device) # 9
                    distances_pa1 = pts[i].float() @ pa[:3] # pa1: N
                    pa1_start = frame_mid.squeeze() + (distances_pa1.min() * pa[:3]) # 3
                    pa1_end = frame_mid.squeeze() + (distances_pa1.max() * pa[:3]) # 3
                    # pa1_vec = pa1_end - pa1_start # 3
                    # pa_list.append(pa1_vec)
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
                # pa_list = torch.stack(pa_list) # N x 3
                all_pa = torch.stack(all_pa) # N x 3 x 3
                # print(datetime.now(), "Done get principle axis", all_pa.shape)
                # frames = rotate_batch_tensor(frames, pa_list)
                # avg_frame = frames.mean(0)[0]
                # print(datetime.now(), "Done rotate frames, return avg", avg_frame.shape)
                all_pa_list.append(all_pa.cpu())
        print(datetime.now(), f"Saving {len(nis_id_list)} PA vectors")
        torch.save({'PA': torch.cat(all_pa_list),'selected_label': torch.LongTensor(nis_id_list),'center':torch.cat(center_list)}, savefn)
            

def brain_shape_tile_location(ptag, btag):
    if gtag == 'P4':
        overlap_r = 0.2
        root1 = '/cajal/Felix/Lightsheet/P4'
        root2 = '/lichtman/Felix/Lightsheet/P4'
        root = f'{root1}/{ptag}/{btag}' if os.path.exists(f'{root1}/{ptag}/{btag}') else f'{root2}/{ptag}/{btag}'
    else:
        overlap_r = 0.1
        root = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P14/{ptag}/{btag}'
    assert os.path.exists(root), root
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/{gtag}/{ptag}/{btag}'
    result_root = result_path + '/UltraII[%02d x %02d]'
    stack_names = [f for f in os.listdir(result_root % (0, 0)) if f.endswith('instance_center.zip')]
    stack_names = sort_stackname(stack_names)
    meta_name = stack_names[0].replace('instance_center', 'seg_meta')
    seg_shape = torch.load(f'{result_root % (0, 0)}/{meta_name}')
    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path) if 'UltraII' in fn])
    ncol, nrow = tile_loc.max(0)+1
    seg_shape = [s.item() for s in seg_shape]
    tile_lt_loc = {
        f'{i}-{j}': [i*seg_shape[1]*(1-overlap_r), j*seg_shape[2]*(1-overlap_r)] for i in range(ncol) for j in range(nrow)
    }
    whole_brain_shape = [seg_shape[0]] + list(np.array(list(tile_lt_loc.values())).max(0) + np.array(seg_shape[1:]))
    whole_brain_shape = [int(s) for s in whole_brain_shape]
    return root, result_root, stack_names, tile_lt_loc, seg_shape, whole_brain_shape

def sort_stackname(stack_names):
    stack_z = []
    for stack_name in stack_names:
        stack_z.append(int(stack_name.split('zmin')[1].split('_')[0]))

    argsort = np.argsort(stack_z)
    return [stack_names[i] for i in argsort]

if __name__=="__main__": main()