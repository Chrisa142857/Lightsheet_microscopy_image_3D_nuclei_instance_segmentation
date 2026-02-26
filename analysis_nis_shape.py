import argparse

import torch, os, sys
import numpy as np
import SimpleITK as sitk
from datetime import datetime
from tqdm import trange, tqdm
import random
# from multiprocessing import Pool

parser = argparse.ArgumentParser(description='None')
parser.add_argument('--gtag')
parser.add_argument('--ptag')
parser.add_argument('--btag')
# parser.add_argument('--ttag')
parser.add_argument('--device')
args = parser.parse_args()
pair_tag = args.ptag
brain_tag = args.btag
gtag = args.gtag
# tilekey = args.ttag
device = args.device

# STAT_ROOT = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/{gtag}_morphology'
r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/{gtag}'
pbtag_ls = []
for ptag in os.listdir(r):
    for btag in os.listdir(f'{r}/{ptag}'):
        pbtag_ls.append([ptag, btag])
pbtag_ls = list(reversed(pbtag_ls))

def main():
    get_pa(pair_tag, brain_tag)


def get_pa(pair_tag, brain_tag, tgt_ijkey=None):
    for ptag, btag in pbtag_ls[4:]:
        if btag.split('_')[1] == brain_tag: break
    assert ptag == pair_tag
    print(ptag, btag, brain_tag)
    # print(datetime.now(), f"Loading {pair_tag} {brain_tag}")
    root, result_root, stack_names, tile_lt_loc, seg_shape, whole_brain_shape = brain_shape_tile_location(pair_tag, btag)
    # print(result_root)
    # col_iterator = range(ncol)
    # row_iterator = range(nrow)
    # col_iterator = [2]
    # row_iterator = [1,2]
    # col_iterator = [2]
    # row_iterator = [0,4]
    # sample_num = 50
    # lrange = 1000
    zratio = 2.5/4
    # down_res = 25
    # down_r = torch.FloatTensor([4/down_res, .75/down_res, .75/down_res]).to(device)
    # zstitch_remap_dict = {}
    # for ijkey in tile_lt_loc:
    #     i, j = ijkey.split('-')
    #     _i, _j = int(i), int(j)
    #     zstitch_remap_dict[ijkey] = torch.load(f"{result_root % (_i, _j)}/{btag.split('_')[1]}_remap.zip").to(device)
            
    for ijkey in tile_lt_loc:
        i, j = ijkey.split('-')
        _i, _j = int(i), int(j)
        if tgt_ijkey is not None and tgt_ijkey != ijkey: continue
        # savefn = f"{result_root % (_i, _j)}/{btag.split('_')[1]}_pa.zip"
        # tx, ty = tile_lt_loc[ijkey]
        # tx, ty = int(tx), int(ty)
        # stat_root = f"{STAT_ROOT}/{pair_tag}"
        print(datetime.now(), f'Processing tile {ijkey}')
        # zstitch_remap = zstitch_remap_dict[ijkey]
        # center_list = []
        pre_num = 0
        for stack_name in stack_names:
            if not os.path.exists(f"{result_root % (_i, _j)}/{stack_name}"): continue
            savefn = f"{result_root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_pa')}"
            savefn = savefn.replace('/cajal', '/scheibel')
            os.makedirs(os.path.dirname(savefn), exist_ok=True)
            zstart = int(stack_name.split('zmin')[1].split('_')[0])
            zstart = int(zstart*zratio)
            
            centerfn = f"{result_root % (_i, _j)}/{stack_name}"
            center = torch.load(centerfn).long().to(device)
            center[:, 0] = center[:, 0] * zratio + zstart

            labelfn = f"{result_root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_label')}"
            label = torch.load(labelfn).long().to(device)

            volfn = f"{result_root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_volume')}"
            total_vol = torch.load(volfn).long().to(device)

            nisfn = f"{result_root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_coordinate')}"
            coordinate = torch.load(nisfn).long()#.to(device)
            coordinate[:, 0] = coordinate[:, 0] * zratio + zstart

            splits = total_vol.cumsum(0)

            ## Random downsampling
            # down_center = (down_r[None] * center).long()
            # dshape = (down_center.max(0)[0]+1).tolist()
            # down_loc = torch.arange(dshape[0]*dshape[1]*dshape[2]).view(dshape[0], dshape[1], dshape[2]).to(device) 
            # all_loc = down_loc[down_center[:, 0], down_center[:, 1], down_center[:, 2]]
            # nis_id_loc = []
            # for loc in all_loc.unique():
            #     loc_id = torch.where(all_loc==loc)[0].tolist()
            #     random.shuffle(loc_id)
            #     nis_id_loc.extend(loc_id[:min(len(loc_id), 5)])
            # nis_uni_loc_mask = torch.zeros_like(total_vol).bool()
            # nis_uni_loc_mask[nis_id_loc] = True
            ############
            # print(datetime.now(), f"Processing {len(nis_id_loc)} / {len(total_vol)} NIS")
            all_pa_list = []
            nis_id_list = []
            print(datetime.now(), f"Processing {len(total_vol)} NIS")
            for vol in tqdm(total_vol.unique(), desc=f'Looping unique volume in tile {ijkey}'):
                if vol < 30: continue
                vol_mask = total_vol==vol
                # nis_id = torch.where(torch.logical_and(vol_mask, nis_uni_loc_mask))[0]
                nis_id = torch.where(vol_mask)[0]
                nis_id_list.append(label[nis_id])
                # nis_id_list.extend((nis_id+pre_num).tolist())
                # center_list.append(center[nis_id].cpu())
                #####################################################
                if len(nis_id) == 0: continue
                coord_id = []
                for i in nis_id:
                    coord_id.append(torch.arange(splits[i]-total_vol[i], splits[i]))
                coord_id = torch.concat(coord_id)
                # print(datetime.now(), f"Start get coordinate {pair_tag} {brain_tag}")
                pts = coordinate[coord_id].reshape(len(nis_id), vol, 3).to(device)
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
                # filter_label = sitk.LabelShapeStatisticsImageFilter()
                # filter_label.SetComputeFeretDiameter(True)
                # pa_list = [] # principle axis list
                all_pa = []
                filter_label = sitk.LabelShapeStatisticsImageFilter()
                filter_label.SetComputeFeretDiameter(True)
                pts = pts - frame_mid.unsqueeze(0) # N x vol x 3

                # for i in trange(len(frames), desc='Get principle axises'):
                # with Pool(processes=30) as itk_pa_pool:
                #     pa_list = list(itk_pa_pool.imap(itk_pa, frames))

                for i in range(len(frames)):
                    frame = frames[i].long()
                    filter_label.Execute(sitk.GetImageFromArray(frame.cpu().numpy()))
                    pa = torch.FloatTensor(filter_label.GetPrincipalAxes(1)).to(device) # 9
                # for i, pa in enumerate(pa_list):
                #     pa = pa.to(device)
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
            # pre_num = len(total_vol)
            nis_id_list = torch.cat(nis_id_list)
            all_pa_list = torch.cat(all_pa_list)
            print(datetime.now(), f"Saving {len(nis_id_list)} PA vectors")
            torch.save({'PA': all_pa_list,'NIS_label': nis_id_list}, savefn)
        # torch.save(torch.cat(all_pa_list), savefn)

def itk_pa(frame):
    filter_label = sitk.LabelShapeStatisticsImageFilter()
    filter_label.SetComputeFeretDiameter(True)
    frame = frame.long()
    filter_label.Execute(sitk.GetImageFromArray(frame.cpu().numpy()))
    pa = torch.FloatTensor(filter_label.GetPrincipalAxes(1)) # 9      
    return pa

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
    # stack_names = [f for f in os.listdir(result_root % (0, 0)) if f.endswith('instance_center.zip')]
    
    
    stack_names = None
    for tile_fn in os.listdir(result_path):
        if not tile_fn.startswith('UltraII'): continue
        if stack_names is None:
            stack_names = [f for f in os.listdir(f'{result_path}/{tile_fn}') if f.endswith('instance_center.zip')]
        
        pstack_names = [f for f in os.listdir(f'{result_path}/{tile_fn}') if f.endswith('instance_center.zip')]
        if len(pstack_names) > len(stack_names): stack_names = pstack_names
        
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

#     pair_tag = 'pair6'
#     brain_tag = 'L57D855P6'
#     data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)
#     # TODO: NIS used data in lichtman, but should to use cajal
#     pair_tag = 'pair6'
#     brain_tag = 'L57D855P2'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)


#     pair_tag = 'pair5'
#     brain_tag = 'L57D855P4'
#     data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)
#     pair_tag = 'pair5'
#     brain_tag = 'L57D855P5'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)

# #########
#     pair_tag = 'pair19'
#     brain_tag = 'L79D769P8'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)
#     pair_tag = 'pair19'
#     brain_tag = 'L79D769P5'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)


#     pair_tag = 'pair17'
#     brain_tag = 'L77D764P2'
#     data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)

#     pair_tag = 'pair17'
#     brain_tag = 'L77D764P9'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)


#     pair_tag = 'pair13'
#     brain_tag = 'L69D764P6'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)
#     pair_tag = 'pair13'
#     brain_tag = 'L69D764P9'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)

#     pair_tag = 'pair15'
#     brain_tag = 'L73D766P4'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)
#     pair_tag = 'pair15'
#     brain_tag = 'L73D766P9'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)


#     pair_tag = 'pair10'
#     brain_tag = 'L64D804P3'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)
#     pair_tag = 'pair10'
#     brain_tag = 'L64D804P9'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)

#     pair_tag = 'pair11'
#     brain_tag = 'L66D764P3'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)
#     pair_tag = 'pair11'
#     brain_tag = 'L66D764P8'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)

#     pair_tag = 'pair12'
#     brain_tag = 'L66D764P5'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)
#     pair_tag = 'pair12'
#     brain_tag = 'L66D764P6'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)
# ##########
#     pair_tag = 'pair14'
#     brain_tag = 'L73D766P5'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)
#     pair_tag = 'pair14'
#     brain_tag = 'L73D766P7'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)

#     pair_tag = 'pair16'
#     brain_tag = 'L74D769P4'
#     data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
#     main(data_root, pair_tag, brain_tag)
    
    # pair_tag = 'pair16'
    # brain_tag = 'L74D769P8'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)

    # pair_tag = 'pair18'
    # brain_tag = 'L77D764P4'
    # data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)

    # pair_tag = 'pair20'
    # brain_tag = 'L79D769P7'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)
    # pair_tag = 'pair20'
    # brain_tag = 'L79D769P9'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)

    # pair_tag = 'pair21'
    # brain_tag = 'L91D814P2'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)
    # pair_tag = 'pair21'
    # brain_tag = 'L91D814P6'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)

    # pair_tag = 'pair22'
    # brain_tag = 'L91D814P3'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)
    # pair_tag = 'pair22'
    # brain_tag = 'L91D814P4'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)

    # pair_tag = 'pair3'
    # brain_tag = 'L35D719P1'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)
    # pair_tag = 'pair3'
    # brain_tag = 'L35D719P4'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)


    # pair_tag = 'pair8'
    # brain_tag = 'L59D878P2'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)
    # pair_tag = 'pair8'
    # brain_tag = 'L59D878P5'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)


    # pair_tag = 'pair9'
    # brain_tag = 'L64D804P4'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)
    # pair_tag = 'pair9'
    # brain_tag = 'L64D804P6'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)

    # pair_tag = 'pair4'
    # brain_tag = 'L35D719P3'
    # data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)
    # pair_tag = 'pair4'
    # brain_tag = 'L35D719P5'
    # data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)

    # pair_tag = 'pair6'
    # brain_tag = 'L57D855P1'
    # data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)

    # pair_tag = 'pair18'
    # brain_tag = 'L77D764P8'
    # data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    # main(data_root, pair_tag, brain_tag)