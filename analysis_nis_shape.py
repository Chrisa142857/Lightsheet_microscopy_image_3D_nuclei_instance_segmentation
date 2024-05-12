import torch
import h5py
import numpy as np
import SimpleITK as sitk
from datetime import datetime
from tqdm import trange, tqdm
import random

STAT_ROOT = '/cajal/ACMUSERS/ziquanw/Lightsheet/statistics/P4'

def main(data_root, pair_tag, brain_tag):
    # print(datetime.now(), f"Loading {pair_tag} {brain_tag}")
    stat_root = f"{STAT_ROOT}/{pair_tag}"
    total_vol = torch.load(f"{stat_root}/{brain_tag}_nis_volume.zip", map_location='cpu')
    coordinate = h5py.File(f"{stat_root}/{brain_tag}_nis_coordinate.h5", 'r')['data']
    splits = total_vol.cumsum(0)
    all_pa_list = []
    nis_id_list = []
    for vol in tqdm(total_vol.unique(), desc='Looping unique volume'):
        nis_id = torch.where(total_vol==vol)[0].tolist()
        nis_max_num = len(nis_id) // 1000
        if nis_max_num == 0: continue
        random.shuffle(nis_id)
        nis_id = nis_id[:nis_max_num]
        nis_id.sort()
        nis_id_list.extend(nis_id)
        coord_id = []
        for i in nis_id:
            coord_id.append(np.arange(splits[i]-total_vol[i], splits[i]))
        coord_id = np.concatenate(coord_id)
        print(datetime.now(), f"Start get coordinate {pair_tag} {brain_tag}")
        pts = torch.from_numpy(coordinate[coord_id].astype(np.int16)).reshape(len(nis_id), vol, 3)
        print(datetime.now(), "Done get coordinate, Start build frames")
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
        frames[frame_id, pts[..., 0].long(), pts[..., 1].long(), pts[..., 2].long()] = True
        # frames = frames.long()
        print(datetime.now(), "Done build frames, Start get principle axis")
        filter_label = sitk.LabelShapeStatisticsImageFilter()
        filter_label.SetComputeFeretDiameter(True)
        # pa_list = [] # principle axis list
        all_pa = []
        pts = pts - frame_mid.unsqueeze(0) # N x vol x 3

        for i in trange(len(frames), desc='Get principle axises'):
            frame = frames[i].long()
            filter_label.Execute(sitk.GetImageFromArray(frame.numpy()))
            pa = torch.FloatTensor(filter_label.GetPrincipalAxes(1)) # 9
            distances_pa1 = pts[i].float() @ pa[:3] # pa1: N
            pa1_start = frame_mid.squeeze() + (distances_pa1.min() * pa[:3]) # 3
            pa1_end = frame_mid.squeeze() + (distances_pa1.max() * pa[:3]) # 3
            pa1_vec = pa1_end - pa1_start # 3
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
        print(datetime.now(), "Done get principle axis", all_pa.shape)
        # frames = rotate_batch_tensor(frames, pa_list)
        # avg_frame = frames.mean(0)[0]
        # print(datetime.now(), "Done rotate frames, return avg", avg_frame.shape)
        all_pa_list.append(all_pa)
    torch.save({'PA': torch.cat(all_pa_list),'NIS id': torch.LongTensor(nis_id_list)}, f"{stat_root}/{brain_tag}_nis_pa.zip")
    return all_pa


if __name__=="__main__":
    img_tags = ['C1_', 'C2_','C3_']

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
    
    pair_tag = 'pair16'
    brain_tag = 'L74D769P8'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)

    pair_tag = 'pair18'
    brain_tag = 'L77D764P4'
    data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)

    pair_tag = 'pair20'
    brain_tag = 'L79D769P7'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)
    pair_tag = 'pair20'
    brain_tag = 'L79D769P9'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)

    pair_tag = 'pair21'
    brain_tag = 'L91D814P2'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)
    pair_tag = 'pair21'
    brain_tag = 'L91D814P6'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)

    pair_tag = 'pair22'
    brain_tag = 'L91D814P3'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)
    pair_tag = 'pair22'
    brain_tag = 'L91D814P4'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)

    pair_tag = 'pair3'
    brain_tag = 'L35D719P1'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)
    pair_tag = 'pair3'
    brain_tag = 'L35D719P4'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)


    pair_tag = 'pair8'
    brain_tag = 'L59D878P2'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)
    pair_tag = 'pair8'
    brain_tag = 'L59D878P5'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)


    pair_tag = 'pair9'
    brain_tag = 'L64D804P4'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)
    pair_tag = 'pair9'
    brain_tag = 'L64D804P6'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)

    pair_tag = 'pair4'
    brain_tag = 'L35D719P3'
    data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)
    pair_tag = 'pair4'
    brain_tag = 'L35D719P5'
    data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)

    pair_tag = 'pair6'
    brain_tag = 'L57D855P1'
    data_root = f'/lichtman/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)

    pair_tag = 'pair18'
    brain_tag = 'L77D764P8'
    data_root = f'/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/stitched'
    main(data_root, pair_tag, brain_tag)