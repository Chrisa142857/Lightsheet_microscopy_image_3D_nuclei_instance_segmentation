import h5py, os, torch, json
import nibabel as nib
import numpy as np
import tifffile as tif
from run_stitch_step import sort_fs, get_rid_of_zoombie_process
import utils
from multiprocessing import Pool
from datetime import datetime

def main():
    tgt_roi = 16001
    tgt_shape = (32, 2500, 2500)
    tgt_chunk_num = 2 # chunk number of each brain
    ratio_2dto3d = [2.5/4, 1, 1]
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
        print(datetime.now(), "Start crop chunks of brain", pair_tag, brain_tag)
        save_r = '%s/%s/%s' % (_r, pair_tag, brain_tag)
        # brain_flow_dir = '%s/%s/%s/flow_3d' % (_r, pair_tag, brain_tag)
        img_r = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/stitched' % (pair_tag, brain_tag)
        img_fnlist = sort_fs([fn for fn in os.listdir(img_r) if fn.endswith('.tif') and '_C1_' in fn], get_i_xy, firsti=1)
        mask_fn = mask_r % (pair_tag, brain_tag, brain_tag)
        nis_fn = remap_fn.replace('_remap.json', '_NIS_results.h5')
        assert os.path.exists(nis_fn), f'{brain_tag} has not complete NIS'
        assert os.path.exists(mask_fn), f'{brain_tag} has no RoI mask'
        orig_roi_mask = torch.from_numpy(np.transpose(nib.load(mask_fn).get_fdata(), (2, 0, 1))[:, :, ::-1].copy())#.cuda()
        roi_mask = orig_roi_mask.clone()
        nis = h5py.File(nis_fn, 'r')
        with open(remap_fn, 'r') as jsonf:
            remaps = json.load(jsonf)
        remap_dict = {}
        print(datetime.now(), 'Collect stitching map dict')
        for remap in remaps:
            dic = remap['remap']
            for oldid in dic:
                assert oldid not in remap_dict, remap['stitching_slices']
                remap_dict[oldid] = dic[oldid]

        nis_seg = nis['nuclei_segmentation']
        brain_shape = nis_seg.shape
        ratio = [b/r for b, r in zip(brain_shape, roi_mask.shape)]
        tgt_index = torch.stack(torch.where(roi_mask==tgt_roi)).float()
        tgt_index[0] = tgt_index[0] * ratio[0]
        tgt_index[1] = tgt_index[1] * ratio[1]
        tgt_index[2] = tgt_index[2] * ratio[2]
        tgt_index[0] = torch.clip(tgt_index[0], min=0, max=brain_shape[0]-tgt_shape[0]-1)
        tgt_index[1] = torch.clip(tgt_index[1], min=0, max=brain_shape[1]-tgt_shape[1]-1)
        tgt_index[2] = torch.clip(tgt_index[2], min=0, max=brain_shape[2]-tgt_shape[2]-1)
        tgt_index = tgt_index.long().T
        cilist = torch.randint(low=0, high=len(tgt_index), size=(tgt_chunk_num,))
        for ci in cilist:
            zs, ze = tgt_index[ci][0], tgt_index[ci][0] + tgt_shape[0] + 1
            ys, ye = tgt_index[ci][1], tgt_index[ci][1] + tgt_shape[1] + 1
            xs, xe = tgt_index[ci][2], tgt_index[ci][2] + tgt_shape[2] + 1
            print(datetime.now(), 'Load segmentation, z%d-%d_y%d-%d_x%d-%d' % (zs,ze,ys,ye,xs,xe))
            chunk = nis_seg[zs:ze, ys:ye, xs:xe]
            for i in np.unique(chunk):
                if i == 0: continue
                if i in remap_dict:
                    chunk[chunk==i] = remap_dict[i]
            for newi, i in enumerate(np.unique(chunk)):
                if i == 0: continue
                chunk[chunk==i] = newi
            chunk_fnlist = []
            print(datetime.now(), 'Load image')
            for z in range(zs, ze):
                z = int(z * ratio_2dto3d[0])
                assert z < len(img_fnlist), "%d, %d" % (z, len(img_fnlist))
                img_fn = img_fnlist[z]
                chunk_fnlist.append(img_fn)
            loader_pool = Pool(processes=10)
            image = list(loader_pool.imap(load_img_chunk, [(os.path.join(img_r, img_fn), ys, ye, xs, xe) for img_fn in chunk_fnlist]))
            get_rid_of_zoombie_process(loader_pool)
            image = np.stack([np.asarray(img) for img in image])
            print(datetime.now(), 'Save chunks of segmentation and image')
            chunk = nib.Nifti1Image(chunk.astype(np.float32), np.eye(4))
            nib.save(chunk, '%s/seg_chunk_forvis_z%d-%d_y%d-%d_x%d-%d.nii' % (save_r,zs,ze,ys,ye,xs,xe))
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, '%s/img_chunk_forvis_z%d-%d_y%d-%d_x%d-%d.nii' % (save_r,zs,ze,ys,ye,xs,xe))

            # tif.imwrite('%s/seg_chunk_forvis_z%d-%d_y%d-%d_x%d-%d.tif' % (save_r,zs,ze,ys,ye,xs,xe), chunk)
            # tif.imwrite('%s/img_chunk_forvis_z%d-%d_y%d-%d_x%d-%d.tif' % (save_r,zs,ze,ys,ye,xs,xe), image)
        

def load_img_chunk(args):
    fn, ys, ye, xs, xe = args
    return utils.imread(fn)[ys:ye, xs:xe]
    

def get_i_xy(fn):
    return int(fn.split('_')[1])

if __name__ == "__main__": main()