import numpy as np
import tifffile as tif
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import models
import torch, sys, os, imageio
from test_tss import test_onestack
from two_slice_stitch import StitchModel
from utils import iou_2mask, mask2mask_list, eval_f1
_edge_remain = 8
_gap_step = 8

def main(gap_types):
    eval_device = 0
    tag = 'tss_stitch_P4'
    os.makedirs('results3D_gap_%s/%s' % (str(gap_types), tag), exist_ok=True)
    
    # trained_model = 'cellpose/models/cellpose_residual_on_style_on_concatenation_off_train_2023_02_28_09_28_46.761192_epoch_41'
    trained_model = 'cellpose/models/P4_models/cellpose_residual_on_style_on_concatenation_off_train_2023_04_10_21_02_12.036323_epoch_41'
    # model = models.CellposeModel(device=torch.device('cuda:%d' % int(sys.argv[3])), pretrained_model=trained_model)
    model = models.CellposeModel(device=torch.device('cuda:1'), pretrained_model=trained_model)

    graph_model = StitchModel('cuda:%d'%eval_device)    
    graph_model.to('cuda:%d'%eval_device)
    graph_model.load_state_dict(torch.load('tss_trainv2_output_filter_pyramid/epoch188.pth'))
    import matplotlib.pyplot as plt
    with open('test_p4_list.txt', 'r') as f:
        testlist = f.read().split('\n')[:-1]
    dset = ImgSpitByGapDataset(testlist, gap_types)
    dloader = DataLoader(dset, batch_size=2, num_workers=8, collate_fn=passCollateFn)
    all_cropped_part_nums = []
    precs, recs, f1s = [], [], []
    whole_precs, whole_recs, whole_f1s = [], [], []
    gtcrop_percent = []
    for di, data in tqdm(enumerate(dloader), total=len(dloader)):
        for cropped_img, gt, cropped_gt, cropped_place, imgn in data:
            inputs = [img for img in cropped_img if 0 not in img.shape]
            outputs = [None if 0 not in img.shape else np.zeros(1) for img in cropped_img]
            cropped_gt = [gt if 0 not in gt.shape else np.zeros(1) for gt in cropped_gt]
            masks, _, _ = model.eval(inputs, diameter=None, channels=[0,0], do_3D=True, stitch_threshold=0, do_sim_Z=True)
            assert sum([1 for m in masks if 0 in m.shape]) == 0, "No. %d, Error" % di
            mi = 0
            for i in range(len(outputs)):
                if outputs[i] is None: 
                    outputs[i] = masks[mi]
                    mi += 1
            whole_stack_mask = stitch_from_gap(outputs, gap_types, tss_in_gap, images=inputs, model=graph_model)[0]
            tif.imwrite('results3D_gap_%s/%s/%s_%s' % (str(gap_types), tag, 'cropped-in-'+str(cropped_place), imgn), whole_stack_mask)
            prec, rec, f1, _, _, _ = eval_f1(torch.from_numpy(mask2mask_list(whole_stack_mask)).to('cuda:%d'%eval_device), torch.from_numpy(mask2mask_list(gt)).to('cuda:%d'%eval_device))
            whole_precs.append(prec)
            whole_recs.append(rec)
            whole_f1s.append(f1)
            whole_stack_mask = stitch_from_gap(outputs, gap_types)[0]
            prec, rec, f1, _, _, _ = eval_f1(torch.from_numpy(mask2mask_list(whole_stack_mask)).to('cuda:%d'%eval_device), torch.from_numpy(mask2mask_list(gt)).to('cuda:%d'%eval_device))
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
            # for cgt, mask in zip(cropped_gt, outputs):
            #     if cgt.max() == 0: continue
            #     if mask.max() == 0 : 
            #         prec, rec, f1 = 0, 0, 0
            #     else:    
            #         prec, rec, f1, _, _, _ = eval_f1(torch.from_numpy(mask2mask_list(mask)).to('cuda:%d'%eval_device), torch.from_numpy(mask2mask_list(cgt)).to('cuda:%d'%eval_device))
            #     precs.append(prec)
            #     recs.append(rec)
            #     f1s.append(f1)
            # cropped_inds, cropped_part_nums = get_cropped_gt_ind(cropped_gt)
            # gtcrop_percent.append(len(cropped_inds) / len(np.unique(gt)))

        # all_cropped_part_nums.extend(cropped_part_nums)
    # plt.hist(all_cropped_part_nums, bins=100)
    # plt.savefig('hist_cropped_part_nums.jpg')
    print("F1: %.6f, Prec: %.6f, Rec: %.6f (nonstitch stacks), cropped percent: %.6f" % (np.mean(f1s), np.mean(precs), np.mean(recs), np.mean(gtcrop_percent)))
    print("F1: %.6f, Prec: %.6f, Rec: %.6f (%s stacks), cropped percent: %.6f" % (np.mean(whole_f1s), np.mean(whole_precs), np.mean(whole_recs), tag, np.mean(gtcrop_percent)))
    # plt.show()
    # plt.close()


def split_img_by_gaps(imgs, gap_places, shape, gap_types, split_times=1, overlap=0):
    if len(gap_types) == 0: 
        return imgs
    out_imgs = []
    for img in imgs:
        cur_shape = img.shape[gap_types[0]]
        gap_place = min(cur_shape-int(_edge_remain/2**split_times), gap_places[0])
        out_imgs.append(split_img_by_gaps([
            np.take(img, np.arange(cur_shape)[:gap_place+overlap], axis=gap_types[0]),
            np.take(img, np.arange(cur_shape)[gap_place-overlap:], axis=gap_types[0])
        ], gap_places[1:], shape[1:], gap_types[1:], split_times+1, overlap=overlap))

    return out_imgs


def reorganize_img_stack(imgs):
    out_list = []
    for img in imgs:
        if isinstance(img, list): 
            out_list.extend(reorganize_img_stack(img))
        else:
            out_list.append(img)

    return out_list


class ImgSpitByGapDataset(Dataset):
    
    def __init__(self, fnlist, gap_types, loop_num=10) -> None:
        '''
            gap type means creating gap along with an axis: 0 or 1 or 2
            ## gap direction of cropped img stacks
            if gap_types = [2, 1, 0], then

            [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]

            [1,       1], [1,       1], [1,       1], [1,       1]

            [2,                                                 2]
        '''
        super().__init__()
        self.img = []
        self.gt = []
        self.cropped_gt = []
        self.cropped_i = []
        self.imgn = []
        self.shape = []
        self.loop_num = loop_num
        gaps = []
        for g in gap_types:
            pre_gap_num = sum([gap==g for gap in gaps])
            self.shape.append(int(tif.imread(fnlist[0]).shape[g] / (2**pre_gap_num)))
            gaps.append(g)
        self.gap_types = gap_types
        self.fnlist = fnlist

    def __getitem__(self, index):
        gt_fn = self.fnlist[index % len(self.fnlist)]
        img_fn = gt_fn.replace('_masks', '').replace('masks', 'images')
        # img = tif.imread(img_fn) * 1.0
        img = np.stack(imageio.v2.mimread(img_fn)) * 1.0
        gt = tif.imread(gt_fn) * 1.0
        gap_place = [np.random.randint(low=_edge_remain, high=s-_edge_remain) for s in self.shape]
        cropped_img_stacks = split_img_by_gaps([img], gap_place, self.shape, self.gap_types)[0]
        cropped_img_stacks = reorganize_img_stack(cropped_img_stacks)
        cropped_gt_stacks = split_img_by_gaps([gt], gap_place, self.shape, self.gap_types)[0]
        cropped_gt_stacks = reorganize_img_stack(cropped_gt_stacks)
        assert len(cropped_img_stacks) == 2 ** (len(self.gap_types))
        return cropped_img_stacks, gt, cropped_gt_stacks, gap_place, img_fn.split('/')[-1]


    def __len__(self):
        return len(self.fnlist)*self.loop_num

def tss_in_gap(pre_stack, next_stack, axis, model, pre_img, next_img):
    device = 'cuda:0'
    pre_mask = np.take(pre_stack, -1, axis=axis)
    next_mask = np.take(next_stack, 0, axis=axis)
    masks = np.stack([pre_mask, next_mask])
    pre_slice = np.take(pre_img, -1, axis=axis)
    next_slice = np.take(next_img, 0, axis=axis)
    imgs = np.stack([pre_slice, next_slice])
    masks, remaps = test_onestack(masks, imgs, model, device)
    for mid in np.unique(next_mask):
        if mid == 0: continue
        if mid not in remaps[0]: continue
        next_stack[next_stack==mid] = remaps[0][mid]
    return np.concatenate([pre_stack, next_stack], axis=axis)    

def nostitch_in_gap(pre_stack, next_stack, axis, model=None, pre_img=None, next_img=None):
    return np.concatenate([pre_stack, next_stack], axis=axis)    

def stitch_from_gap(masks, gap_types, stitcher=nostitch_in_gap, images=None, model=None):
    out = []
    for i in range(0, len(masks), 2):
        next_stack = masks[i+1]
        next_stack[next_stack != 0] += masks[i].max()
        if images is not None:
            next_img = images[i+1]
        out.append(stitcher(masks[i], next_stack, gap_types[-1], model=model, pre_img=images[i] if images is not None else None, next_img=next_img if images is not None else None))
    if len(out) < 2:
        return out
    else:
        return stitch_from_gap(out, gap_types[:-1], stitcher=stitcher, images=images, model=model)

def get_cropped_gt_ind(cropped_gt):
    gt_inds = []
    for gt in cropped_gt:
        gt_inds.append(np.unique(gt))
    cropped_inds = []
    cropped_part_nums = []
    for i in range(len(gt_inds)):
        for j in gt_inds[i]:
            if j in cropped_inds: continue
            cropped_num = sum([1 for k in range(len(gt_inds)) if k != i and j in gt_inds[k]])
            if cropped_num > 0: 
                cropped_inds.append(j)
                cropped_part_nums.append(cropped_num+1)
    return cropped_inds, cropped_part_nums
       
def passCollateFn(data):
    return data

if __name__ == "__main__":
    main(gap_types=[0,0])
    # main(gap_types=[2, 2])