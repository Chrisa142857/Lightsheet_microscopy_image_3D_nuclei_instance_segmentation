import tifffile as tif
import nibabel as nib
import numpy as np
import os, torch, imageio
from utils import eval_f1, mask2mask_list, extract_bboxes_torch


def main(pred_r):
    eval_device = 0
    # gt_r = '/BAND/USERS/ziquanw/data/Felix_P4/'
    gt_r = '/BAND/USERS/ziquanw/data/Carolyn_org_Sept/images'
    fs = os.listdir(pred_r)
    tp_outs = []
    fp_outs = []
    fn_outs = []
    for f in fs:
        img = np.array(imageio.v2.mimread(os.path.join(gt_r, f)))
        pred = tif.imread(os.path.join(pred_r, f))
        # gtn = f.replace('.tif', '- completed.nii')
        # gt = nib.load(os.path.join(gt_r, gtn)).get_fdata()
        gtn = f.replace('.tif', '.nii')
        gt = nib.load(os.path.join(gt_r.replace('images', 'masks'), gtn)).get_fdata()
        gt = np.transpose(gt, (2, 1, 0))[:, ::-1].copy()
        pred_list = mask2mask_list(pred)
        gt_list = mask2mask_list(gt)
        prec, rec, f1, _, pred_tp, gt_not_hitted = eval_f1(torch.from_numpy(pred_list).to('cuda:%d'%eval_device), torch.from_numpy(gt_list).to('cuda:%d'%eval_device))
        tp_pred = pred_list[np.where(pred_tp==1)[0]]
        fp_pred = pred_list[np.where(pred_tp==0)[0]]
        fn_gt = gt_list[gt_not_hitted]
        tp_out = analyse_mask(tp_pred, img, eval_device)
        tp_outs.append(tp_out)
        fp_out = analyse_mask(fp_pred, img, eval_device)
        fp_outs.append(fp_out)
        fn_out = analyse_mask(fn_gt, img, eval_device)  
        fn_outs.append(fn_out)
    tp_out = np.array(tp_outs).mean(0).tolist()
    fp_out = np.array(fp_outs).mean(0).tolist()
    fn_out = np.array(fn_outs).mean(0).tolist()
    print('tp_out: \n\
            px     avg=% 4.6f, std=% 4.6f, \n\
            area   avg=% 4.6f, std=% 4.6f, \n\
            shapez avg=% 4.6f, std=% 4.6f, \n\
            shapey avg=% 4.6f, std=% 4.6f, \n\
            shapex avg=% 4.6f, std=% 4.6f, \n\
            locz   avg=% 4.6f, std=% 4.6f, \n\
            locy   avg=% 4.6f, std=% 4.6f, \n\
            locx   avg=% 4.6f, std=% 4.6f' % tuple(tp_out))
    print('fp_out: \n\
            px     avg=% 4.6f, std=% 4.6f, \n\
            area   avg=% 4.6f, std=% 4.6f, \n\
            shapez avg=% 4.6f, std=% 4.6f, \n\
            shapey avg=% 4.6f, std=% 4.6f, \n\
            shapex avg=% 4.6f, std=% 4.6f, \n\
            locz   avg=% 4.6f, std=% 4.6f, \n\
            locy   avg=% 4.6f, std=% 4.6f, \n\
            locx   avg=% 4.6f, std=% 4.6f' % tuple(fp_out))
    print('fn_out: \n\
            px     avg=% 4.6f, std=% 4.6f, \n\
            area   avg=% 4.6f, std=% 4.6f, \n\
            shapez avg=% 4.6f, std=% 4.6f, \n\
            shapey avg=% 4.6f, std=% 4.6f, \n\
            shapex avg=% 4.6f, std=% 4.6f, \n\
            locz   avg=% 4.6f, std=% 4.6f, \n\
            locy   avg=% 4.6f, std=% 4.6f, \n\
            locx   avg=% 4.6f, std=% 4.6f' % tuple(fn_out))


def analyse_mask(mask_list, img, eval_device):
    out = []
    bbox = extract_bboxes_torch(torch.from_numpy(mask_list).to('cuda:%d'%eval_device), is_3d=True)
    bbox = bbox.cpu().numpy()
    for mask in mask_list:
        out.append([img[mask].mean(), img[mask].std(), mask.sum()])
    out = np.array(out)
    out_mean = out.mean(0)
    outs = [out_mean[0], out_mean[1]]
    for avg, std in zip(out_mean[2:], out[2:].std(0)):
        outs.extend([avg, std])
    outs.extend([bbox[:, 3].mean(), bbox[:, 3].std(), bbox[:, 4].mean(), bbox[:, 4].std(), bbox[:, 5].mean(), bbox[:, 5].std(), bbox[:, 0].mean(), bbox[:, 0].std(), bbox[:, 1].mean(), bbox[:, 1].std(), bbox[:, 2].mean(), bbox[:, 2].std()])
    return outs
        



if __name__ == "__main__":
    # main('results_Felix_P4/cellpose3D')
    main('results_Caroly_P15/cellpose3D')