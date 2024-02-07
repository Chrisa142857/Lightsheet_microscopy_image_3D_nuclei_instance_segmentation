import numpy as np
import tifffile as tif
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
# from assign2Dto3D_graph import assign_fun
import models
import torch, sys, os, imageio
# from test_tss import test_onestack as tsstest_onestack
from two_slice_stitch import StitchModel
from utils import mask2mask_list, eval_f1
import nibabel as nib
# from graph_models import MOTMPNet, get_model_config
# from graph_track import encode_graph_output, encode_graph_output_low_ram, graph_track_once, nms_3dmasks

def main_cellpose3D(_r, save_tag='Carolyn_P15'):
    print(save_tag, "cellpose3D")
    eval_device = 0
    tag = 'cellpose3D'
    save_r = '/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/train_data/test_out_rescaled_n12'
    # save_r = 'results_%s/%s' % (save_tag, tag)
    os.makedirs(save_r, exist_ok=True)
    # trained_model = 'cellpose/models/cellpose_residual_on_style_on_concatenation_off_train_2023_02_28_09_28_46.761192_epoch_41'
    # trained_model = 'cellpose/models/P4_models/cellpose_residual_on_style_on_concatenation_off_train_2023_04_10_21_02_12.036323_epoch_41'
    # trained_model = 'cellpose/models/P15_P4_models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_15_12_23_32.710910_epoch_61'
    # trained_model = 'cellpose/models/P15_P4_models_rescaled/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_18_33_47.131380_epoch_61'
    # trained_model = 'cellpose/models/P15_P4_models_rescaled_nuclei/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_22_42_54.153497_epoch_21'
    trained_model = '../../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n12/models/cellpose_residual_on_style_on_concatenation_off_train_n12_2024_01_22_18_19_24.791368_epoch_81'
    # model = models.CellposeModel(device=torch.device('cuda:%d' % int(sys.argv[3])), pretrained_model=trained_model)
    model = models.CellposeModel(device=torch.device('cuda:0'), pretrained_model=trained_model)
    # testlist = [f for f in os.listdir(_r) if f.endswith('.tif')]
    # with open('test_p%d_list.txt' % (4 if '4' in save_tag else 15), 'r') as f:
    with open('../../downloads/stack_data/test_list.txt', 'r') as f:
        testlist = f.read().split('\n')[:-1]
    # testlist = [f.split('/')[-1].replace('_masks', '') for f in testlist]
    precs, recs, f1s = [], [], []
    all_time = 0
    for f in testlist:
        # img = np.array(imageio.v2.mimread(os.path.join(_r, f)))
        img = np.array(imageio.v2.mimread(f))
        if 'rescaled-as-P15' not in _r:
            gtn = f.replace('.tif', '- completed.nii') if 'Felix' in save_tag else f.replace('.tif', '.nii')
            gt = nib.load(os.path.join(_r.replace('images', 'masks'), gtn)).get_fdata()
            gt = np.transpose(gt, (2, 1, 0))[:, ::-1].copy()
        else:
            gt = tif.imread(os.path.join(_r, f.replace('.tif', '_masks.tif')))
        otime = time.time()
        masks, _, _ = model.eval(img, diameter=None, channels=[0,0], do_3D=True, stitch_threshold=0)
        all_time += time.time()-otime
        tif.imwrite('%s/%s' % (save_r, f), masks) # cellpose
        continue
        prec, rec, f1, _, _, _ = eval_f1(torch.from_numpy(mask2mask_list(masks)).to('cuda:%d'%eval_device), torch.from_numpy(mask2mask_list(gt)).to('cuda:%d'%eval_device))
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    # print("F1: %.6f, Prec: %.6f, Rec: %.6f, time: %.6f" % (np.mean(f1s), np.mean(precs), np.mean(recs), all_time/len(testlist)))

def main_ours(_r, gt_r=None, save_tag='Carolyn_P15'):
    # print(save_tag, "cellpose2D_simZ")
    # if gt_r is None: gt_r = _r
    eval_device = 0
    save_r = '/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/train_data/test_out_rescaled_n8'
    # tag = 'cellpose2D_cosine_sim'
    # save_r = 'results_%s/%s' % (save_tag, tag)
    # os.makedirs(save_r, exist_ok=True)
    # if '4' in save_tag:
    #     trained_model = 'cellpose/models/cellpose_residual_on_style_on_concatenation_off_train_2023_02_28_09_28_46.761192_epoch_41'
    # else:
    #     trained_model = 'cellpose/models/P4_models/cellpose_residual_on_style_on_concatenation_off_train_2023_04_10_21_02_12.036323_epoch_41'
    # trained_model = 'cellpose/models/P15_P4_models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_15_12_23_32.710910_epoch_61'
    # trained_model = 'cellpose/models/P15_P4_models_rescaled/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_18_33_47.131380_epoch_61'
    # trained_model = '../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n12/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_18_33_47.131380_epoch_81'
    trained_model = '../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n8/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_18_33_47.131380_epoch_1'
    # trained_model = 'downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_22_42_54.153497_epoch_21'

    # model = models.CellposeModel(device=torch.device('cuda:%d' % int(sys.argv[3])), pretrained_model=trained_model)
    model = models.CellposeModel(device=torch.device('cuda:0'), pretrained_model=trained_model)
    # testlist = [f for f in os.listdir(_r) if f.endswith('.tif')]
    # with open('downloads/stack_data/test_p%d_list.txt' % (4 if '4' in save_tag else 15), 'r') as f:
    #     testlist = f.read().split('\n')[:-1]
    
    with open('../downloads/stack_data/test_list.txt', 'r') as f:
        testlist = f.read().split('\n')[:-1]
    # testlist = [f.split('/')[-1].replace('_masks', '') for f in testlist]
    # testlist = ['FelixP4_L91D814P6_stitched_x6700y5000z0900.tif']
    precs, recs, f1s = [], [], []
    all_time = 0
    for f in testlist:
        # img = np.array(imageio.v2.mimread(os.path.join(_r, f)))
        img = np.array(imageio.v2.mimread(f))
        otime = time.time()
        masks, _, _ = model.eval(img, diameter=None, channels=[0,0], do_3D=True, stitch_threshold=0, do_sim_Z=True)
        all_time += time.time()-otime
        tif.imwrite('%s/%s' % (save_r, f.split('/')[-1]), masks) # cellpose
        continue
        # exit()
        if 'rescaled-as-P15' not in gt_r:
            gtn = f.replace('.tif', '- completed.nii') if 'P4' in save_tag else f.replace('.tif', '.nii')
            gt = nib.load(os.path.join(gt_r.replace('images', 'masks'), gtn)).get_fdata()
            gt = np.transpose(gt, (2, 1, 0))[:, ::-1].copy()
        else:
            gt = tif.imread(os.path.join(gt_r, f.replace('.tif', '_masks.tif')))
        prec, rec, f1, _, _, _ = eval_f1(torch.from_numpy(mask2mask_list(masks)).to('cuda:%d'%eval_device), torch.from_numpy(mask2mask_list(gt)).to('cuda:%d'%eval_device))
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    print("F1: %.6f, Prec: %.6f, Rec: %.6f, time: %.6f" % (np.mean(f1s), np.mean(precs), np.mean(recs), all_time/len(testlist)))

# def main_cellpose2D_simZ_graph(_r, save_tag='Carolyn_P15'):
#     print(save_tag, "cellpose2D_simZ")
#     eval_device = 2
#     tag = 'cellpose2D_simZ_graph'
#     save_r = 'results_%s/%s' % (save_tag, tag)
#     os.makedirs(save_r, exist_ok=True)
#     # trained_model = 'cellpose/models/cellpose_residual_on_style_on_concatenation_off_train_2023_02_28_09_28_46.761192_epoch_41'
#     trained_model = 'cellpose/models/P4_models/cellpose_residual_on_style_on_concatenation_off_train_2023_04_10_21_02_12.036323_epoch_41'
#     # model = models.CellposeModel(device=torch.device('cuda:%d' % int(sys.argv[3])), pretrained_model=trained_model)
#     model = models.CellposeModel(device=torch.device('cuda:1'), pretrained_model=trained_model)    
#     graph_model = StitchModel('cuda:%d'%eval_device)    
#     graph_model.to('cuda:%d'%eval_device)
#     graph_model.load_state_dict(torch.load('tss_trainv2_output_filter_pyramid/epoch188.pth'))
#     testlist = [f for f in os.listdir(_r) if f.endswith('.tif') and not f.endswith('_masks.tif')]
#     with open('test_p4_list.txt', 'r') as f:
#         testlist = f.read().split('\n')[:-1]
#     testlist = [f.split('/')[-1].replace('_masks', '') for f in testlist]
#     precs, recs, f1s = [], [], []
#     all_time = 0
#     for f in testlist:
#         img = np.array(imageio.v2.mimread(os.path.join(_r, f)))
#         gtn = f.replace('.tif', '- completed.nii') if 'Felix' in save_tag else f.replace('.tif', '.nii')
#         gt = nib.load(os.path.join(_r.replace('images', 'masks'), gtn)).get_fdata()
#         gt = np.transpose(gt, (2, 1, 0))[:, ::-1].copy()
#         otime = time.time()
#         masks = []
#         for im in img:
#             mask, _, _ = model.eval(im, diameter=None, channels=[0,0], do_3D=False, stitch_threshold=0)
#             masks.append(mask)
#         masks, _ = tsstest_onestack(np.stack(masks), img, graph_model, 'cuda:%d'%eval_device)
#         all_time += time.time()-otime
#         tif.imwrite('%s/%s' % (save_r, f), masks) # cellpose
#         prec, rec, f1, _, _, _ = eval_f1(torch.from_numpy(mask2mask_list(masks)).to('cuda:%d'%eval_device), torch.from_numpy(mask2mask_list(gt)).to('cuda:%d'%eval_device))
#         precs.append(prec)
#         recs.append(rec)
#         f1s.append(f1)

#     print("F1: %.6f, Prec: %.6f, Rec: %.6f, time: %.6f" % (np.mean(f1s), np.mean(precs), np.mean(recs), all_time/len(testlist)))

# def main_cellpose2D_graph(_r):
#     best_graph_epoch = 116 # 38 116
#     graph_model_ckpt = '/BAND/USERS/ziquanw/wholebrain_code/models/graph_model_epoch%d.pth' % best_graph_epoch
#     graph_model = MOTMPNet(get_model_config(2))
#     graph_model.load_state_dict(torch.load(graph_model_ckpt, map_location='cpu'))
#     graph_model.eval()
#     graph_device = 'cuda:0'
#     num_slice_one_graph = 8
#     graph_step = 8
#     eval_device = 0
#     tag = 'cellpose2D_graph_stitch'
#     save_r = 'results_Felix_P4/%s' % tag
#     os.makedirs(save_r, exist_ok=True)
#     trained_model = 'cellpose/models/cellpose_residual_on_style_on_concatenation_off_train_2023_02_28_09_28_46.761192_epoch_41'
#     # model = models.CellposeModel(device=torch.device('cuda:%d' % int(sys.argv[3])), pretrained_model=trained_model)
#     model = models.CellposeModel(device=torch.device('cuda:1'), pretrained_model=trained_model)
#     testlist = [f for f in os.listdir(_r) if f.endswith('.tif')]
#     precs, recs, f1s = [], [], []
#     for f in testlist:
#         imgs = imageio.v2.mimread(os.path.join(_r, f))
#         gtn = f.replace('.tif', '- completed.nii')
#         gt = nib.load(os.path.join(_r, gtn)).get_fdata()
#         gt = np.transpose(gt, (2, 1, 0))[:, ::-1].copy()
#         out_2d, _, _ = model.eval(imgs, diameter=None, channels=[0,0], do_3D=False, stitch_threshold=0)
#         out_2d = [torch.from_numpy(np.stack([masks == mi for mi in np.unique(masks) if mi != 0])).to(graph_device) for masks in out_2d]
#         masks, _, _ = assign_fun(out_2d, torch.from_numpy(np.stack(imgs).astype(np.float32)).to(graph_device), graph_model, num_slice_one_graph, step=graph_step, graph_device=graph_device, post_device='cuda:%d'%eval_device)
#         mask_list = []
#         mask0 = masks[0].clone()
#         for mi, mask in enumerate(masks):
#             mask_list.extend([(mask==mid).permute(2, 0, 1) for mid in mask.unique() if mid != 0])
#             if mi != 0: mask0 += mask
#         mask0 = mask0.permute(2, 0, 1)
#         tif.imwrite('%s/%s' % (save_r, f), mask0.cpu().numpy()) # cellpose
#         prec, rec, f1, _, _, _ = eval_f1(torch.stack(mask_list).to('cuda:%d'%eval_device), torch.from_numpy(mask2mask_list(gt)).to('cuda:%d'%eval_device))
#         precs.append(prec)
#         recs.append(rec)
#         f1s.append(f1)

#     print("F1: %.6f, Prec: %.6f, Rec: %.6f" % (np.mean(f1s), np.mean(precs), np.mean(recs)))

def main_cellpose2D_iou(_r):
    eval_device = 0
    tag = 'cellpose2D_iou_stitch'
    save_r = 'results_Felix_P4/%s' % tag
    os.makedirs(save_r, exist_ok=True)
    trained_model = 'cellpose/models/cellpose_residual_on_style_on_concatenation_off_train_2023_02_28_09_28_46.761192_epoch_41'
    # model = models.CellposeModel(device=torch.device('cuda:%d' % int(sys.argv[3])), pretrained_model=trained_model)
    model = models.CellposeModel(device=torch.device('cuda:1'), pretrained_model=trained_model)
    testlist = [f for f in os.listdir(_r) if f.endswith('.tif')]
    precs, recs, f1s = [], [], []
    for f in testlist:
        imgs = np.stack(imageio.v2.mimread(os.path.join(_r, f)))
        gtn = f.replace('.tif', '- completed.nii')
        gt = nib.load(os.path.join(_r, gtn)).get_fdata()
        gt = np.transpose(gt, (2, 1, 0))[:, ::-1].copy()
        masks, _, _ = model.eval(imgs, diameter=None, channels=[0,0], do_3D=False, stitch_threshold=0.3)
        tif.imwrite('%s/%s' % (save_r, f), masks) # cellpose
        prec, rec, f1, _, _, _ = eval_f1(torch.from_numpy(mask2mask_list(masks)).to('cuda:%d'%eval_device), torch.from_numpy(mask2mask_list(gt)).to('cuda:%d'%eval_device))
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    print("F1: %.6f, Prec: %.6f, Rec: %.6f" % (np.mean(f1s), np.mean(precs), np.mean(recs)))


if __name__ == "__main__":
    import time
    # main_cellpose3D('/BAND/USERS/ziquanw/data/Felix_P4/', 'Felix_P4')
    # main_cellpose2D_graph('/BAND/USERS/ziquanw/data/Felix_P4/')
    # main_cellpose2D_iou('/BAND/USERS/ziquanw/data/Felix_P4/')
    # main_cellpose3D('/BAND/USERS/ziquanw/data/Carolyn_org_Sept/images', 'Carolyn_P15')
    main_ours('downloads/stack_data/P4/', gt_r='downloads/stack_data/P4_rescaled-as-P15/', save_tag='P4')
    # main_ours('downloads/stack_data/P4/', save_tag='P4')
    # main_ours('../downloads/stack_data/P4_rescaled-as-P15/', 'P4')
    # main_ours('/BAND/USERS/ziquanw/data/Carolyn_org_Sept/images', 'Carolyn_P15')
    # main_cellpose3D('/BAND/USERS/ziquanw/data/Felix_P4/', 'Felix_P4')
    # main_cellpose3D('/BAND/USERS/ziquanw/data/Felix_P4_rescaled-as-P15/', 'Felix_P4')
    # main_ours('/BAND/USERS/ziquanw/data/Felix_P4/', 'Felix_P4')
    # main_cellpose2D_simZ_graph('/BAND/USERS/ziquanw/data/Carolyn_org_Sept/images', 'Carolyn_P15')
    # main_cellpose2D_simZ_graph('/BAND/USERS/ziquanw/data/Felix_P4/', 'Felix_P4')
