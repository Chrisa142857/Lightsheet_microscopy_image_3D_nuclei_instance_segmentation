import numpy as np
import os, torch, copy, scipy, skimage, json
from tqdm import trange, tqdm
import nibabel as nib
from PIL import Image
from multiprocessing import Pool
import matplotlib.pyplot as plt

from skimage.registration import phase_cross_correlation


from datetime import datetime
OVERLAP_R = 0.15 # P14
zratio = 2.5/4

def main():
    # ptag='pair4'
    # btag='220904_L35D719P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_21-49-38'
    # btag='220902_L35D719P3_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-41-36'
    # get_stitch_tform(ptag, btag)
    # stitch_ls_image(ptag, btag)
    # exit()
    not_ready = ['pair15', 'pair22', 'pair3', 'pair10', 'pair8', 'pair9', 'pair5', 'pair12', 'pair21', 'pair6', 'pair11']
    r = '/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P14'
    # for ptag in os.listdir(r):
    #     if ptag in not_ready or ptag in done: continue
    #     for btag in os.listdir(f'{r}/{ptag}'):
    #         save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/{ptag}/{btag.split("_")[1]}'
    #         result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}/{btag}'
    #         root = result_path + '/UltraII[%02d x %02d]'
    #         stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    #         if len(stack_names) == 0: 
    #             print(ptag, btag.split('_')[1], "NIS not completed, skip")
    #             continue
    #         print("=== PCC stitch ", ptag, btag.split('_')[1], "===")
    #         if not os.path.exists(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine.json'):
    #             get_stitch_tform(ptag, btag)
    done = ['L95P2']
    brain_ready = [
        # ['female', '231028_L106P5_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_01-41-16'],
        # ['female', '231026_L102P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-39-32'],
        # ['female', '230902_L86P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_22-54-23'],
        # ['female', '230902_L86P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-25-11'],
        # ['female', '231028_L106P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_14-48-05'],
        # ['female', '230907_L88P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_18-15-15'],
        # ['male', '230826_L94P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_23-30-15'],
        # ['male', '230820_L88P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-00-37'],
        # ['male', '230826_L94P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_19-04-29'],
        # ['male', '230820_L88P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_21-23-23'],
        # ['male', '230818_L87D868P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_15-26-37'],
        # ['male', '230813_L82D711P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_23-59-14'],
        # ['male', '230810_L68P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_15-49-53'],
        ['male', '230811_L68D767P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_12-13-51'],
        ['male', '230819_L87P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_09-23-37'],
        ['male', '230825_L92P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_08-59-16']
    ]
    for ptag, btag in brain_ready:
    # for ptag in ['male', 'female']:
    #     for btag in os.listdir(f'{r}/{ptag}'):
            if btag.split('_')[1] in done: continue
            ls_image_root = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P14/{ptag}/{btag}'
            save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/P14/{ptag}/{btag.split("_")[1]}'
            result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P14/{ptag}/{btag}'
            root = result_path + '/UltraII[%02d x %02d]'
            if not os.path.exists(root % (0, 0)): continue
            stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
            if len(stack_names) == 0: 
                print(ptag, btag.split('_')[1], "NIS not completed, skip")
                continue
            print("=== PCC stitch ", ptag, btag.split('_')[1], "===")
            if not os.path.exists(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine.json'):
                get_stitch_tform(ptag, btag, ls_image_root, save_path, result_path)

    # for ptag in ['male', 'female']:
    # # for ptag in os.listdir(r):
    #     # if ptag in not_ready or ptag in done: continue
    #     for btag in os.listdir(f'{r}/{ptag}'):
    #         if 'L74D769P4' in btag: continue
    #         save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/P14/{ptag}/{btag.split("_")[1]}'
    #         if os.path.exists(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine.json'):
    #             print("=== Stitch LS", ptag, btag.split('_')[1], "===")
    #             stitch_ls_image(ptag, btag)

    # ptag='pair6'
    # btag='220415_L57D855P1_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_16-27-16'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair14'
    # btag='220722_L73D766P5_OUT_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_16-49-30'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair14'
    # btag='220723_L73D766P7_OUT_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_11-23-11'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair20'
    # btag='220917_L79D769P9_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-06-03'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)

    # ptag='pair9'
    # btag='220702_L64D804P4_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_09-53-19'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair9'
    # btag='220701_L64D804P6_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-52-42'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair6'
    # btag='220416_L57D855P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_09-52-07'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair5'
    # btag='220422_L57D855P4_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_16-54-32'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair5'
    # btag='220423_L57D855P5_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_09-02-27'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair12'
    # btag='220716_L66D764P6_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_09-49-51'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair12'
    # btag='220715_L66D764P5_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-49-35'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair21'
    # btag='220923_L91D814P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-24-18'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair21'
    # btag='220927_L91D814P6_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-26-45'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair11'
    # btag='220703_L66D764P3_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_08-47-16'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair11'
    # btag='220704_L66D764P8_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_10-46-01'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair15'
    # btag='220729_L73D766P4_topro_brn2_ctip2_4x_50sw_11hdf_0_108na_4z_20ov_17-07-38'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair15'
    # btag='220730_L73D766P9_topro_brn2_ctip2_4x_50sw_11hdf_0_108na_4z_20ov_10-58-21'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair22'
    # btag='220928_L91D814P3_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-03-23'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair22'
    # btag='220924_L91D814P4_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_09-07-40'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair3'
    # btag='220219_L35D719P4_topro_ctip2_brn2_4x_50sw_0_108na_11hdf_4z_15ov_11-46-33'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair10'
    # btag='220624_L64D804P3_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_16-44-01'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair10'
    # btag='220625_L64D804P9_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_10-50-29'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair8'
    # btag='220429_L59D878P2_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_17-04-49'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair8'
    # btag='220430_L59D878P5_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_13-03-21'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair16'
    # btag='220805_L74D769P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_15-51-36'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair18'
    # btag='220808_L77D764P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_16-45-14'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    #################################################
    # ptag='pair18'
    # btag='220809_L77D764P8_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_09-52-21'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    #######################################

    # ptag='pair21'
    # btag='220923_L91D814P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-24-18'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair6'
    # btag='220415_L57D855P1_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_16-27-16'
    # print("=== PCC stitch ", ptag, btag.split("_")[1], "===")
    # get_stitch_tform(ptag, btag)
    # ptag='pair13'
    # btag='220828_L69D764P9_OUT_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-49-09'

def get_stitch_tform(
    ptag='pair4', 
    btag='220904_L35D719P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_21-49-38',
    # btag='220902_L35D719P3_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-41-36',
    ls_image_root = f'/lichtman/Felix/Lightsheet/P4',
    save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg',
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4',
    overlap_r=OVERLAP_R
):

    # overlap_r = OVERLAP_R
    # ls_image_root = f'{ls_image_root}/{ptag}/{btag}'
    # save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/{ptag}/{btag.split("_")[1]}'
    # result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}/{btag}'
    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path) if 'Ultra' in fn])
    ncol, nrow = tile_loc.max(0)+1
    # print(tile_loc, nrow, ncol)
    assert len(tile_loc) == nrow*ncol, f'tile of raw data is not complete, tile location: {tile_loc}'

    root = result_path + '/UltraII[%02d x %02d]'
    stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    stack_names = sort_stackname(stack_names)
    # neighbor = [[-1, 0], [0, -1], [-1, -1], [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1]]
    os.makedirs(f'{save_path}/NIS_tranform', exist_ok=True)

    for stack_name in stack_names:
        meta_name = stack_name.replace('instance_center', 'seg_meta')
        zstart = int(stack_name.split('zmin')[1].split('_')[0])
        zstart = int(zstart*zratio)
        seg_shape = torch.load(f'{root % (0, 0)}/{meta_name}')
        zend = int(zstart + seg_shape[0].item()*zratio)

    zstart = 0 

    # tile_lt_loc = {
    #     f'{i}-{j}': [i*seg_shape[1]*(1-overlap_r), j*seg_shape[2]*(1-overlap_r)] for i in range(ncol) for j in range(nrow)
    # }

    row_order = [nrow//2] + [(-1)**i*(i//2+1)+nrow//2 for i in range(nrow-1)]
    row_order = [i for i in row_order if i in list(range(nrow))]
    for i in range(nrow): 
        if i not in row_order: row_order.append(i)
    col_order = [ncol//2] + [(-1)**i*(i//2+1)+ncol//2 for i in range(ncol-1)]
    col_order = [i for i in col_order if i in list(range(ncol))]
    for i in range(ncol): 
        if i not in col_order: col_order.append(i)

    print(datetime.now(), f"Preload {ncol*nrow} raw 3D LS images for stitch")
    # fn_ = ls_image_root.split('_')[-1]+'_'+'_'.join(btag.split('_')[1:-1])
    tile_overlap = {}
    for i in range(ncol):
        for j in range(nrow):
            if f'{i}-{j}' not in tile_overlap:
                fn_ = os.listdir(f'{ls_image_root}/UltraII[{i:02d} x {j:02d}]')[0]
                fn_ = fn_.split('_')[0] + '_%04d_' + '_'.join(fn_.split('_')[2:-1]) + '_xyz-Table Z%04d.ome.tif'
                # fn = f'{ls_image_root}/UltraII[{i:02d} x {j:02d}]/'
                fn = f'{ls_image_root}/UltraII[{i:02d} x {j:02d}]/{fn_}'
                tile_overlap[f'{i}-{j}'] = get_tile_stack_overlap_area(fn, range(zstart, zend), overlap_r)

    '''
    Coarse reg by 3D PCC (down_r = 0.5)
    '''
    tform_stack_coarse_colrow = {}
    neighbor_row = [[0, -1], [0, 1]]
    neighbor_col = [[-1, 0], [1, 0]]
    down_r = 0.5
    for i in range(ncol):
        j = 0
        tform_stack_coarse_colrow[f'{i}-{j}'] = [[0], [0], [0]]
        for j in range(1, nrow):
            # if f'{i}-{j}' not in tile_overlap:
            #     fn = f'{ls_image_root}/{fn_}_UltraII[{i:02d} x {j:02d}]_C01_xyz-Table Z%04d.ome.tif'
            #     tile_overlap[f'{i}-{j}'] = get_tile_stack_overlap_area(fn, range(zstart, zend), overlap_r)

            moving_image = tile_overlap[f'{i}-{j}']
            if f'{i}-{j}' not in tform_stack_coarse_colrow:
                tform_stack_coarse_colrow[f'{i}-{j}'] = [[], [], []]
            for pi, pj in neighbor_row:
                if f'{i+pi}-{j+pj}' in tform_stack_coarse_colrow: 
                    # if f'{i+pi}-{j+pj}' not in tile_overlap:
                    #     fn = f'{ls_image_root}/{fn_}_UltraII[{(i+pi):02d} x {(j+pj):02d}]_C01_xyz-Table Z%04d.ome.tif'
                    #     tile_overlap[f'{i+pi}-{j+pj}'] = get_tile_stack_overlap_area(fn, range(zstart, zend), overlap_r)

                    reference_image = tile_overlap[f'{i+pi}-{j+pj}']
                    if pj < 0:
                        moving_image = moving_image[2].astype(float)
                        reference_image = reference_image[3].astype(float)
                    elif pj > 0:
                        moving_image = moving_image[3].astype(float)
                        reference_image = reference_image[2].astype(float)
                    
                    moving_image = torch.nn.functional.interpolate(torch.from_numpy(moving_image)[None, None], scale_factor=down_r)[0,0].numpy()
                    reference_image = torch.nn.functional.interpolate(torch.from_numpy(reference_image)[None, None], scale_factor=down_r)[0,0].numpy()
                    print(datetime.now(), f"start phase cross correlation mov ({i},{j}) to ({i+pi},{j+pj})")
                    shift, error, diffphase = phase_cross_correlation(reference_image, moving_image, overlap_ratio=0.8)
                    shift = [s/down_r for s in shift]

                    tz, tx, ty = shift
                    print(datetime.now(), f"done phase cross correlation, shift: {shift}, error: {error:.6f}, dphase: {diffphase:.6f}")
                    tform_stack_coarse_colrow[f'{i}-{j}'][0].append(tz)
                    tform_stack_coarse_colrow[f'{i}-{j}'][1].append(tx)
                    tform_stack_coarse_colrow[f'{i}-{j}'][2].append(ty)
                    break
                    
    for j in range(nrow):
        for i in range(1, ncol):
            moving_image = tile_overlap[f'{i}-{j}']
            assert f'{i}-{j}' in tform_stack_coarse_colrow, f'{i}-{j}'
            for pi, pj in neighbor_col:
                if f'{i+pi}-{j+pj}' in tform_stack_coarse_colrow: 
                    reference_image = tile_overlap[f'{i+pi}-{j+pj}']
                    if pi < 0:
                        moving_image = moving_image[0].astype(float)
                        reference_image = reference_image[1].astype(float)
                    elif pi > 0:
                        moving_image = moving_image[1].astype(float)
                        reference_image = reference_image[0].astype(float)
                    
                    moving_image = torch.nn.functional.interpolate(torch.from_numpy(moving_image)[None, None], scale_factor=down_r)[0,0].numpy()
                    reference_image = torch.nn.functional.interpolate(torch.from_numpy(reference_image)[None, None], scale_factor=down_r)[0,0].numpy()
                    print(datetime.now(), f"start phase cross correlation mov ({i},{j}) to ({i+pi},{j+pj})")
                    shift, error, diffphase = phase_cross_correlation(reference_image, moving_image, overlap_ratio=0.8)
                    shift = [s/down_r for s in shift]

                    tz, tx, ty = shift
                    print(datetime.now(), f"done phase cross correlation, shift: {shift}, error: {error:.6f}, dphase: {diffphase:.6f}")
                    tform_stack_coarse_colrow[f'{i}-{j}'][0].append(tz)
                    tform_stack_coarse_colrow[f'{i}-{j}'][1].append(tx)
                    tform_stack_coarse_colrow[f'{i}-{j}'][2].append(ty)
                    
                    break

    # tformed_tile_lt_loc = {zstart: copy.deepcopy(tile_lt_loc)}
    tform_stack_coarse = fuse_colrow_tform(tform_stack_coarse_colrow)
    for k in tform_stack_coarse_colrow:
        tz, tx, ty = tform_stack_coarse[k]
        # tformed_tile_lt_loc[0][k][0] = tformed_tile_lt_loc[0][k][0] + tx
        # tformed_tile_lt_loc[0][k][1] = tformed_tile_lt_loc[0][k][1] + ty
        # print(k, tformed_tile_lt_loc[zstart][k], tform_stack_coarse[k])
        print(k, tform_stack_coarse[k])

    with open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_coarse.json', 'w', encoding='utf-8') as f:
        json.dump(tform_stack_coarse, f, ensure_ascii=False, indent=4)

    for k in tqdm(tile_overlap, desc='Apply coarse to img'):
        for i, img in enumerate(tile_overlap[k]):
            tile_overlap[k][i] = shift_image(torch.from_numpy(img.astype(float)), tform_stack_coarse[k]).numpy().astype(np.uint16)
            # print(img.shape, tile_overlap[k][i].shape)
    
    '''
    Refine 2D by PCC (down_r = 1)
    '''
    tform_stack_refine_colrow = [{} for _ in range(zstart, zend)]
    neighbor_row = [[0, -1], [0, 1]]
    neighbor_col = [[-1, 0], [1, 0]]
    down_r = 1
    max_shift = 0.05
    for i in range(ncol):
        j = 0
        for zi in range(zstart, zend):
            tform_stack_refine_colrow[zi][f'{i}-{j}'] = [[0], [0]]
        
        for j in range(1, nrow):
            moving_image = tile_overlap[f'{i}-{j}']
            assert moving_image[0].shape[0] == zend-zstart, moving_image[0].shape[0]
            pre_shift = [0, 0]
            for zi in trange(moving_image[0].shape[0]-1, -1, -1, desc=f"mov ({i},{j})"):
                if f'{i}-{j}' not in tform_stack_refine_colrow[zi]:
                    tform_stack_refine_colrow[zi][f'{i}-{j}'] = [[], []]
                    
                for pi, pj in neighbor_row:
                    if f'{i+pi}-{j+pj}' in tform_stack_refine_colrow[zi]: 
                        reference_image = tile_overlap[f'{i+pi}-{j+pj}']
                        
                        if pj < 0:
                            moving_image2d = moving_image[2]
                            reference_image2d = reference_image[3]
                        elif pj > 0:
                            moving_image2d = moving_image[3]
                            reference_image2d = reference_image[2]
                        
                        reference_image2d = reference_image2d[zi].astype(float)
                        moving_image2d = moving_image2d[zi].astype(float)
                        # if down_r != 1:
                        moving_image2d = torch.nn.functional.interpolate(torch.from_numpy(moving_image2d)[None, None], scale_factor=down_r)[0,0].numpy()
                        reference_image2d = torch.nn.functional.interpolate(torch.from_numpy(reference_image2d)[None, None], size=moving_image2d.shape)[0,0].numpy()
                        assert moving_image2d.shape[0] == reference_image2d.shape[0] and moving_image2d.shape[1] == reference_image2d.shape[1], f"{moving_image2d.shape} != {reference_image2d.shape}"
                        shift, error, diffphase = phase_cross_correlation(reference_image2d, moving_image2d, overlap_ratio=0.99)
                        if shift[0] > reference_image2d.shape[0]*max_shift or shift[1] > reference_image2d.shape[1]*max_shift:
                            shift = pre_shift
                        else:
                            shift = [s/down_r for s in shift]

                        tx, ty = shift
                        tform_stack_refine_colrow[zi][f'{i}-{j}'][0].append(tx)
                        tform_stack_refine_colrow[zi][f'{i}-{j}'][1].append(ty)
                        pre_shift = shift
                        break
                    
    for j in range(nrow):
        for i in range(1, ncol):
            moving_image = tile_overlap[f'{i}-{j}']
            for zi in trange(moving_image[0].shape[0]-1, -1, -1, desc=f"mov ({i},{j})"):
                pre_shift = [0, 0]
                assert f'{i}-{j}' in tform_stack_refine_colrow[zi], f'{i}-{j}'
                for pi, pj in neighbor_col:
                    if f'{i+pi}-{j+pj}' in tform_stack_refine_colrow[zi]: 
                        reference_image = tile_overlap[f'{i+pi}-{j+pj}']
                        if pi < 0:
                            moving_image2d = moving_image[0]
                            reference_image2d = reference_image[1]
                        elif pi > 0:
                            moving_image2d = moving_image[1]
                            reference_image2d = reference_image[0]

                        reference_image2d = reference_image2d[zi].astype(float)
                        moving_image2d = moving_image2d[zi].astype(float)
                        # if down_r != 1:
                        moving_image2d = torch.nn.functional.interpolate(torch.from_numpy(moving_image2d)[None, None], scale_factor=down_r)[0,0].numpy()
                        reference_image2d = torch.nn.functional.interpolate(torch.from_numpy(reference_image2d)[None, None], size=moving_image2d.shape)[0,0].numpy()
                        shift, error, diffphase = phase_cross_correlation(reference_image2d, moving_image2d, overlap_ratio=0.99)
                        if shift[0] > reference_image2d.shape[0]*max_shift or shift[1] > reference_image2d.shape[1]*max_shift:
                            shift = pre_shift
                        else:
                            shift = [s/down_r for s in shift]

                        tx, ty = shift
                        tform_stack_refine_colrow[zi][f'{i}-{j}'][0].append(tx)
                        tform_stack_refine_colrow[zi][f'{i}-{j}'][1].append(ty)
                        pre_shift = shift
                        break

    tform_stack_refine = [{} for _ in range(len(tform_stack_refine_colrow))]
    # tformed_tile_lt_loc_refine = {zi: copy.deepcopy(tformed_tile_lt_loc[0]) for zi in range(zstart, zend)}
    for zi in range(moving_image[0].shape[0]-1, -1, -1):
        tform_stack_refine[zi] = fuse_colrow_tform(tform_stack_refine_colrow[zi])
        for k in tform_stack_refine_colrow[zi]:
            tx, ty = tform_stack_refine[zi][k]
            # tformed_tile_lt_loc_refine[zi][k][0] = tformed_tile_lt_loc_refine[zi][k][0] + tx
            # tformed_tile_lt_loc_refine[zi][k][1] = tformed_tile_lt_loc_refine[zi][k][1] + ty
    
    with open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine.json', 'w', encoding='utf-8') as f:
        json.dump(tform_stack_refine, f, ensure_ascii=False, indent=4)


def stitch_ls_image(
    ptag='pair4', 
    btag='220904_L35D719P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_21-49-38',
    # btag='220902_L35D719P3_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-41-36'
):
    '''
    Apply stitch to LS image for QC
    '''
    overlap_r = OVERLAP_R

    save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/{ptag}/{btag.split("_")[1]}'
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}/{btag}'
    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path) if 'Ultra' in fn])
    ncol, nrow = tile_loc.max(0)+1
    # print(tile_loc, nrow, ncol)
    assert len(tile_loc) == nrow*ncol, f'tile of raw data is not complete, tile location: {tile_loc}'

    ls_image_root = f'/lichtman/Felix/Lightsheet/P4/{ptag}/{btag}'
    fn_ = ls_image_root.split('_')[-1]+'_'+'_'.join(btag.split('_')[1:-1])
    root = result_path + '/UltraII[%02d x %02d]'
    stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    stack_names = sort_stackname(stack_names)
    print(stack_names)
    neighbor = [[-1, 0], [0, -1], [-1, -1], [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1]]
    os.makedirs(f'{save_path}/NIS_tranform', exist_ok=True)

    for stack_name in stack_names:
        meta_name = stack_name.replace('instance_center', 'seg_meta')
        zstart = int(stack_name.split('zmin')[1].split('_')[0])
        zstart = int(zstart*zratio)
        seg_shape = torch.load(f'{root % (0, 0)}/{meta_name}')
        zend = int(zstart + seg_shape[0].item()*zratio)

    zstart = 0 
    tform_xy_max = [0.05*seg_shape[1], 0.05*seg_shape[2]]

    tile_w = int(seg_shape[1].item()*(1-overlap_r))
    tile_h = int(seg_shape[2].item()*(1-overlap_r))
    overlap_w = int(seg_shape[1].item()*overlap_r)
    overlap_h = int(seg_shape[2].item()*overlap_r)
    max_pixel = None
    image_stitcher = image_stitch_QCer(seg_shape, overlap_w, overlap_h)
    os.makedirs(f'{save_path}/LS_image_stitched', exist_ok=True)
    pre_startx, pre_starty = {}, {}

    tform_stack_coarse = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_coarse.json', 'r', encoding='utf-8'))
    tform_stack_refine = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine.json', 'r', encoding='utf-8'))
    if os.path.exists(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine_ptreg.json'):
        tform_stack_ptreg = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine_ptreg.json', 'r', encoding='utf-8'))
    else:
        tform_stack_ptreg = None
    '''
    Get lefttop location of tile stack after stitching
    '''
    tile_lt_loc = {
        f'{i}-{j}': [i*seg_shape[1]*(1-overlap_r), j*seg_shape[2]*(1-overlap_r)] for i in range(ncol) for j in range(nrow)
    }
    ## Coarse
    tformed_tile_lt_loc = {0: copy.deepcopy(tile_lt_loc)}
    for k in tform_stack_coarse:
        tz, tx, ty = tform_stack_coarse[k]
        tformed_tile_lt_loc[0][k][0] = tformed_tile_lt_loc[0][k][0] + tx
        tformed_tile_lt_loc[0][k][1] = tformed_tile_lt_loc[0][k][1] + ty
    ## Refine
    tformed_tile_lt_loc_refine = {zi: copy.deepcopy(tformed_tile_lt_loc[0]) for zi in range(zstart, zend)}
    for zi in range(zstart, zend):
        for k in tform_stack_refine[zi]:
            tx, ty = tform_stack_refine[zi][k]
            tformed_tile_lt_loc_refine[zi][k][0] = tformed_tile_lt_loc_refine[zi][k][0] + tx
            tformed_tile_lt_loc_refine[zi][k][1] = tformed_tile_lt_loc_refine[zi][k][1] + ty

    print(zstart, zend, len(tformed_tile_lt_loc_refine))
    for zi in tqdm(tformed_tile_lt_loc_refine, desc='Apply transformation'):
        valid_wsi = True
        for ijstr in tformed_tile_lt_loc_refine[zi]:
            i, j = ijstr.split('-')
            i, j = int(i), int(j)
            tz = tform_stack_coarse[f'{i}-{j}'][0]
            tz = int(np.around(tz))
            wsii = zi - tz
            valid_wsi = wsii>=0 and wsii<len(tformed_tile_lt_loc_refine)
            if not valid_wsi: break
            
        if not valid_wsi: continue
        wsi = np.zeros((tile_w*ncol+overlap_w+1, tile_h*nrow+overlap_h+1), dtype=np.float32)
        for ijstr in tformed_tile_lt_loc_refine[zi]:
            if ijstr not in pre_startx:
                pre_startx[ijstr] = None
                pre_starty[ijstr] = None
            
            i, j = ijstr.split('-')
            i, j = int(i), int(j)
                
            tz = tform_stack_coarse[f'{i}-{j}'][0]
            tz = int(np.around(tz))
            wsii = zi - tz

            fn = f'{fn_}_UltraII[{i:02d} x {j:02d}]_C01_xyz-Table Z{wsii:04d}.ome.tif'
            tile_img = Image.open(f'{ls_image_root}/{fn}')
            tile_img_ = np.asarray(tile_img)
            max_pix = np.percentile(tile_img_, 99)
            if max_pixel is None: max_pixel = max_pix
            max_pixel = max(max_pix, max_pixel)
            tile_img = (tile_img_-tile_img_.min())/(max_pixel-tile_img_.min())
            
            if abs(tform_stack_refine[zi][ijstr][0]) > tform_xy_max[0] or abs(tform_stack_refine[zi][ijstr][1]) > tform_xy_max[1]:
                if pre_startx[ijstr] is not None: 
                    startx, starty = pre_startx[ijstr], pre_starty[ijstr]
                else:
                    for zii in range(zi, len(tform_stack_refine)):
                        if abs(tform_stack_refine[zii][ijstr][0]) <= tform_xy_max[0] and abs(tform_stack_refine[zii][ijstr][1]) <= tform_xy_max[1]:
                            zii = max(zii + tz, 0)
                            zii = min(max(list(tformed_tile_lt_loc_refine.keys())), zii)
                            startx, starty = tformed_tile_lt_loc_refine[zii][ijstr]
                            break
            else:
                startx, starty = tformed_tile_lt_loc_refine[zi][ijstr]
            
            pre_startx[ijstr] = startx
            pre_starty[ijstr] = starty

            if tform_stack_ptreg is not None:
                if ijstr in tform_stack_ptreg[zi]:
                    tx, ty = tform_stack_ptreg[zi][ijstr]
                    startx = startx + tx
                    starty = starty + ty

            startx, endx = int(startx), tile_img.shape[0] + int(startx)
            starty, endy = int(starty), tile_img.shape[1] + int(starty)
            if startx < 0:
                tile_img = tile_img[-startx:]
                startx = 0
            if starty < 0:
                tile_img = tile_img[:, -starty:]
                starty = 0
            if endx > wsi.shape[0]: 
                tile_img = tile_img[:-(endx-wsi.shape[0])]
                endx = wsi.shape[0]

            if endy > wsi.shape[1]: 
                tile_img = tile_img[:, :-(endy-wsi.shape[1])]
                endy = wsi.shape[1]

            cur_tile = wsi[startx:endx, starty:endy]
            fg_mask = cur_tile>0
            if fg_mask.any():
                tile_img[fg_mask] = image_stitcher.adaptive_image_stitch(tile_img[fg_mask], cur_tile[fg_mask], fg_mask)

            wsi[startx:endx, starty:endy] = tile_img
        
        save_fn = f'{save_path}/LS_image_stitched/{btag.split("_")[1]}_TOPRO_C01_Z{zi:04d}.ptreg_stitch.tif'
        Image.fromarray(wsi).save(save_fn)


def fuse_colrow_tform(tform_colrow):
    rows = []
    cols = []
    for k in tform_colrow:
        i, j = k.split('-')
        i, j = int(i), int(j)
        if i not in rows: rows.append(i)
        if j not in cols: cols.append(j)
    ## average the shorter dimension
    # do_avg_row = max(cols) > max(rows)
    D = np.array(tform_colrow[k]).shape[0]
    ## average the std smaller dimension
    std_col = []
    for i in rows:
        avg_col = []
        for j in cols:
            k = f'{i}-{j}'
            avg_col.append(np.array(tform_colrow[k])[:, 0])
        std_col.append(np.stack(avg_col).std(0)[0])
    std_row = []
    for j in cols:
        avg_row = []
        for i in rows:
            k = f'{i}-{j}'
            if i>0: avg_row.append(np.array(tform_colrow[k])[:, 1])
        std_row.append(np.stack(avg_row).std(0)[0])
    do_avg_row = np.mean(std_row) < np.mean(std_col)

    ## accumulate and average
    tform = {}
    avg = {}
    for i in rows:
        for j in cols:
            k = f'{i}-{j}'
            if do_avg_row:
                if i>0:
                    tform[k] = np.array(tform_colrow[k])[:, 1]
                else:
                    tform[k] = np.zeros(D)
                tform[k] = tform[k] + (tform[f'{i-1}-{j}'] if i>0 else 0)
                if j not in avg: avg[j] = []
                avg[j].append(np.array(tform_colrow[k])[:, 0])
            else:
                tform[k] = np.array(tform_colrow[k])[:, 0]
                tform[k] = tform[k] + (tform[f'{i}-{j-1}'] if j>0 else 0)
                if i not in avg: avg[i] = []
                if i>0: 
                    avg[i].append(np.array(tform_colrow[k])[:, 1])
                else:
                    avg[i].append(np.zeros(D))

    for i in avg:
        if i == 0: 
            avg[i] = 0
        else:
            avg[i] = np.stack(avg[i]).mean(0) + avg[i-1]

    ## fuse two dimensions
    for i in rows:
        for j in cols:
            k = f'{i}-{j}'
            if do_avg_row:
                tform[k] = tform[k] + avg[j]
            else:
                tform[k] = tform[k] + avg[i]
    
    for k in tform:
        tform[k] = [float(t) for t in tform[k]]
        
    return tform


class image_stitch_QCer:
    def __init__(self, seg_shape, overlap_w, overlap_h):
        x_grad1 = torch.cat([torch.arange(overlap_w).float() / overlap_w, 
            torch.ones(seg_shape[1]-2*overlap_w),
            torch.arange(overlap_w, 0, -1) / overlap_w])

        y_grad1 = torch.cat([torch.arange(overlap_h).float() / overlap_h, 
            torch.ones(seg_shape[2]-2*overlap_h),
            torch.arange(overlap_h, 0, -1) / overlap_h])

        self.mov_adaptive_grad = x_grad1[:, None] * y_grad1[None, :]

        x_grad2 = torch.cat([torch.arange(overlap_w, 0, -1) / overlap_w, 
            torch.zeros(seg_shape[1]-2*overlap_w),
            torch.arange(overlap_w).float() / overlap_w])

        y_grad2 = torch.cat([torch.arange(overlap_h, 0, -1) / overlap_h, 
            torch.zeros(seg_shape[2]-2*overlap_h),
            torch.arange(overlap_h).float() / overlap_h])

        self.tgt_adaptive_grad = x_grad2[:, None] * y_grad1[None, :] + x_grad1[:, None] * y_grad2[None, :] + x_grad2[:, None] * y_grad2[None, :]

    def adaptive_image_stitch(self, mov_overlap, tgt_overlap, overlap_mask):
        overlap_coord = np.where(overlap_mask)
        if len(overlap_coord) == 3:
            _, ox, oy = overlap_coord
        else:
            ox, oy = overlap_coord
            
        mov = self.mov_adaptive_grad[ox, oy].numpy() * mov_overlap
        tgt = self.tgt_adaptive_grad[ox, oy].numpy() * tgt_overlap
        return (mov + tgt)


def get_tile_stack_overlap_area(fn, zrange, overlap_r):
    stack_overlaps = [[], [], [], []]
    with Pool(processes=min(len(zrange),30)) as loader_pool:
        image_list = list(loader_pool.imap(Image.open, tqdm([fn % (zi,zi) for zi in zrange], desc=f'Load {fn.split("/")[-2]}')))
    # for zi in zrange:
    for tile_img in image_list:
        # tile_img = Image.open(fn % (zi,zi))
        tile_img = np.asarray(tile_img)
        h, w = tile_img.shape
        oh = int(h*overlap_r) + 1
        ow = int(w*overlap_r) + 1
        t = tile_img[:oh]
        b = tile_img[h-oh:]
        assert t.shape[0] == b.shape[0], f"{t.shape[0]} != {b.shape[0]}"
        l = tile_img[:, :ow]
        r = tile_img[:, w-ow:]
        assert l.shape[1] == r.shape[1], f"{l.shape[1]} != {r.shape[1]}"
        stack_overlaps[0].append(t)
        stack_overlaps[1].append(b)
        stack_overlaps[2].append(l)
        stack_overlaps[3].append(r)
    
    stack_overlaps = [
        np.stack(stack_overlaps[0]),
        np.stack(stack_overlaps[1]),
        np.stack(stack_overlaps[2]),
        np.stack(stack_overlaps[3]),
    ]
    return stack_overlaps

def shift_image(image, shift):
    s = int(shift[0])
    if s < 0:
        image = torch.cat([image[-s:], torch.zeros(-s, image.shape[1], image.shape[2], dtype=image.dtype)])
    elif s > 0:
        image = torch.cat([torch.zeros(s, image.shape[1], image.shape[2], dtype=image.dtype), image[:-s]])
    
    s = int(shift[1])
    if s < 0:
        image = torch.cat([image[:, -s:], torch.zeros(image.shape[0], -s, image.shape[2], dtype=image.dtype)], 1)
    elif s > 0:
        image = torch.cat([torch.zeros(image.shape[0], s, image.shape[2], dtype=image.dtype), image[:, :-s]], 1)
        
    s = int(shift[2])
    if s < 0:
        image = torch.cat([image[:, :, -s:], torch.zeros(image.shape[0], image.shape[1], -s, dtype=image.dtype)], 2)
    elif s > 0:
        image = torch.cat([torch.zeros(image.shape[0], image.shape[1], s, dtype=image.dtype), image[:, :, :-s]], 2)
        
    return image
    
def sort_stackname(stack_names):
    stack_z = []
    for stack_name in stack_names:
        stack_z.append(int(stack_name.split('zmin')[1].split('_')[0]))

    argsort = np.argsort(stack_z)
    return [stack_names[i] for i in argsort]

if __name__ == '__main__': main()