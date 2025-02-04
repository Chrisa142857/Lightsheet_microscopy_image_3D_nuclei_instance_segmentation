import torch, os, json, copy
import numpy as np
from tqdm import trange, tqdm
# from final_stitch import ZRANGE, OVERLAP_R
import nibabel as nib
from datetime import datetime
from torchvision.ops._utils import _upcast
from torch_scatter import scatter_max

OVERLAP_R = 0.2
zratio = 2.5/4

def main():
    device = 'cuda:3'
    # ptag='pair5'
    # # btag='220423_L57D855P5_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_09-02-27'
    # btag='220422_L57D855P4_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_16-54-32'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag = 'pair13'
    # btag = '220827_L69D764P6_OUT_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-47-13'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    ptag = 'pair14'
    btag = '220722_L73D766P5_OUT_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_16-49-30'
    print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    remove_doubled_cell(ptag, btag, device)
    whole_brain_map(ptag, btag, device)
    # ptag = 'pair18'
    # btag = '220809_L77D764P8_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_09-52-21'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    exit()
    # whole_brain_map(ptag='pair16', btag='220805_L74D769P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_15-51-36')
    # exit()
    # remove_doubled_cell(ptag='pair4', btag='220904_L35D719P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_21-49-38')
    whole_brain_map(ptag='pair4', btag='220904_L35D719P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_21-49-38', device=device)
    # remove_doubled_cell(ptag='pair16', btag='220807_L74D769P8_OUT_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_11-51-47')
    # whole_brain_map(ptag='pair16', btag='220807_L74D769P8_OUT_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_11-51-47')
    # exit()
    # whole_brain_map(ptag='pair4', btag='220902_L35D719P3_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-41-36')
    # not_ready = ['pair15', 'pair22', 'pair3', 'pair10', 'pair8', 'pair9', 'pair5', 'pair12', 'pair21', 'pair6']
    # done = ['pair4']
    # r = '/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4'
    # device = 'cuda:3'
    # not_ready_btag = ['220415_L57D855P1_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_16-27-16', '220911_L79D769P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_02-59-03', '220828_L69D764P9_OUT_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-49-09', '220722_L73D766P5_OUT_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_16-49-30', '220723_L73D766P7_OUT_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_11-23-11', '220917_L79D769P9_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-06-03']
    # for ptag in os.listdir(r):
    #     if ptag in not_ready or ptag in done: continue
    #     for btag in os.listdir(f'{r}/{ptag}'):
    #         if 'L74D769P4' in btag: continue
    #         if btag in not_ready_btag: continue
    #         save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/{ptag}/{btag.split("_")[1]}'
    #         # if os.path.exists(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine.json'):
    #         print("=== Remove doubled cell", ptag, btag.split('_')[1], "===")
    #         remove_doubled_cell(ptag, btag, device)
    #         print("=== Density map", ptag, btag.split('_')[1], "===")
    #         whole_brain_map(ptag, btag, device)

    # ptag='pair13'
    # btag='220828_L69D764P9_OUT_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-49-09'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair19'
    # btag='220911_L79D769P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_02-59-03'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    ptag='pair20'
    btag='220912_L79D769P7_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_11-27-41'
    print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    whole_brain_map(ptag, btag, device)
    # ptag='pair14'
    # btag='220722_L73D766P5_OUT_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_16-49-30'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair6'
    # btag='220415_L57D855P1_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_16-27-16'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)

    # ptag='pair9'
    # btag='220702_L64D804P4_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_09-53-19'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair9'
    # btag='220701_L64D804P6_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-52-42'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair6'
    # btag='220416_L57D855P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_09-52-07'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair14'
    # btag='220723_L73D766P7_OUT_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_11-23-11'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    ptag='pair20'
    btag='220917_L79D769P9_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-06-03'
    print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    whole_brain_map(ptag, btag, device)
    
    #########################
    # ptag='pair19'
    # btag='220911_L79D769P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_02-59-03'
    # print("=== NMS remove doubled NIS", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair20'
    # btag='220912_L79D769P7_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_11-27-41'
    # print("=== NMS remove doubled NIS", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)

    #########################
    # ptag='pair6'
    # btag='220415_L57D855P1_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_16-27-16'
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)

    # ptag='pair6'
    # btag='220416_L57D855P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_09-52-07'
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)

    # ptag='pair9'
    # btag='220702_L64D804P4_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_09-53-19'
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)

    # ptag='pair9'
    # btag='220701_L64D804P6_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-52-42'
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)

    # ptag='pair13'
    # btag='220828_L69D764P9_OUT_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-49-09'
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)

    # ptag='pair14'
    # btag='220723_L73D766P7_OUT_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_11-23-11'
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)

    ########################
    ptag='pair4'
    btag='220902_L35D719P3_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-41-36'
    whole_brain_map(ptag, btag, device)
    # ptag='pair16'
    # btag='220807_L74D769P8_OUT_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_11-51-47'
    # whole_brain_map(ptag, btag, device)
    # ptag='pair17'
    # btag='220819_L77D764P2_OUT_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_16-54-05'
    # whole_brain_map(ptag, btag, device)
    # ptag='pair19'
    # btag='220911_L79D769P8_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-27-26'
    # whole_brain_map(ptag, btag, device)
    # ptag='pair20'
    # btag='220917_L79D769P9_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-06-03'
    # whole_brain_map(ptag, btag, device)

    #########################
    
    # ptag='pair19'
    # btag='220911_L79D769P8_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-27-26'
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair20'
    # btag='220917_L79D769P9_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-06-03'
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)

    #########################

    # ptag='pair5'
    # btag='220422_L57D855P4_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_16-54-32'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair5'
    # btag='220423_L57D855P5_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_09-02-27'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair12'
    # btag='220716_L66D764P6_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_09-49-51'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair12'
    # btag='220715_L66D764P5_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-49-35'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    ptag='pair21'
    btag='220923_L91D814P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-24-18'
    print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    whole_brain_map(ptag, btag, device)
    ptag='pair21'
    btag='220927_L91D814P6_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-26-45'
    print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    whole_brain_map(ptag, btag, device)
    # ptag='pair11'
    # btag='220703_L66D764P3_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_08-47-16'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair11'
    # btag='220704_L66D764P8_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_10-46-01'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair15'
    # btag='220729_L73D766P4_topro_brn2_ctip2_4x_50sw_11hdf_0_108na_4z_20ov_17-07-38'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair15'
    # btag='220730_L73D766P9_topro_brn2_ctip2_4x_50sw_11hdf_0_108na_4z_20ov_10-58-21'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair22'
    # btag='220928_L91D814P3_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-03-23'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair22'
    # btag='220924_L91D814P4_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_09-07-40'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair3'
    # btag='220219_L35D719P4_topro_ctip2_brn2_4x_50sw_0_108na_11hdf_4z_15ov_11-46-33'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    ptag='pair10'
    btag='220624_L64D804P3_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_16-44-01'
    print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    whole_brain_map(ptag, btag, device)
    ptag='pair10'
    btag='220625_L64D804P9_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_10-50-29'
    print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    whole_brain_map(ptag, btag, device)
    # ptag='pair8'
    # btag='220429_L59D878P2_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_17-04-49'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair8'
    # btag='220430_L59D878P5_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_13-03-21'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair16'
    # btag='220805_L74D769P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_15-51-36'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    # ptag='pair18'
    # btag='220808_L77D764P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_16-45-14'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)
    #############################################
    # ptag='pair6'
    # btag='220415_L57D855P1_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_16-27-16'
    # whole_brain_map(ptag, btag, 'cuda:2')
    # exit()
    #############################################
    # ptag='pair6'
    # btag='220416_L57D855P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_09-52-07'
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair5'
    # btag='220422_L57D855P4_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_16-54-32'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair5'
    # btag='220423_L57D855P5_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_09-02-27'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair12'
    # btag='220716_L66D764P6_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_09-49-51'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair12'
    # btag='220715_L66D764P5_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-49-35'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair21'
    # btag='220927_L91D814P6_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-26-45'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair11'
    # btag='220703_L66D764P3_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_08-47-16'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair11'
    # btag='220704_L66D764P8_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_10-46-01'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    ptag='pair15'
    btag='220729_L73D766P4_topro_brn2_ctip2_4x_50sw_11hdf_0_108na_4z_20ov_17-07-38'
    print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    whole_brain_map(ptag, btag, device)
    ptag='pair15'
    btag='220730_L73D766P9_topro_brn2_ctip2_4x_50sw_11hdf_0_108na_4z_20ov_10-58-21'
    print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    whole_brain_map(ptag, btag, device)
    # ptag='pair22'
    # btag='220928_L91D814P3_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-03-23'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair22'
    # btag='220924_L91D814P4_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_09-07-40'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair3'
    # btag='220219_L35D719P4_topro_ctip2_brn2_4x_50sw_0_108na_11hdf_4z_15ov_11-46-33'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair10'
    # btag='220624_L64D804P3_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_16-44-01'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair10'
    # btag='220625_L64D804P9_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_10-50-29'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair8'
    # btag='220429_L59D878P2_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_17-04-49'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair8'
    # btag='220430_L59D878P5_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_13-03-21'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    ptag='pair16'
    btag='220805_L74D769P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_15-51-36'
    print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    whole_brain_map(ptag, btag, device)
    # ptag='pair18'
    # btag='220808_L77D764P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_16-45-14'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    #################
    
    # ptag='pair21'
    # btag='220923_L91D814P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-24-18'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair3'
    # btag='220219_L35D719P4_topro_ctip2_brn2_4x_50sw_0_108na_11hdf_4z_15ov_11-46-33'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair6'
    # btag='220415_L57D855P1_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_16-27-16'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair9'
    # btag='220701_L64D804P6_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-52-42'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    ptag='pair16'
    btag='220807_L74D769P8_OUT_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_11-51-47'
    # remove_doubled_cell(ptag, btag, device)
    whole_brain_map(ptag, btag, device)
    # ptag='pair19'
    # btag='220911_L79D769P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_02-59-03'
    # print("=== NMS remove doubled NIS", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair20'
    # btag='220912_L79D769P7_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_11-27-41'
    # print("=== NMS remove doubled NIS", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair19'
    # btag='220911_L79D769P8_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-27-26'
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    ########################
    
    # ptag='pair13'
    # btag='220828_L69D764P9_OUT_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-49-09'
    # print("=== NMS remove doubled NIS", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    ########################
    # ptag='pair4'
    # btag='220902_L35D719P3_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-41-36'
    # print("=== NMS remove doubled NIS", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)

    #########################
    # ptag='pair15'
    # btag='220729_L73D766P4_topro_brn2_ctip2_4x_50sw_11hdf_0_108na_4z_20ov_17-07-38'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair15'
    # btag='220730_L73D766P9_topro_brn2_ctip2_4x_50sw_11hdf_0_108na_4z_20ov_10-58-21'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    ptag='pair16'
    btag='220807_L74D769P8_OUT_topro_brn2_ctip2_4x_11hdf_50sw_0_108na_4z_20ov_11-51-47'
    print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    whole_brain_map(ptag, btag, device)
    # ptag='pair19'
    # btag='220911_L79D769P8_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-27-26'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    ptag='pair17'
    btag='220819_L77D764P2_OUT_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_16-54-05'
    print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    whole_brain_map(ptag, btag, device)
    ptag='pair17'
    btag='220820_L77D764P9_OUT_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_17-00-41'
    print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    whole_brain_map(ptag, btag, device)
    # ptag='pair21'
    # btag='220923_L91D814P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_16-24-18'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair14'
    # btag='220723_L73D766P7_OUT_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_11-23-11'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair13'
    # btag='220828_L69D764P9_OUT_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-49-09'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair19'
    # btag='220911_L79D769P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_02-59-03'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # ptag='pair11'
    # btag='220704_L66D764P8_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_10-46-01'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    ## stitching lines ############################################
    # ptag='pair5'
    # btag='220423_L57D855P5_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_09-02-27'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)
    # posthoc_stitch_line(ptag, btag, device)
    # ptag='pair8'
    # btag='220430_L59D878P5_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_13-03-21'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # remove_doubled_cell(ptag, btag, device)
    # whole_brain_map(ptag, btag, device)

    ## wired doubled NIS ############################################
    # ptag='pair6'
    # btag='220416_L57D855P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_09-52-07'
    # print("=== whole_brain_map ", ptag, btag.split("_")[1], "===")
    # whole_brain_map(ptag, btag, device)


def remove_doubled_cell(ptag, btag, device='cuda:3'):
    overlap_r = OVERLAP_R

    save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/{ptag}/{btag.split("_")[1]}'
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}/{btag}'
    root = result_path + '/UltraII[%02d x %02d]'
    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path) if 'Ultra' in fn])
    ncol, nrow = tile_loc.max(0)+1
    assert len(tile_loc) == nrow*ncol, f'tile of raw data is not complete, tile location: {tile_loc}'
    stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    stack_names = sort_stackname(stack_names)
    for stack_name in stack_names:
        meta_name = stack_name.replace('instance_center', 'seg_meta')
        zstart = int(stack_name.split('zmin')[1].split('_')[0])
        zstart = int(zstart*zratio)
        seg_shape = torch.load(f'{root % (0, 0)}/{meta_name}')
        zend = int(zstart + seg_shape[0].item()*zratio)

    zstart = 0 

    if os.path.exists(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_manual.json'):
        tform_stack_manual = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_manual.json', 'r', encoding='utf-8'))

    else:
        tform_stack_manual = None
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
    if tform_stack_manual is not None:
        tformed_tile_lt_loc_refined = {zi: copy.deepcopy(tile_lt_loc) for zi in range(zstart, zend)}
        for zi in range(zstart, zend):
            for k in tform_stack_manual[zi]:
                tz, tx, ty = tform_stack_manual[zi][k]
                tformed_tile_lt_loc_refined[zi][k][0] = tformed_tile_lt_loc_refined[zi][k][0] + tx
                tformed_tile_lt_loc_refined[zi][k][1] = tformed_tile_lt_loc_refined[zi][k][1] + ty
    else:
        ## Coarse
        tformed_tile_lt_loc = {0: copy.deepcopy(tile_lt_loc)}
        for k in tform_stack_coarse:
            tz, tx, ty = tform_stack_coarse[k]
            tformed_tile_lt_loc[0][k][0] = tformed_tile_lt_loc[0][k][0] + tx
            tformed_tile_lt_loc[0][k][1] = tformed_tile_lt_loc[0][k][1] + ty
        ## Refine
        tformed_tile_lt_loc_refined = {zi: copy.deepcopy(tformed_tile_lt_loc[0]) for zi in range(zstart, zend)}
        for zi in range(zstart, zend):
            for k in tform_stack_refine[zi]:
                tx, ty = tform_stack_refine[zi][k]
                tformed_tile_lt_loc_refined[zi][k][0] = tformed_tile_lt_loc_refined[zi][k][0] + tx
                tformed_tile_lt_loc_refined[zi][k][1] = tformed_tile_lt_loc_refined[zi][k][1] + ty

    '''
    Get bbox and apply tform for NMS
    '''
    stack_nis_bbox = {}
    stack_nis_label = {}
    tile_center = {}
    # tform_xy_max = [0.05*seg_shape[1], 0.05*seg_shape[2]]
    tform_xy_max = [0.2*seg_shape[1], 0.2*seg_shape[2]]
    # for _j in trange(nrow, desc='Get bounding box of NIS'):
    print(datetime.now(), "Load bounding box after transform")
    for _j in range(nrow):
        for _i in range(ncol):
            k = f'{_i}-{_j}'
            _bbox = []
            _label = []
            for stack_name in stack_names:
                if not os.path.exists(f"{root % (_i, _j)}/{stack_name}"): continue
                zstart = int(stack_name.split('zmin')[1].split('_')[0])
                zstart = int(zstart*zratio)
                labelfn = f"{root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_label')}"
                label = torch.load(labelfn).long()
                bboxfn = f"{root % (_i, _j)}/{stack_name.replace('instance_center', 'instance_bbox')}"
                bbox = torch.load(bboxfn)
                bbox[:, 0] = bbox[:, 0] * zratio + zstart
                bbox[:, 3] = bbox[:, 3] * zratio + zstart
                    
                _bbox.append(bbox)
                _label.append(label)
            if len(_bbox) == 0: continue

            if k not in stack_nis_bbox: stack_nis_bbox[k] = []
            if k not in stack_nis_label: stack_nis_label[k] = []
            _bbox = torch.cat(_bbox)
            _label = torch.cat(_label)
            
            if tform_stack_manual is None:
                tz = tform_stack_coarse[k][0]
                _bbox[:, 0] = _bbox[:, 0] + tz
                _bbox[:, 3] = _bbox[:, 3] + tz
            ct = (_bbox[:, :3] + _bbox[:, 3:]) / 2
            pre_refine_lt_x = None
            pre_refine_lt_y = None
            for zi in tformed_tile_lt_loc_refined.keys():
                ct_zmask = torch.where(ct[:, 0].long() == zi)[0]
                if len(ct_zmask) == 0: continue
                bbox = _bbox[ct_zmask].clone()
                label = _label[ct_zmask]
                if tform_stack_manual is None:
                    if abs(tform_stack_refine[zi][k][0]) > tform_xy_max[0] or abs(tform_stack_refine[zi][k][1]) > tform_xy_max[1]:
                        if pre_refine_lt_x is not None:
                            refine_lt_x = pre_refine_lt_x
                            refine_lt_y = pre_refine_lt_y
                        else:
                            for zii in range(zi, len(tform_stack_refine)):
                                if abs(tform_stack_refine[zii][k][0]) <= tform_xy_max[0] and abs(tform_stack_refine[zii][k][1]) <= tform_xy_max[1]:
                                    refine_lt_x = tformed_tile_lt_loc_refined[zii][k][0]
                                    refine_lt_y = tformed_tile_lt_loc_refined[zii][k][1]
                                    break
                    else:
                        refine_lt_x = tformed_tile_lt_loc_refined[zi][k][0]
                        refine_lt_y = tformed_tile_lt_loc_refined[zi][k][1]
                else:
                    refine_lt_x = tformed_tile_lt_loc_refined[zi][k][0]
                    refine_lt_y = tformed_tile_lt_loc_refined[zi][k][1]
                pre_refine_lt_x = refine_lt_x
                pre_refine_lt_y = refine_lt_y
                if tform_stack_manual is None:
                    if tform_stack_ptreg is not None:
                        if k in tform_stack_ptreg[zi]:
                            tx, ty = tform_stack_ptreg[zi][k]
                            refine_lt_x = refine_lt_x + tx
                            refine_lt_y = refine_lt_y + ty

                bbox[:, 1] = bbox[:, 1] + refine_lt_x
                bbox[:, 2] = bbox[:, 2] + refine_lt_y
                bbox[:, 4] = bbox[:, 4] + refine_lt_x
                bbox[:, 5] = bbox[:, 5] + refine_lt_y
                if tform_stack_manual is not None:
                    bbox[:, 0] = bbox[:, 0] + tform_stack_manual[zi][k][0]
                    bbox[:, 3] = bbox[:, 3] + tform_stack_manual[zi][k][0]
                bbox = bbox.clip(min=0)
                stack_nis_bbox[k].append(bbox)
                stack_nis_label[k].append(label)
                if k not in tile_center:
                    tile_center[k] = np.array([refine_lt_x+seg_shape[1]*0.5, refine_lt_y+seg_shape[2]*0.5])
                else:
                    tile_center[k] = (tile_center[k] + np.array([refine_lt_x+seg_shape[1]*0.5, refine_lt_y+seg_shape[2]*0.5])) / 2

            stack_nis_bbox[k] = torch.cat(stack_nis_bbox[k])
            stack_nis_label[k] = torch.cat(stack_nis_label[k])

    '''
    NMS to remove doubled cells
    '''
    neighbor = [[-1, 0], [0, -1], [-1, -1], [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1]]
    rm_label = {}
    nms_r = overlap_r + 0.15*2
    nms_computed = []
    # if tform_stack_manual is None:
    tform_xy_max = [t*2 for t in tform_xy_max]
    for k in stack_nis_bbox:
        tile_lt_loc[k] = [tile_lt_loc[k][0]-tform_xy_max[0], tile_lt_loc[k][1]-tform_xy_max[1]]

    for k in stack_nis_bbox:
    # for k in ['1-1']:
        torch.cuda.empty_cache()
        i, j = k.split('-')
        i, j = int(i), int(j)
        if f'{i}-{j}' not in rm_label: rm_label[f'{i}-{j}'] = []
        bbox_tile = stack_nis_bbox[k].clone().to(device)
        label_tile = stack_nis_label[k].clone().to(device)
        lt_loc_tile = tile_lt_loc[k]
        for pi, pj in neighbor:
            if f'{i+pi}-{j+pj}' not in stack_nis_bbox: continue
            if f'{i}-{j}-{i+pi}-{j+pj}' in nms_computed: continue
            if f'{i+pi}-{j+pj}-{i}-{j}' in nms_computed: continue
            bbox_nei = stack_nis_bbox[f'{i+pi}-{j+pj}'].clone().to(device)
            label_nei = stack_nis_label[f'{i+pi}-{j+pj}'].clone().to(device)
            nms_computed.append(f'{i}-{j}-{i+pi}-{j+pj}')
            ###################################
            # mov_to_tgt = {}
            # if pi < 0: # left of mov to right of tgt
            #     mov_to_tgt[0] = 1
            # if pi > 0: # right of mov to left of tgt
            #     mov_to_tgt[1] = 0
            # if pj < 0: # bottom of mov to top of tgt
            #     mov_to_tgt[2] = 3
            # if pj > 0: # top of mov to bottom of tgt
            #     mov_to_tgt[3] = 2
            # mov_masks = bbox_in_stitching_seam(bbox_tile, lt_loc_tile, [seg_shape[1], seg_shape[2]], i, j, ncol, nrow, nms_r)
            # lt_loc_nei = tile_lt_loc[f'{i+pi}-{j+pj}']
            # tgt_masks = bbox_in_stitching_seam(bbox_nei, lt_loc_nei, [seg_shape[1], seg_shape[2]], i+pi, j+pj, ncol, nrow, nms_r)
            # tgt_mask = []
            # mov_mask = []
            # for mov_mi in range(len(mov_masks)):
            #     if mov_masks[mov_mi] is None: continue
            #     if mov_mi not in mov_to_tgt: continue
            #     tgt_mi = mov_to_tgt[mov_mi]
            #     if tgt_masks[tgt_mi] is None: continue
            #     mov_mask.append(mov_masks[mov_mi])
            #     tgt_mask.append(tgt_masks[tgt_mi])
            # assert len(mov_mask) <= 2, len(mov_mask)
            # if len(mov_mask) > 1: # corner of tile
            #     mov_mask = torch.logical_and(mov_mask[0], mov_mask[1])
            #     tgt_mask = torch.logical_and(tgt_mask[0], tgt_mask[1])
            # else: # edge of tile
            #     mov_mask = mov_mask[0]
            #     tgt_mask = tgt_mask[0]
            # mov_mask = torch.where(mov_mask)[0]
            # tgt_mask = torch.where(tgt_mask)[0]
            # bbox_tgt = bbox_nei[tgt_mask]
            # blabel_tgt = label_nei[tgt_mask]
            # bbox_mov = bbox_tile[mov_mask]
            # blabel_mov = label_tile[mov_mask]
            #########################
            bbox_tgt = bbox_nei
            blabel_tgt = label_nei
            bbox_mov = bbox_tile
            blabel_mov = label_tile
            #########################
            if len(bbox_tgt) == 0 or len(bbox_mov) == 0: continue
            ## minimal iou threshold
            print(datetime.now(), f"NMS between {i}-{j}, {i+pi}-{j+pj}", bbox_tgt.shape, bbox_mov.shape)
            # default threshold = 0.001
            rm_ind_tgt, rm_ind_mov = nms_bbox(
                bbox_tgt, bbox_mov, iou_threshold=0.1, 
                tile_tgt_center=tile_center[f'{i+pi}-{j+pj}'], tile_mov_center=tile_center[k], 
                seg_shape=[seg_shape[1], seg_shape[2]], device=device
            )

            rm_mask_tgt = torch.zeros(len(bbox_tgt), device=rm_ind_tgt.device, dtype=bool)
            rm_mask_tgt[rm_ind_tgt] = True
            rm_mask_mov = torch.zeros(len(bbox_mov), device=rm_ind_mov.device, dtype=bool)
            rm_mask_mov[rm_ind_mov] = True

            if f'{i+pi}-{j+pj}' not in rm_label: rm_label[f'{i+pi}-{j+pj}'] = []
                
            rm_label[f'{i+pi}-{j+pj}'].append(blabel_tgt[rm_mask_tgt])
            rm_label[f'{i}-{j}'].append(blabel_mov[rm_mask_mov])

            rm_mask_nei = torch.zeros(len(bbox_nei), device=bbox_nei.device, dtype=bool)
            rm_mask_tile = torch.zeros(len(bbox_tile), device=bbox_tile.device, dtype=bool)
            # rm_mask_nei[tgt_mask[rm_ind_tgt]] = True
            # rm_mask_tile[mov_mask[rm_ind_mov]] = True
            rm_mask_nei[rm_ind_tgt] = True
            rm_mask_tile[rm_ind_mov] = True
            bbox_tile = bbox_tile[torch.logical_not(rm_mask_tile)]
            label_tile = label_tile[torch.logical_not(rm_mask_tile)]
            stack_nis_bbox[f'{i+pi}-{j+pj}'] = stack_nis_bbox[f'{i+pi}-{j+pj}'][torch.logical_not(rm_mask_nei).cpu()]
            stack_nis_label[f'{i+pi}-{j+pj}'] = stack_nis_label[f'{i+pi}-{j+pj}'][torch.logical_not(rm_mask_nei).cpu()]

        stack_nis_bbox[f'{i}-{j}'] = bbox_tile
        stack_nis_label[f'{i}-{j}'] = label_tile
                
    for k in rm_label:
        if len(rm_label[k]) > 0:
            rm_label[k] = torch.cat(rm_label[k]).cpu()
        else:
            rm_label[k] = torch.zeros(0)
        print(rm_label[k].shape, "removals being recorded in tile", k)
    os.makedirs(f'{save_path}/doubled_NIS_label', exist_ok=True)
    torch.save(rm_label, f'{save_path}/doubled_NIS_label/{btag.split("_")[1]}_doubled_label.zip')
    torch.cuda.empty_cache()
    
def whole_brain_map(ptag, btag, device='cuda:3'):
    '''
    Whole brain map
    '''
    overlap_r = OVERLAP_R
    new_header, affine_m = init_nib_header()

    vol_unit = 0.75*0.75*2.5 # 2.5 is the fixed Z resolution of NIS
    downsample_res = [25, 25, 25]
    seg_res = [4, 0.75, 0.75]
    save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/{ptag}/{btag.split("_")[1]}'
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}/{btag}'
    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path) if 'Ultra' in fn])

    ncol, nrow = tile_loc.max(0)+1
    # print(tile_loc, nrow, ncol)
    assert len(tile_loc) == nrow*ncol, f'tile of raw data is not complete, tile location: {tile_loc}'

    root = result_path + '/UltraII[%02d x %02d]'
    stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    stack_names = sort_stackname(stack_names)
    sshape = [0, 0, 0]
    for si, stack_name in enumerate(stack_names):
        meta_name = stack_name.replace('instance_center', 'seg_meta')
        zstart = int(stack_name.split('zmin')[1].split('_')[0])
        zstart = int(zstart*zratio)
        seg_shape = torch.load(f'{root % (0, 0)}/{meta_name}')
        zend = int(zstart + seg_shape[0].item()*zratio)
        if sshape[0] < zend:
            sshape = [zend, int(seg_shape[1].item()*(1-overlap_r)*(ncol-1)+seg_shape[1].item()), int(seg_shape[2].item()*(1-overlap_r)*(nrow-1)+seg_shape[2].item())]
    
    tform_xy_max = [0.05*seg_shape[1], 0.05*seg_shape[2]]
    dratio = [s/d for s, d in zip(seg_res, downsample_res)]
    dshape = [0, 0, 0]
    dshape[0] = int(sshape[0] * dratio[0])
    dshape[1] = int(sshape[1] * dratio[1])
    dshape[2] = int(sshape[2] * dratio[2])
    print(datetime.now(), f"Downsample space shape: {dshape}, ratio: {dratio}, LS space shape {sshape}")
    zstart = 0
    tile_lt_loc = {
        f'{i}-{j}': [i*seg_shape[1]*(1-overlap_r), j*seg_shape[2]*(1-overlap_r)] for i in range(ncol) for j in range(nrow)
    }
    
    if os.path.exists(f'{save_path}/doubled_NIS_label/{btag.split("_")[1]}_doubled_label.zip'):
        rm_label = torch.load(f'{save_path}/doubled_NIS_label/{btag.split("_")[1]}_doubled_label.zip')
        for k in rm_label:
            rm_label[k] = rm_label[k].to(device)
            print(k, rm_label[k].shape, rm_label[k].min(), rm_label[k].max())
    else:
        rm_label = {}

    # if os.path.exists(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_manual.json'):
    #     tform_stack_manual = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_manual.json', 'r', encoding='utf-8'))
    #     print(datetime.now(), "Loaded tform of Imaris stitcher")
    # else:
    #     tform_stack_manual = None
    tform_stack_coarse = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_coarse.json', 'r', encoding='utf-8'))
    tform_stack_refine = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine.json', 'r', encoding='utf-8'))
    print(len(tform_stack_refine))
    if os.path.exists(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine_ptreg.json'):
        tform_stack_ptreg = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine_ptreg.json', 'r', encoding='utf-8'))
        print(datetime.now(), "Loaded tform of coarse, refine and pt-reg", len(tform_stack_ptreg), [tform_stack_ptreg[zi].keys() for zi in range(len(tform_stack_ptreg)) if len(tform_stack_ptreg[zi].keys()) > 0][-1])
    else:
        tform_stack_ptreg = None
        print(datetime.now(), "Loaded tform of coarse, refine")

    
    # if tform_stack_manual is not None:
    #     tformed_tile_lt_loc_refined = {zi: copy.deepcopy(tile_lt_loc) for zi in range(zstart, zend)}
    #     for zi in range(zstart, zend):
    #         for k in tform_stack_manual[zi]:
    #             tz, tx, ty = tform_stack_manual[zi][k]
    #             tformed_tile_lt_loc_refined[zi][k][0] = tformed_tile_lt_loc_refined[zi][k][0] + tx
    #             tformed_tile_lt_loc_refined[zi][k][1] = tformed_tile_lt_loc_refined[zi][k][1] + ty
    # else:
    ## Coarse
    tformed_tile_lt_loc = {0: copy.deepcopy(tile_lt_loc)}
    for k in tform_stack_coarse:
        tz, tx, ty = tform_stack_coarse[k]
        tformed_tile_lt_loc[0][k][0] = tformed_tile_lt_loc[0][k][0] + tx
        tformed_tile_lt_loc[0][k][1] = tformed_tile_lt_loc[0][k][1] + ty
    ## Refine
    tformed_tile_lt_loc_refined = {zi: copy.deepcopy(tformed_tile_lt_loc[0]) for zi in range(zstart, zend)}
    for zi in range(zstart, zend):
        for k in tform_stack_refine[zi]:
            tx, ty = tform_stack_refine[zi][k]
            tformed_tile_lt_loc_refined[zi][k][0] = tformed_tile_lt_loc_refined[zi][k][0] + tx
            tformed_tile_lt_loc_refined[zi][k][1] = tformed_tile_lt_loc_refined[zi][k][1] + ty
            if tform_stack_ptreg is not None:
                if k in tform_stack_ptreg[zi]:
                    tx, ty = tform_stack_ptreg[zi][k]
                    tformed_tile_lt_loc_refined[zi][k][0] = tformed_tile_lt_loc_refined[zi][k][0] + tx
                    tformed_tile_lt_loc_refined[zi][k][1] = tformed_tile_lt_loc_refined[zi][k][1] + ty

    '''
    Whole brain map
    '''
    centers = []
    vols = []
    labels = []
    colocs = []

    lrange = 1000
    for i in range(ncol):
        for j in range(nrow):
            k = f'{i}-{j}'
            print(datetime.now(), k)
            i, j = k.split('-')
            i, j = int(i), int(j)

            _bbox = []
            _label = []
            _vol = []
            for stack_name in stack_names:
                if not os.path.exists(f"{root % (i, j)}/{stack_name}"): continue
                zstart = int(stack_name.split('zmin')[1].split('_')[0])
                zstart = int(zstart*zratio)
                volfn = f"{root % (i, j)}/{stack_name.replace('instance_center', 'instance_volume')}"
                labelfn = f"{root % (i, j)}/{stack_name.replace('instance_center', 'instance_label')}"
                label = torch.load(labelfn).long()
                bboxfn = f"{root % (i, j)}/{stack_name.replace('instance_center', 'instance_bbox')}"
                bbox = torch.load(bboxfn)
                bbox[:, 0] = bbox[:, 0] * zratio + zstart
                bbox[:, 3] = bbox[:, 3] * zratio + zstart
                _vol.append(torch.load(volfn).long())    
                _bbox.append(bbox)
                _label.append(label)
            if len(_bbox) == 0: continue
            _vol = torch.cat(_vol)
            _bbox = torch.cat(_bbox)
            _label = torch.cat(_label)
            
            # if tform_stack_manual is None:
            tz = tform_stack_coarse[k][0]
            _bbox[:, 0] = _bbox[:, 0] + tz
            _bbox[:, 3] = _bbox[:, 3] + tz
            ct = (_bbox[:, :3] + _bbox[:, 3:]) / 2
            pre_refine_lt_x = None
            pre_refine_lt_y = None
            bbox = []
            label = []
            vol = []
            for zi in tformed_tile_lt_loc_refined.keys():
                ct_zmask = torch.where(ct[:, 0].long() == zi)[0]
                if len(ct_zmask) == 0: continue
                b = _bbox[ct_zmask].clone()
                l = _label[ct_zmask]
                v = _vol[ct_zmask]
                # if tform_stack_manual is None:
                if (abs(tform_stack_refine[zi][k][0]) > tform_xy_max[0] or abs(tform_stack_refine[zi][k][1]) > tform_xy_max[1]):
                    if pre_refine_lt_x is not None:
                        refine_lt_x = pre_refine_lt_x
                        refine_lt_y = pre_refine_lt_y
                    else:
                        for zii in range(zi, len(tform_stack_refine)):
                            if abs(tform_stack_refine[zii][k][0]) <= tform_xy_max[0] and abs(tform_stack_refine[zii][k][1]) <= tform_xy_max[1]:
                                refine_lt_x = tformed_tile_lt_loc_refined[zii][k][0]
                                refine_lt_y = tformed_tile_lt_loc_refined[zii][k][1]
                                break
                else:
                    refine_lt_x = tformed_tile_lt_loc_refined[zi][k][0]
                    refine_lt_y = tformed_tile_lt_loc_refined[zi][k][1]
                # else:
                #     refine_lt_x = tformed_tile_lt_loc_refined[zi][k][0]
                #     refine_lt_y = tformed_tile_lt_loc_refined[zi][k][1]
                pre_refine_lt_x = refine_lt_x
                pre_refine_lt_y = refine_lt_y
                # if tform_stack_manual is None:
                if tform_stack_ptreg is not None:
                    if k in tform_stack_ptreg[zi]:
                        tx, ty = tform_stack_ptreg[zi][k]
                        refine_lt_x = refine_lt_x + tx
                        refine_lt_y = refine_lt_y + ty

                b[:, 1] = b[:, 1] + refine_lt_x
                b[:, 2] = b[:, 2] + refine_lt_y
                b[:, 4] = b[:, 4] + refine_lt_x
                b[:, 5] = b[:, 5] + refine_lt_y
                # if tform_stack_manual is not None:
                #     b[:, 0] = b[:, 0] + tform_stack_manual[zi][k][0]
                #     b[:, 3] = b[:, 3] + tform_stack_manual[zi][k][0]
                bbox.append(b)
                label.append(l)
                vol.append(v)

            bbox = torch.cat(bbox).to(device)
            label = torch.cat(label).to(device)
            vol = torch.cat(vol).to(device)
            pt = (bbox[:, :3] + bbox[:, 3:]) / 2
            # pt[:, 0] = pt[:, 0] + tform_stack_coarse[k][0]
            # pt[:, 1] = pt[:, 1] + tform_stack_coarse[k][1] 
            # pt[:, 2] = pt[:, 2] + tform_stack_coarse[k][2] 
            torch.cuda.empty_cache()
            ########
            print("before remove doubled pt", pt.shape, label.shape)
            if k in rm_label:
                keep_ind = []
                for labeli in range(0, len(label), lrange):
                    label_batch = label[labeli:labeli+lrange]
                    label2rm = label_batch[:, None] == rm_label[k][None, :]
                    do_rm = label2rm.any(1)
        #                         rm_label[f'{i}-{j}'] = rm_label[f'{i}-{j}'][torch.logical_not(label2rm.any(0))]
                    keep_ind.append(torch.arange(labeli, labeli+len(label_batch), device=device)[torch.logical_not(do_rm)])
                if len(label) > 0:
                    keep_ind = torch.cat(keep_ind)
                    pt = pt[keep_ind]
                    vol = vol[keep_ind]
                    label = label[keep_ind]

            print("after remove doubled pt", pt.shape, label.shape)
            ####################
            if os.path.exists(f"{root % (i, j)}/{btag.split('_')[1]}_remap.zip"):
                zstitch_remap = torch.load(f"{root % (i, j)}/{btag.split('_')[1]}_remap.zip").to(device)
                print("z-stitch remap dict shape", zstitch_remap.shape)

                ## loc: gnn stitch source (current tile) nis index, stitch_remap_loc: index of pairs in the stitch remap list
                loc, stitch_remap_loc = [], []
                for lrangei in range(0, len(label), lrange):
                    lo, stitch_remap_lo = torch.where(label[lrangei:lrangei+lrange, None] == zstitch_remap[0, None, :])
                    loc.append(lo+lrangei)
                    stitch_remap_loc.append(stitch_remap_lo)
                loc, stitch_remap_loc = torch.cat(loc), torch.cat(stitch_remap_loc)

                ## pre_loc: gnn stitch target (previous tile) nis index, tloc: index of remaining Z stitch pairs after nis being removed by X-Y stitching
                pre_loc, tloc = [], []
                for lrangei in range(0, len(label), lrange):
                    pre_lo, tlo = torch.where(label[lrangei:lrangei+lrange, None] == zstitch_remap[1, None, stitch_remap_loc])
                    pre_loc.append(pre_lo+lrangei)
                    tloc.append(tlo)
                pre_loc, tloc = torch.cat(pre_loc), torch.cat(tloc)

                ## source nis is removed from keeping mask
                keep_mask = torch.ones(len(pt)).bool()
                keep_mask[loc] = False
            #     keep_masks[stack_name][f'{i}-{j}'] = torch.logical_and(keep_masks[stack_name][f'{i}-{j}'], keep_mask)

                # merge stitched source nis to target nis
                loc = loc[tloc]
                pt[pre_loc] = (pt[loc] + pt[pre_loc]) / 2
                vol[pre_loc] = vol[loc] + vol[pre_loc]

                pt = pt[keep_mask]
                vol = vol[keep_mask]
                label = label[keep_mask]
                print("after remove z-stitched pt", pt.shape, label.shape)
            ###########################

            ## random round the coordinates 
            # pt = pt + (torch.rand_like(pt) - 0.5)
            ##
            centers.append(pt.cpu())
            vols.append(vol.cpu())
            labels.append(label.cpu())
        

    centers = torch.cat(centers).cpu()
    vols = torch.cat(vols).cpu()
    # vols = torch.zeros(10)
    torch.cuda.empty_cache()
    print(datetime.now(), "Whole-brain nuclei counting", centers.shape[0])
    density_map, volavg_map = downsample(centers, vols, dratio, dshape, device, skip_vol=True)
    print(datetime.now(), f"Saving total density {ptag} {btag.split('_')[1]}, Max density {density_map.max()}")
    os.makedirs(f'{save_path}/whole_brain_map', exist_ok=True)
    if not os.path.exists(f'{save_path}/doubled_NIS_label/{btag.split("_")[1]}_doubled_label.zip'):
        nms_tag = "before"
    else:
        nms_tag = "after"
    
    # if tform_stack_manual is not None:
    #     nib.save(nib.Nifti1Image(density_map.numpy().astype(np.float64), affine_m, header=new_header), f'{save_path}/whole_brain_map/NIS_density_{nms_tag}_manual_{ptag}_{btag.split("_")[1]}.nii.gz')
    # else:
    if tform_stack_ptreg is not None:
        nib.save(nib.Nifti1Image(density_map.numpy().astype(np.float64), affine_m, header=new_header), f'{save_path}/whole_brain_map/NIS_density_{nms_tag}_rm2_{ptag}_{btag.split("_")[1]}.nii.gz')
    else:
        nib.save(nib.Nifti1Image(density_map.numpy().astype(np.float64), affine_m, header=new_header), f'{save_path}/whole_brain_map/NIS_density_{nms_tag}_rm_{ptag}_{btag.split("_")[1]}.nii.gz')

    # if tform_stack_manual is not None:
    #     nib.save(nib.Nifti1Image(volavg_map.numpy().astype(np.float64), affine_m, header=new_header), f'{save_path}/whole_brain_map/NIS_volavg_{nms_tag}_manual_{ptag}_{btag.split("_")[1]}.nii.gz')
    # else:
    #     if tform_stack_ptreg is not None:
    #         nib.save(nib.Nifti1Image(volavg_map.numpy().astype(np.float64), affine_m, header=new_header), f'{save_path}/whole_brain_map/NIS_volavg_{nms_tag}_rm2_{ptag}_{btag.split("_")[1]}.nii.gz')
    #     else:
    #         nib.save(nib.Nifti1Image(volavg_map.numpy().astype(np.float64), affine_m, header=new_header), f'{save_path}/whole_brain_map/NIS_volavg_{nms_tag}_rm_{ptag}_{btag.split("_")[1]}.nii.gz')
        
    
def posthoc_stitch_line(ptag, btag, device):
    '''
    Whole brain map
    '''
    overlap_r = OVERLAP_R
    new_header, affine_m = init_nib_header()

    downsample_res = [25, 25, 25]
    seg_res = [4, 0.75, 0.75]
    save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/{ptag}/{btag.split("_")[1]}'
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}/{btag}'
    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path) if 'Ultra' in fn])

    ncol, nrow = tile_loc.max(0)+1
    assert len(tile_loc) == nrow*ncol, f'tile of raw data is not complete, tile location: {tile_loc}'
    root = result_path + '/UltraII[%02d x %02d]'
    stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    stack_names = sort_stackname(stack_names)
    sshape = [0, 0, 0]
    for stack_name in stack_names:
        meta_name = stack_name.replace('instance_center', 'seg_meta')
        zstart = int(stack_name.split('zmin')[1].split('_')[0])
        zstart = int(zstart*zratio)
        seg_shape = torch.load(f'{root % (0, 0)}/{meta_name}')
        zend = int(zstart + seg_shape[0].item()*zratio)

    zstart = 0 

    tform_xy_max = [0.05*seg_shape[1], 0.05*seg_shape[2]]
    dratio = [s/d for s, d in zip(seg_res, downsample_res)]
    dshape = [0, 0, 0]
    dshape[0] = int(sshape[0] * dratio[0])
    dshape[1] = int(sshape[1] * dratio[1])
    dshape[2] = int(sshape[2] * dratio[2])
    print(datetime.now(), f"Downsample space shape: {dshape}, ratio: {dratio}, LS space shape {sshape}")

    tform_stack_coarse = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_coarse.json', 'r', encoding='utf-8'))
    tform_stack_refine = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine.json', 'r', encoding='utf-8'))
    print(len(tform_stack_refine))
    if os.path.exists(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine_ptreg.json'):
        tform_stack_ptreg = json.load(open(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine_ptreg.json', 'r', encoding='utf-8'))
        print(datetime.now(), "Loaded tform of coarse, refine and pt-reg", len(tform_stack_ptreg), [tform_stack_ptreg[zi].keys() for zi in range(len(tform_stack_ptreg)) if len(tform_stack_ptreg[zi].keys()) > 0][-1])
    else:
        tform_stack_ptreg = None
        print(datetime.now(), "Loaded tform of coarse, refine")

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
    tformed_tile_lt_loc_refined = {zi: copy.deepcopy(tformed_tile_lt_loc[0]) for zi in range(zstart, zend)}
    for zi in range(zstart, zend):
        for k in tform_stack_refine[zi]:
            tx, ty = tform_stack_refine[zi][k]
            tformed_tile_lt_loc_refined[zi][k][0] = tformed_tile_lt_loc_refined[zi][k][0] + tx
            tformed_tile_lt_loc_refined[zi][k][1] = tformed_tile_lt_loc_refined[zi][k][1] + ty
            if tform_stack_ptreg is not None:
                if k in tform_stack_ptreg[zi]:
                    tx, ty = tform_stack_ptreg[zi][k]
                    tformed_tile_lt_loc_refined[zi][k][0] = tformed_tile_lt_loc_refined[zi][k][0] + tx
                    tformed_tile_lt_loc_refined[zi][k][1] = tformed_tile_lt_loc_refined[zi][k][1] + ty


    if not os.path.exists(f'{save_path}/doubled_NIS_label/{btag.split("_")[1]}_doubled_label.zip'):
        nms_tag = "before"
    else:
        nms_tag = "after"

    if tform_stack_ptreg is not None:
        density_map = torch.from_numpy(np.array(nib.load(f'{save_path}/whole_brain_map/NIS_density_{nms_tag}_rm2_{ptag}_{btag.split("_")[1]}.nii.gz').get_fdata())).float()
    else:
        density_map = torch.from_numpy(np.array(nib.load(f'{save_path}/whole_brain_map/NIS_density_{nms_tag}_rm_{ptag}_{btag.split("_")[1]}.nii.gz').get_fdata())).float()

    stitch_line_mask_dim1 = torch.zeros_like(density_map).bool()
    stitch_line_mask_dim2 = torch.zeros_like(density_map).bool()
    ra = 1
    re = ra+1
    new_density_map = density_map.clone()
    for zi in tformed_tile_lt_loc_refined:
        for k in tformed_tile_lt_loc_refined[zi]:
            dz = min(int(zi * dratio[0]), stitch_line_mask_dim1.shape[0]-1)
            i, j = tformed_tile_lt_loc_refined[zi][k]
            iranges = [
                [i, i+seg_shape[1]], 
            ]
            
            jranges = [
                [j, j+seg_shape[2]], 
            ]

            for istart, iend in iranges:
                istart = min(int(istart*dratio[1]), stitch_line_mask_dim1.shape[1]-re)
                iend = min(int(iend*dratio[1]), stitch_line_mask_dim1.shape[1]-re)
                istart = max(istart, re)
                iend = max(iend, re)
                jstart0 = j
                jend0 = j + seg_shape[2]
                jstart0 = min(int(jstart0*dratio[2]), stitch_line_mask_dim1.shape[2]-re)
                jend0 = min(int(jend0*dratio[2]), stitch_line_mask_dim1.shape[2]-re)
                stitch_line_mask_dim1[dz, istart:istart+ra, jstart0:jend0] = True
                stitch_line_mask_dim1[dz, iend-ra:iend, jstart0:jend0] = True
                # new_density_map[dz, istart:istart+ra, jstart0:jend0] = density_map[dz, istart-ra:istart, jstart0:jend0].flip([0])
                # new_density_map[dz, iend-ra:iend, jstart0:jend0] = density_map[dz, iend:iend+ra, jstart0:jend0].flip([0])

            for jstart, jend in jranges:
                jstart = min(int(jstart*dratio[2]), stitch_line_mask_dim1.shape[2]-re)
                jend = min(int(jend*dratio[2]), stitch_line_mask_dim1.shape[2]-re)
                jstart = max(jstart, re)
                jend = max(jend, re)
                istart0 = i
                iend0 = i + seg_shape[1]
                istart0 = min(int(istart0*dratio[1]), stitch_line_mask_dim1.shape[1]-re)
                iend0 = min(int(iend0*dratio[1]), stitch_line_mask_dim1.shape[1]-re)
                stitch_line_mask_dim2[dz, istart0:iend0, jstart:jstart+ra] = True
                stitch_line_mask_dim2[dz, istart0:iend0, jend-ra:jend] = True
                # new_density_map[dz, istart0:iend0, jstart:jstart+ra] = density_map[dz, istart0:iend0, jstart-ra:jstart].flip([1])
                # new_density_map[dz, istart0:iend0, jend-ra:jend] = density_map[dz, istart0:iend0, jend:jend+ra].flip([1])


    new_density_map_dim1 = density_map.clone().transpose(1, 2)
    new_density_map_dim2 = density_map.clone()
    kernel_size = 3
    p = 2
    # pooler = median_pooling_1d(kernel_size, stride=1, padding=kernel_size//2)
    # pooler = torch.nn.LPPool1d(p, kernel_size, stride=1)
    # padder = torch.nn.ReflectionPad1d(kernel_size//2)
    for _ in range(1):
        # new_density_map = pooler(new_density_map[None])[0]
        # new_density_map = new_density_map / ((kernel_size**2)**(1/p))
        new_density_map_dim1 = median_pooling_1d(new_density_map_dim1, kernel_size, stride=1, padding=kernel_size//2)
        new_density_map_dim2 = median_pooling_1d(new_density_map_dim2, kernel_size, stride=1, padding=kernel_size//2)
        # new_density_map_dim1 = pooler(new_density_map_dim1)
        # new_density_map_dim1 = padder(new_density_map_dim1)
        # new_density_map_dim1 = new_density_map_dim1 / ((kernel_size)**(1/p))
        # new_density_map_dim2 = pooler(new_density_map_dim2)
        # new_density_map_dim2 = padder(new_density_map_dim2)
        # new_density_map_dim2 = new_density_map_dim2 / ((kernel_size)**(1/p))
    
    new_density_map_dim1 = new_density_map_dim1.transpose(1, 2)
    new_density_map[stitch_line_mask_dim2] = new_density_map_dim1[stitch_line_mask_dim2]
    new_density_map[stitch_line_mask_dim1] = new_density_map_dim2[stitch_line_mask_dim1]
    # org_mask = torch.logical_not(stitch_line_mask)
    # new_density_map[org_mask] = density_map[org_mask]

    # nib.save(nib.Nifti1Image(stitch_line_mask.numpy().astype(np.float64), affine_m, header=new_header), f'{save_path}/whole_brain_map/NIS_density_posthoc-mask_{ptag}_{btag.split("_")[1]}.nii.gz')
    if tform_stack_ptreg is not None:
        nib.save(nib.Nifti1Image(new_density_map.numpy().astype(np.float64), affine_m, header=new_header), f'{save_path}/whole_brain_map/NIS_density_{nms_tag}_posthoc_rm2_{ptag}_{btag.split("_")[1]}.nii.gz')
    else:
        nib.save(nib.Nifti1Image(new_density_map.numpy().astype(np.float64), affine_m, header=new_header), f'{save_path}/whole_brain_map/NIS_density_{nms_tag}_posthoc_rm_{ptag}_{btag.split("_")[1]}.nii.gz')
        


def bbox_in_stitching_seam(bbox, lt_loc, wh, i, j, ncol, nrow, nms_r):
    # mask = [left, right, bottom, top]
    masks = [None, None, None, None]
    if i > 0:
        # 0 ~ overlap_r
        mask = bbox[:, 1] < lt_loc[0] + wh[0]*(nms_r)
        masks[0] = mask

    if i < ncol-1:
        # 1-overlap_r ~ 1
        mask = bbox[:, 4] > lt_loc[0] + wh[0]*(1-nms_r)
        masks[1] = mask

    if j > 0:
        # 0 ~ overlap_r
        mask = bbox[:, 2] < lt_loc[1] + wh[1]*nms_r
        masks[2] = mask

    if j < nrow-1:
        # 1-overlap_r ~ 1
        mask = bbox[:, 5] > lt_loc[1] + wh[1]*(1-nms_r)
        masks[3] = mask

    return masks

def nms_bbox(bbox_tgt, bbox_mov, iou_threshold=0.1, tile_tgt_center=None, tile_mov_center=None, seg_shape=None, device=None):
    # remove touching boundary boxes first
    rm_mask_tgt = (bbox_tgt[:, 1] == 0) | (bbox_tgt[:, 1] == seg_shape[0]) | (bbox_tgt[:, 4] == 0) | (bbox_tgt[:, 4] == seg_shape[0]) | \
    (bbox_tgt[:, 2] == 0) | (bbox_tgt[:, 2] == seg_shape[1]) | (bbox_tgt[:, 5] == 0) | (bbox_tgt[:, 5] == seg_shape[1])
    rm_mask_mov = (bbox_mov[:, 1] == 0) | (bbox_mov[:, 1] == seg_shape[0]) | (bbox_mov[:, 4] == 0) | (bbox_mov[:, 4] == seg_shape[0]) | \
    (bbox_mov[:, 2] == 0) | (bbox_mov[:, 2] == seg_shape[1]) | (bbox_mov[:, 5] == 0) | (bbox_mov[:, 5] == seg_shape[1])
    remain_id_tgt = torch.where(~rm_mask_tgt)[0]
    remain_id_mov = torch.where(~rm_mask_mov)[0]
    bbox_tgt = bbox_tgt[remain_id_tgt]
    bbox_mov = bbox_mov[remain_id_mov]
    # distance to tile center
    tgt_cx = (bbox_tgt[:, 1] + bbox_tgt[:, 4]) / 2
    tgt_cy = (bbox_tgt[:, 2] + bbox_tgt[:, 5]) / 2
    tgt_dis_to_tctr = ((tgt_cx - tile_tgt_center[0])**2 + (tgt_cy - tile_tgt_center[1])**2).sqrt()
    tgt_dis_to_tctr = (tgt_dis_to_tctr - tgt_dis_to_tctr.min()) / (tgt_dis_to_tctr.max() - tgt_dis_to_tctr.min())
    mov_cx = (bbox_mov[:, 1] + bbox_mov[:, 4]) / 2
    mov_cy = (bbox_mov[:, 2] + bbox_mov[:, 5]) / 2
    mov_dis_to_tctr = ((mov_cx - tile_mov_center[0])**2 + (mov_cy - tile_mov_center[1])**2).sqrt()
    mov_dis_to_tctr = (mov_dis_to_tctr - mov_dis_to_tctr.min()) / (mov_dis_to_tctr.max() - mov_dis_to_tctr.min())
    # compute iou
    D = int(bbox_mov.shape[1]/2)
    area_tgt = box_area(bbox_tgt, D)
    area_mov = box_area(bbox_mov, D)
    iou, index = box_iou(bbox_tgt, bbox_mov, D, area1=area_tgt, area2=area_mov)
    ## scatter max among each of mov bbox (use this)
    max_iou, argmax = scatter_max(iou, index[1])
    valid = max_iou>0
    ## scatter max among each of tgt bbox
    # max_iou, argmax = scatter_max(iou, index[0])
    ## max iou larger than threshold
    ## adaptive threshold based on distance to tile center
    threshold_e = mov_dis_to_tctr[index[1, argmax[valid]]] * tgt_dis_to_tctr[index[0, argmax[valid]]]
    # threshold_e = -1 * threshold_e
    threshold_e = (threshold_e - threshold_e.min()) / (threshold_e.max()- threshold_e.min())
    threshold_e = threshold_e.clip(min=0.05, max=0.5)
    iou_threshold = iou_threshold * threshold_e
    big_iou = max_iou[valid] >= iou_threshold
    remove_ind_mov = index[1, argmax[valid][big_iou]]
    remove_ind_tgt = index[0, argmax[valid][big_iou]]
    ## remove small area of tgt or mov bbox
    # remove_area_tgt = area_tgt[remove_ind_tgt] 
    # remove_area_mov = area_mov[remove_ind_mov] 
    # remove_ind_tgt = remove_ind_tgt[remove_area_tgt < remove_area_mov].unique()
    # remove_ind_mov = remove_ind_mov[remove_area_mov <= remove_area_tgt].unique()
    ## remove tgt or mov bbox randomly
    # rand_choose = torch.rand(len(remove_ind_tgt)) >= 0.5
    # remove_ind_tgt = remove_ind_tgt[rand_choose]
    # remove_ind_mov = remove_ind_mov[torch.logical_not(rand_choose)]
    ## adaptive randomness of removing bbox based on the distance to tile center
    tgt_cx = (bbox_tgt[remove_ind_tgt, 1] + bbox_tgt[remove_ind_tgt, 4]) / 2
    tgt_cy = (bbox_tgt[remove_ind_tgt, 2] + bbox_tgt[remove_ind_tgt, 5]) / 2
    dis_to_tctr = ((tgt_cx - tile_tgt_center[0])**2 + (tgt_cy - tile_tgt_center[1])**2).sqrt()
    dis_to_tctr = (dis_to_tctr - dis_to_tctr.min()) / (dis_to_tctr.max() - dis_to_tctr.min())
    # print(dis_to_tctr.max(), dis_to_tctr.min())
    rand_choose = torch.rand(len(remove_ind_tgt)).to(device) >= dis_to_tctr
    remove_ind_tgt = remove_ind_tgt[rand_choose]
    remove_ind_mov = remove_ind_mov[torch.logical_not(rand_choose)]
    ## use original index
    remove_ind_tgt = torch.cat([remain_id_tgt[remove_ind_tgt], torch.where(rm_mask_tgt)[0]])
    remove_ind_mov = torch.cat([remain_id_mov[remove_ind_mov], torch.where(rm_mask_mov)[0]])
    return remove_ind_tgt, remove_ind_mov

def box_iou(boxes1, boxes2, D=2, area1=None, area2=None):
    if area1 is None:
        area1 = box_area(boxes1, D)
    if area2 is None:
        area2 = box_area(boxes2, D)
    index = []
    inter = []
    lrange = 100
    if lrange*boxes2.shape[0] >= 2147483647: # INT_MAX
        lrange = int(2147483647 / boxes2.shape[0])
    # for i in trange(0, len(boxes1), lrange, desc=f'Compute IoU between {boxes1.shape} and {boxes2.shape} boxes'):
    for i in range(0, len(boxes1), lrange):
        lt = torch.max(boxes1[i:i+lrange, None, :D], boxes2[:, :D])  # [N,M,D]
        rb = torch.min(boxes1[i:i+lrange, None, D:], boxes2[:, D:])  # [N,M,D]
        ind1, ind2 = torch.where((rb > lt).all(dim=-1))
        # if len(ind1) == 0: continue
        wh = _upcast(rb[ind1, ind2] - lt[ind1, ind2])
        assert (wh>0).all()
        inter.append(wh.cumprod(-1)[..., -1]) # [N*M]
        ind1 = ind1 + i
        index.append(torch.stack([ind1, ind2])) # [2, N*M]
    
    inter = torch.cat(inter)
    index = torch.cat(index, 1)
    union = area1[index[0]] + area2[index[1]] - inter
    iou = inter / union
    return iou, index

def box_area(bbox, D):
    bbox = _upcast(bbox)
    wh = bbox[:, D:] - bbox[:, :D]
    return torch.cumprod(wh, dim=1)[:, -1]


def downsample(center, vol, ratio, dshape, device, skip_vol=False):
    center = center.clone().to(device)
    if vol is not None:
        vol = vol.float().to(device)
    center[:,0] = center[:,0] * ratio[0]
    center[:,1] = center[:,1] * ratio[1]
    center[:,2] = center[:,2] * ratio[2]
    dshape = [max(int(dshape[0]*ratio[0]), int(center[:,0].max()+2)), max(int(dshape[1]*ratio[1]), int(center[:,1].max()+2)), max(int(dshape[2]*ratio[2]), int(center[:,2].max()+2))]
    # dshape = [int(center[:,0].max()+1), int(center[:,1].max()+1), int(center[:,2].max()+1)]
    # outbound_mask = torch.logical_or(center[:, 0].round() > dshape[0]-1, center[:, 1].round() > dshape[1]-1)
    # outbound_mask = torch.logical_or(outbound_mask, center[:, 2].round() > dshape[2]-1)
    # center = center[torch.logical_not(outbound_mask)]
    z = center[:, 0].clip(min=0)
    y = center[:, 1].clip(min=0)
    x = center[:, 2].clip(min=0)
    # print(center.shape, z, x, y)
    loc = torch.arange(dshape[0]*dshape[1]*dshape[2]).view(dshape[0], dshape[1], dshape[2]).to(device) 
    loc = loc[(z.round().long(), y.round().long(), x.round().long())] # all nis location in the downsample space
    loc_count = loc.bincount() 
    loc_count = loc_count[loc_count!=0] 
    atlas_loc = loc.unique().to(device) # unique location in the downsample space
    ## volume avg & local intensity
    vol_avg = None
    if not skip_vol:
        loc_argsort = loc.argsort().cpu()
        loc_splits = loc_count.cumsum(0).cpu()
        loc_vol = torch.tensor_split(vol[loc_argsort], loc_splits)
        assert len(loc_vol[-1]) == 0
        loc_vol = loc_vol[:-1]
        loc_vol = torch.nn.utils.rnn.pad_sequence(loc_vol, batch_first=True, padding_value=-1)
        loc_fg = loc_vol!=-1
        loc_num = loc_fg.sum(1)
        loc_vol[loc_vol==-1] = 0
        vol_avg = torch.zeros(dshape[0]*dshape[1]*dshape[2]).float()#.to(device)
        vol_avg[atlas_loc] = (loc_vol.sum(1) / loc_num).cpu().float()
        # for loci in tqdm(atlas_loc, desc="Collect NIS property in local cube"): 
        #     where_loc = torch.where(loc==loci)[0]
        #     vol_avg[loci] = vol[where_loc].mean()
        vol_avg = vol_avg.view(dshape[0], dshape[1], dshape[2])#.cpu()
    ## density map
    density = torch.zeros(dshape[0]*dshape[1]*dshape[2], dtype=torch.float64).to(device)
    density[atlas_loc] = loc_count.double() #/ center.shape[0]
    density = density.view(dshape[0], dshape[1], dshape[2]).cpu()
    return density, vol_avg


def init_nib_header():
    mask_fn = "/lichtman/Felix/Lightsheet/P4/pair15/output_L73D766P4/registered/L73D766P4_MASK_topro_25_all.nii"
    new_header = nib.load(mask_fn).header
    new_header['quatern_b'] = 0.5
    new_header['quatern_c'] = -0.5
    new_header['quatern_d'] = 0.5
    new_header['qoffset_x'] = -0.0
    new_header['qoffset_y'] = -0.0
    new_header['qoffset_z'] = 0.0
    affine_m = np.eye(4, 4)
    affine_m[:3, :3] = 0
    affine_m[0, 1] = -1
    affine_m[1, 2] = -1
    affine_m[2, 0] = 1
    return new_header, affine_m

def sort_stackname(stack_names):
    stack_z = []
    for stack_name in stack_names:
        stack_z.append(int(stack_name.split('zmin')[1].split('_')[0]))

    argsort = np.argsort(stack_z)
    return [stack_names[i] for i in argsort]

import torch.nn as nn
import torch.nn.functional as F

class MedianPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MedianPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        x = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        x = x.contiguous().view(x.size()[:4] + (-1,))
        x, _ = torch.median(x, dim=-1)
        return x
    
def median_pooling_1d(input, kernel_size, stride=1, padding=0):
    """
    Performs median pooling for 1D data in PyTorch.

    Args:
        input: Input tensor of shape (batch_size, channels, length).
        kernel_size: Size of the pooling window.
        stride: Stride of the pooling operation.
        padding: Padding to add to the input tensor.

    Returns:
        Output tensor of shape (batch_size, channels, output_length).
    """

    batch_size, channels, length = input.shape
    output_length = (length + 2 * padding - kernel_size) // stride + 1

    output = torch.zeros(batch_size, channels, output_length)

    for i in range(output_length):
        start = i * stride
        end = start + kernel_size
        input_chunk = input[:, :, start:end]
        output[:, :, i] = torch.median(input_chunk, dim=-1)[0]

    return output
    
if __name__ == '__main__': main()