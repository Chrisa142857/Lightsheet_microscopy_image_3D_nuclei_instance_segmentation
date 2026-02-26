from phase_correlation_stitch import get_stitch_tform
import os

brain_ready = [  
    # ['female', '230916_L92P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_10-23-53'],
    # ['female', '230915_L92P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_19-40-49'],
    # ['female', '230907_L88P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_18-15-15'],
    # ['female', '231020_L95P4_sox9toproneun_sw50_4x_df9_4z_15ov_50ex_21-45-35'],
    # ['female', '231028_L106P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_14-48-05'],
    ['female', '231027_L102P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-10-58'],
    # ['female', '230917_L94P3sox9toproneun_sw50_4x_df9_4z_ov15_ex50_10-51-40'],
    # ['female', '230902_L86P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-25-11'],
    # ['female', '231019_L95P3_sox9toproneun_sw50_4x_df9_4z_15ov_50ex_10-14-35'],
    # ['female', '230902_L86P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_22-54-23'],
    # ['female', '230909_L90P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_22-29-59'],
    # ['female', '230908_L88P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_10-33-21'],
    # ['female', '231026_L102P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-39-32'],
    # ['female', '231028_L106P5_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_01-41-16'],
    # ['female', '230909_L90P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-53-39'],
    # ['female', '231021_L96P3_sox9toproneun_sw50_4x_df9_4z_15ov_50ex_11-29-29'],
    # ['female', '230917_L94P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_22-33-33'],
    # ['female', '231020_L96P4_sox9toproneun_sw50_4x_df9_4z_15ov_50ex_11-32-52'],
    # ['male', '230825_L92P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_08-59-16'],
    # ['male', '230720_L72D759P3_topro40sox990neun50_4x_df9_4z_ov15_ext50_12-14-18'],
    # ['male', '230812_L82D711P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_10-39-39'],
    # ['male', '230819_L87P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_09-23-37'],
    # ['male', '230811_L68D767P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_12-13-51'],
    # ['male', '230810_L68P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_15-49-53'],
    # ['male', '230813_L82D711P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_23-59-14'],
    # ['male', '230818_L87D868P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_15-26-37'],
    # ['male', '230805_L72P3_topro40sox990neun50_4x_df10_4z_ov15_ext50_19-02-07'],
    # ['male', '230820_L88P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_21-23-23'],
    # ['male', '230826_L94P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_19-04-29'],
    # ['male', '230820_L88P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-00-37'],
    # ['male', '230826_L94P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_23-30-15'],
    # ['male', '230901_L95P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-31-57'],
    # ['extrabrains', '240614_L97P1_sox990topro40neun40_sw50_4x_df9_4z_ov15_ex50_16-44-49'],
    # ['extrabrains', '240615_L97P2_sox990topro40neun40_sw50_4x_df9_4z_ov15_ex50_08-39-12'],
]
# done = ['L95P2']
for ptag, btag in brain_ready:
    ls_image_root = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P14/{ptag}/{btag}'             # path to raw image 
    save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/P14/{ptag}/{btag.split("_")[1]}'       # path to saving folder
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P14/{ptag}/{btag}'                           # path to NIS results
    root = result_path + '/UltraII[%02d x %02d]'
    if not os.path.exists(root % (0, 0)): continue
    stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    if len(stack_names) == 0: 
        print(ptag, btag.split('_')[1], "NIS not completed, skip")
        continue
    print("=== PCC stitch ", ptag, btag.split('_')[1], "===")
    # if not os.path.exists(f'{save_path}/NIS_tranform/{btag.split("_")[1]}_tform_refine.json'):
    #     get_stitch_tform(ptag, btag, ls_image_root, save_path, result_path)
    get_stitch_tform(ptag, btag, ls_image_root, save_path, result_path)