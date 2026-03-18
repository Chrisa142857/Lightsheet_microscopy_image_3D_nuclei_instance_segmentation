ptags = ['male', 'male', 'female', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'female', 'extrabrains', 'extrabrains', 'female', 'female']
btags = ['L82D711P2', 'L82D711P3', 'L86P4', 'L87D868P2', 'L87P1', 'L88P1', 'L88P2', 'L88P3', 'L88P4', 'L90P3', 'L92P2', 'L92P3', 'L92P4', 'L94P1', 'L94P3sox9toproneun', 'L94P4', 'L95P2', 'L95P3', 'L95P4', 'L96P3', 'L97P1', 'L97P2', 'L106P3', 'L106P5']
# skipb = ['L87P1', 'L88P1', 'L88P3', 'L88P4', 'L90P3', 'L88P2', 'L82D711P3', 'L92P2', 'L97P1', 'L97P2']
# btags = [b for b in btags if b not in skipb]

btag2b = {
    'L86P3': '230902_L86P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-25-11',
    'L86P4': '230902_L86P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_22-54-23',
    'L106P3': '231028_L106P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_14-48-05',
    'L106P5': '231028_L106P5_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_01-41-16',
    'L102P1': '231026_L102P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-39-32',
    'L102P2': '231027_L102P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-10-58',
    'L94P1': '230826_L94P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_19-04-29',
    'L88P1': '230820_L88P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-00-37',
    'L92P2': '230825_L92P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_08-59-16',
    'L94P2': '230826_L94P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_23-30-15',
    'L88P2': '230820_L88P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_21-23-23',
    'L87D868P2': '230818_L87D868P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_15-26-37',
    'L82D711P3': '230813_L82D711P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_23-59-14',
    'L94P4': '230917_L94P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_22-33-33',
    'L68D767P4': '230811_L68D767P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_12-13-51',
    'L68P3': '230810_L68P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_15-49-53',
    'L82D711P2': '230812_L82D711P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_10-39-39',
    'L87P1': '230819_L87P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_09-23-37',
    'L96P3': '231021_L96P3_sox9toproneun_sw50_4x_df9_4z_15ov_50ex_11-29-29',
    'L97P1': '240614_L97P1_sox990topro40neun40_sw50_4x_df9_4z_ov15_ex50_16-44-49',
    'L97P2': '240615_L97P2_sox990topro40neun40_sw50_4x_df9_4z_ov15_ex50_08-39-12',
    'L88P1': '230820_L88P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-00-37',
    'L88P3': '230907_L88P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_18-15-15',
    'L90P3': '230909_L90P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-53-39',
    'L90P4': '230909_L90P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_22-29-59',
    'L95P2': '230901_L95P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-31-57',
    'L92P3': '230915_L92P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_19-40-49',
    'L92P4': '230916_L92P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_10-23-53',
    'L94P3sox9toproneun': '230917_L94P3sox9toproneun_sw50_4x_df9_4z_ov15_ex50_10-51-40',
    'L88P4': '230908_L88P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_10-33-21',
    'L95P3': '231019_L95P3_sox9toproneun_sw50_4x_df9_4z_15ov_50ex_10-14-35',
    'L95P4': '231020_L95P4_sox9toproneun_sw50_4x_df9_4z_15ov_50ex_21-45-35'
}
import os
import numpy as np
cmds = []
basecmd = '/ram/USERS/ziquanw/softwares/miniconda3/envs/wholeBrain/bin/python /ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/analysis_nis_shape.py --gtag P14 --ptag %s --btag %s --ttag %d-%d'
for ptag, btag in zip(ptags, btags):
    b = btag2b[btag]
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P14/{ptag}/{b}'
    result_root = result_path + '/UltraII[%02d x %02d]'
    for i in range(4):
        for j in range(4):
            stack_names = os.listdir(result_root % (i, j))
            stack_names = [n for n in stack_names if 'instance_center' in n]
            tgt_names = [f"{result_root % (i, j)}/{n.replace('instance_center', 'instance_pa')}".replace('cajal', 'scheibel') for n in stack_names]
            ifexist = [os.path.exists(n) for n in tgt_names]
            if not np.all(ifexist):
                # print(np.array(stack_names)[~np.array(ifexist)])
                # print(np.array(tgt_names)[~np.array(ifexist)])
                cmds.append(f"{basecmd % (ptag, btag, i, j)}")
            
print('\n'.join(cmds))
