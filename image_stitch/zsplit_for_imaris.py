import os
from tqdm import tqdm


brain_ready = [  
    # ['female', '230916_L92P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_10-23-53'],
    # ['female', '230915_L92P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_19-40-49'],
    # ['female', '230907_L88P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_18-15-15'],
    # ['female', '231020_L95P4_sox9toproneun_sw50_4x_df9_4z_15ov_50ex_21-45-35'],
    # ['female', '231028_L106P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_14-48-05'],
    # ['female', '231027_L102P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-10-58'],
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
    ['male', '230824_L92P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_15-55-47'],
    # ['extrabrains', '240614_L97P1_sox990topro40neun40_sw50_4x_df9_4z_ov15_ex50_16-44-49'],
    # ['extrabrains', '240615_L97P2_sox990topro40neun40_sw50_4x_df9_4z_ov15_ex50_08-39-12'],
]
for ptag, btag in brain_ready:
    z_portion = 50
    r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P14/{ptag}/{btag}'
    save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P14_{z_portion}chunk/{ptag}/{btag}'

    for d in tqdm(sorted(os.listdir(r))):
        if 'Ultra' not in d: continue
        if '.' in d: continue
        for fn in sorted(os.listdir(f'{r}/{d}')):
            if not fn.endswith('.ome.tif'): continue
            z = int(fn.split('_')[1])
            if z % z_portion == 0:
                save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
                os.makedirs(save_d, exist_ok=True)
            os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')


    z_portion = 100
    r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P14/{ptag}/{btag}'
    save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P14_{z_portion}chunk/{ptag}/{btag}'

    for d in tqdm(sorted(os.listdir(r))):
        if 'Ultra' not in d: continue
        if '.' in d: continue
        for fn in sorted(os.listdir(f'{r}/{d}')):
            if not fn.endswith('.ome.tif'): continue
            z = int(fn.split('_')[1])
            if z % z_portion == 0:
                save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
                os.makedirs(save_d, exist_ok=True)
            os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')



# ptag = 'pair5'
# btag = '220423_L57D855P5_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_09-02-27'
# r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}/{btag}'
# save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P4/{ptag}/{btag}'

# z_portion = 50
# for d in tqdm(sorted(os.listdir(r))):
#     if 'Ultra' not in d: continue
#     if '.' in d: continue
#     for fn in sorted(os.listdir(f'{r}/{d}')):
#         if not fn.endswith('.ome.tif'): continue
#         z = int(fn.split('_')[1])
#         if z % z_portion == 0:
#             save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
#             os.makedirs(save_d, exist_ok=True)
#         os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')


# ptag = 'pair5'
# btag = '220422_L57D855P4_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_16-54-32'
# r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}/{btag}'
# save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P4/{ptag}/{btag}'

# z_portion = 50
# for d in tqdm(sorted(os.listdir(r))):
#     if 'Ultra' not in d: continue
#     for fn in sorted(os.listdir(f'{r}/{d}')):
#         if not fn.endswith('.ome.tif'): continue
#         z = int(fn.split('_')[1])
#         if z % z_portion == 0:
#             save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
#             os.makedirs(save_d, exist_ok=True)
#         os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')


# ptag = 'pair8'
# btag = '220430_L59D878P5_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_13-03-21'
# r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}/{btag}'
# save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P4/{ptag}/{btag}'

# z_portion = 50
# for d in tqdm(sorted(os.listdir(r))):
#     if 'Ultra' not in d: continue
#     for fn in sorted(os.listdir(f'{r}/{d}')):
#         if not fn.endswith('.ome.tif'): continue
#         z = int(fn.split('_')[1])
#         if z % z_portion == 0:
#             save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
#             os.makedirs(save_d, exist_ok=True)
#         os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')


# ptag = 'pair11'
# btag = '220704_L66D764P8_OUT_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_10-46-01'
# r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}/{btag}'
# save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P4/{ptag}/{btag}'

# z_portion = 50
# for d in tqdm(sorted(os.listdir(r))):
#     if 'Ultra' not in d: continue
#     for fn in sorted(os.listdir(f'{r}/{d}')):
#         if not fn.endswith('.ome.tif'): continue
#         z = int(fn.split('_')[1])
#         if z % z_portion == 0:
#             save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
#             os.makedirs(save_d, exist_ok=True)
#         os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')


# ptag = 'pair6'
# btag = '220416_L57D855P2_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_09-52-07'
# r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}/{btag}'
# save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P4/{ptag}/{btag}'

# z_portion = 50
# for d in tqdm(sorted(os.listdir(r))):
#     if 'Ultra' not in d: continue
#     for fn in sorted(os.listdir(f'{r}/{d}')):
#         if not fn.endswith('.ome.tif'): continue
#         z = int(fn.split('_')[1])
#         if z % z_portion == 0:
#             save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
#             os.makedirs(save_d, exist_ok=True)
#         os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')


# ptag = 'pair9'
# btag = '220702_L64D804P4_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_20ov_09-53-19'
# r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}/{btag}'
# save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P4/{ptag}/{btag}'

# z_portion = 50
# for d in tqdm(sorted(os.listdir(r))):
#     if 'Ultra' not in d: continue
#     for fn in sorted(os.listdir(f'{r}/{d}')):
#         if not fn.endswith('.ome.tif'): continue
#         z = int(fn.split('_')[1])
#         if z % z_portion == 0:
#             save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
#             os.makedirs(save_d, exist_ok=True)
#         os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')


# ptag = 'pair22'
# btag = '220928_L91D814P3_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_18-03-23'
# r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}/{btag}'
# save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P4/{ptag}/{btag}'

# z_portion = 50
# for d in tqdm(sorted(os.listdir(r))):
#     if 'Ultra' not in d: continue
#     for fn in sorted(os.listdir(f'{r}/{d}')):
#         if not fn.endswith('.ome.tif'): continue
#         z = int(fn.split('_')[1])
#         if z % z_portion == 0:
#             save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
#             os.makedirs(save_d, exist_ok=True)
#         os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')


# ptag = 'pair16'
# btag = '220805_L74D769P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_15-51-36'
# r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}/{btag}'
# save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P4/{ptag}/{btag}'

# z_portion = 50
# for d in tqdm(sorted(os.listdir(r))):
#     if 'Ultra' not in d: continue
#     for fn in sorted(os.listdir(f'{r}/{d}')):
#         if not fn.endswith('.ome.tif'): continue
#         z = int(fn.split('_')[1])
#         if z % z_portion == 0:
#             save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
#             os.makedirs(save_d, exist_ok=True)
#         os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')



# ptag = 'pair13'
# btag = '220827_L69D764P6_OUT_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-47-13'
# r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}/{btag}'
# save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P4/{ptag}/{btag}'

# z_portion = 50
# for d in tqdm(sorted(os.listdir(r))):
#     if 'Ultra' not in d: continue
#     for fn in sorted(os.listdir(f'{r}/{d}')):
#         if not fn.endswith('.ome.tif'): continue
#         z = int(fn.split('_')[1])
#         if z % z_portion == 0:
#             save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
#             os.makedirs(save_d, exist_ok=True)
#         os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')



# ptag = 'pair18'
# btag = '220809_L77D764P8_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_09-52-21'
# r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}/{btag}'
# save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P4/{ptag}/{btag}'

# z_portion = 50
# for d in tqdm(sorted(os.listdir(r))):
#     if 'Ultra' not in d: continue
#     for fn in sorted(os.listdir(f'{r}/{d}')):
#         if not fn.endswith('.ome.tif'): continue
#         z = int(fn.split('_')[1])
#         if z % z_portion == 0:
#             save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
#             os.makedirs(save_d, exist_ok=True)
#         os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')


# ptag = 'pair14'
# btag = '220722_L73D766P5_OUT_topro_brn2_ctip2_4x_50sw_0_108na_11hdf_4z_20ov_16-49-30'
# r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}/{btag}'
# save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P4/{ptag}/{btag}'

# z_portion = 50
# for d in tqdm(sorted(os.listdir(r))):
#     if 'Ultra' not in d: continue
#     for fn in sorted(os.listdir(f'{r}/{d}')):
#         if not fn.endswith('.ome.tif'): continue
#         z = int(fn.split('_')[1])
#         if z % z_portion == 0:
#             save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
#             os.makedirs(save_d, exist_ok=True)
#         os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')

