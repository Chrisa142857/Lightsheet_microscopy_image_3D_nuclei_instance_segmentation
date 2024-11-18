import os
from tqdm import tqdm

# ptag = 'pair5'
# btag = '220423_L57D855P5_topro_ctip2_brn2_4x_0_108na_50sw_11hdf_4z_15ov_09-02-27'
# r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}/{btag}'
# save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P4/{ptag}/{btag}'

# z_portion = 250
# for d in tqdm(sorted(os.listdir(r))):
#     if 'Ultra' not in d: continue
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


ptag = 'pair16'
btag = '220805_L74D769P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_15-51-36'
r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}/{btag}'
save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_imaris_stitch/P4/{ptag}/{btag}'

z_portion = 50
for d in tqdm(sorted(os.listdir(r))):
    if 'Ultra' not in d: continue
    for fn in sorted(os.listdir(f'{r}/{d}')):
        if not fn.endswith('.ome.tif'): continue
        z = int(fn.split('_')[1])
        if z % z_portion == 0:
            save_d = f'{save_r}/Z{z:04d}-Z{z+z_portion:04d}/{d}'
            os.makedirs(save_d, exist_ok=True)
        os.symlink(f'{r}/{d}/{fn}', f'{save_d}/{fn}')

