from ptreg_stitch import stitch_by_ptreg
device = 'cuda:1'
stitch_tile_ij_lst = [
    [[i, j] for i in range(4) for j in range(4)],
    [[i, j] for i in range(4) for j in range(4)],
    [[i, j] for i in range(4) for j in range(4)],
    [[i, j] for i in range(4) for j in range(4)],
    [[i, j] for i in range(4) for j in range(4)],
    [[i, j] for i in range(4) for j in range(4)],
    [[i, j] for i in range(4) for j in range(4)],
]
stitch_slice_ranges_lst = [
    [[0, 1] for i in range(4) for j in range(4)],
    [[0, 1] for i in range(4) for j in range(4)],
    [[0, 1] for i in range(4) for j in range(4)],
    [[0, 1] for i in range(4) for j in range(4)],
    [[0, 1] for i in range(4) for j in range(4)],
    [[0, 1] for i in range(4) for j in range(4)],
    [[0, 1] for i in range(4) for j in range(4)],
]
ptags = [
    'female',
    'female',
    'female',
    'female',
    'male',
    'male',
    'male',
]
btags = [
    '231026_L102P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-39-32',
    '231028_L106P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_14-48-05',
    '231028_L106P5_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_01-41-16',
    '230908_L88P4_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_10-33-21',
    '230819_L87P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_09-23-37',
    '230820_L88P1_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_11-00-37',
    '230825_L92P2_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_08-59-16',

]
for stitch_tile_ij, stitch_slice_ranges, ptag, btag in zip(stitch_tile_ij_lst, stitch_slice_ranges_lst, ptags, btags):
    save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/P14/{ptag}/{btag.split("_")[1]}'
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P14/{ptag}/{btag}'
    stitch_by_ptreg(stitch_tile_ij, stitch_slice_ranges, ptag=ptag, btag=btag, save_path=save_path, result_path=result_path, device=device)

