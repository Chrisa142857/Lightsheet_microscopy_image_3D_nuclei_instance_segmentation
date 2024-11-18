from final_stitch import stitch_by_ptreg
from whole_brain_map import whole_brain_map



def main():
    stitch_by_ptreg(ptag='pair16', btag='220805_L74D769P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_15-51-36', device='cuda:3')
    whole_brain_map(ptag='pair16', btag='220805_L74D769P4_OUT_topro_ctip2_brn2_4x_11hdf_50sw_0_108na_4z_20ov_15-51-36', device='cuda:3')
    stitch_by_ptreg(ptag='pair4', btag='220904_L35D719P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_21-49-38', device='cuda:3')
    whole_brain_map(ptag='pair4', btag='220904_L35D719P5_topro_brn2_ctip2_4x_0_108na_50sw_11hdf_4z_20ov_21-49-38', device='cuda:3')
    stitch_by_ptreg(ptag='pair4', btag='220902_L35D719P3_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-41-36', device='cuda:3')
    whole_brain_map(ptag='pair4', btag='220902_L35D719P3_topro_brn2_ctip2_4x_11hdf_0_108na_50sw_4z_20ov_16-41-36', device='cuda:3')


if __name__ == '__main__': main()