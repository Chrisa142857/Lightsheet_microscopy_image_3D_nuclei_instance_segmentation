import os, argparse



parser = argparse.ArgumentParser()
parser.add_argument('--gtag')
parser.add_argument('--ptag')
parser.add_argument('--btag')
# parser.add_argument('--tileij', nargs="+", type=int)
parser.add_argument('--savetag', default='', type=str)
parser.add_argument('--patch_num', default=100, type=int)
args = parser.parse_args()
tgtr = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{args.gtag}_ann_patches_layer23_bgrm_round2/brainwise'
btag = args.btag
print(f'''cd numorph_src
addpath(genpath(pwd));
generate_classifications('{btag}_coloc_layer23_bgrm', '{tgtr}/{btag}_isocortex_patches_with_bbox.tif', '{tgtr}/{btag}_isocortex_patch_info.csv')''')