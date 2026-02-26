# First time setup NM
cd annotation_gui
addpath(genpath(pwd));
NM_setup;

###
# Manual annotation key strokes:
#   l/r arrow: navigate patch number
#   1-9:       assign class annotation
#   r:         display only red (1st) channel
#   g:         display only green (2nd) channel
#   b          display only blue (3rd) channel
#   c:         display composite (all channels)
#   a:         increase brightness of selected channels
#   h:         display help menu
#   s:         subset patches of a given annotation
#   f:         find nearest unannotated patch
#   w:         save results to csv file
#   l:         load previous results from csv file
###
cd numorph_src
addpath(genpath(pwd));
generate_classifications('L57D855P5_coloc_layer23_bgrm_round2', '/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/P4_ann_patches_layer23_bgrm_round2/brainwise/L57D855P5_isocortex_patches_with_bbox.tif', '/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/P4_ann_patches_layer23_bgrm_round2/brainwise/L57D855P5_isocortex_patch_info.csv')

