from phase_correlation_stitch import get_stitch_tform
import os

brain_ready = [  
    ['test_pair', 'test_brain'],
]
overlap_r = 0.2
ls_image = 'downloads/data'
nis_cpp_output = 'downloads/cpp_output'
save_root = 'downloads/stitch_tform'
for ptag, btag in brain_ready:
    ls_image_root = f'{ls_image}/{ptag}/{btag}'                 # path to raw image 
    save_path = f'{save_root}/{ptag}/{btag}'      # path to saving folder
    result_path = f'{nis_cpp_output}/{ptag}/{btag}'             # path to NIS results
    root = result_path + '/UltraII[%02d x %02d]'
    if not os.path.exists(root % (0, 0)): continue
    stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    if len(stack_names) == 0: 
        print(ptag, btag, "NIS not completed, skip")
        continue
    print("=== PCC stitch ", ptag, btag, "===")
    # if not os.path.exists(f'{save_path}/NIS_tranform/{btag}_tform_refine.json'):
    get_stitch_tform(ptag, btag, ls_image_root, save_path, result_path, overlap_r=overlap_r, btag_split=False)