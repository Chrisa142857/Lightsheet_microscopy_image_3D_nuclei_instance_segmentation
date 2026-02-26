import json, os, mat73, torch
import numpy as np

gtag = 'P14'
zratio = 2.5/4
overlap_r = 0.1
def main():
    
    brain_name = '231028_L106P3_sox9toproneun_sw50_4x_df9_4z_ov15_ex50_14-48-05'
    func('female', brain_name)
    # func()

def func(ptag, brain_name):
    btag = brain_name.split('_')[1]
    # matpath = f'/lichtman/Ian/Lightsheet/P14/stitched/{ptag}/{btag}/variables/stitch_tforms.mat'
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/{gtag}/{ptag}/{brain_name}'
    save_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/{gtag}/{ptag}/{btag}'

    # hv_tform = mat73.loadmat(matpath)
    disp_path = '/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/image_stitch/P14_L106P3_WobblyStitcher_displacements.npy'
    disp = np.load(disp_path)
    print(disp.shape)
    exit()
    h_tform = hv_tform['h_stitch_tforms']
    v_tform = hv_tform['v_stitch_tforms']
    zpath = f'numorph_param/{btag}_adjusted_z.json'
    with open(zpath, 'r') as file:
        z_adj = json.load(file)

    print(h_tform.shape, v_tform.shape, len(z_adj))
    '''
    Get lefttop location of tile
    '''


    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path) if 'Ultra' in fn])
    ncol, nrow = tile_loc.max(0)+1
    # print(tile_loc, nrow, ncol)
    assert len(tile_loc) == nrow*ncol, f'tile of raw data is not complete, tile location: {tile_loc}'

    root = result_path + '/UltraII[%02d x %02d]'
    stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    stack_names = sort_stackname(stack_names)
    # neighbor = [[-1, 0], [0, -1], [-1, -1], [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1]]
    os.makedirs(f'{save_path}/NIS_tranform', exist_ok=True)
    save_fn = f'{save_path}/NIS_tranform/{btag}_tform_numorph.json'

    for stack_name in stack_names:
        meta_name = stack_name.replace('instance_center', 'seg_meta')
        zstart = int(stack_name.split('zmin')[1].split('_')[0])
        zstart = int(zstart*zratio)
        seg_shape = torch.load(f'{root % (0, 0)}/{meta_name}')
        zend = int(zstart + seg_shape[0].item()*zratio)
    seg_shape[0] = zend
    print(seg_shape)
    z_tform = {}
    for z in z_adj:
        file = z['file']
        if '_C01_' not in file: continue
        sample_id = z['sample_id']
        assert btag == sample_id, f"{sample_id} != {btag}"
        i, j = file.split('UltraII[')[1].split(' x ')
        i = int(i)
        j = int(j[:2])
        ijkey = f'{i}-{j}'
        if ijkey not in z_tform: z_tform[ijkey] = [None for _ in range(seg_shape[0])]
        zi = int(file.split('Table Z')[1][:4])
        
        if zi >= seg_shape[0]: continue
        # if zi == 0: print(zi)
        z_tform[ijkey][zi] = z['z_adj'] - 1 - zi

        zi = zi - 1
        while zi >= 0:
            if z_tform[ijkey][zi] is None: z_tform[ijkey][zi] = z_tform[ijkey][zi+1]
            zi = zi - 1

    for k in z_tform:
        for zi, tz in enumerate(z_tform[k]):
            if tz is None: 
                z_tform[k][zi] = z_tform[k][zi-1]

    print('prepared z_tform', z_tform.keys())
    
    h_tformx = h_tform[::2]
    h_tformy = h_tform[1::2]
    v_tformx = v_tform[::2]
    v_tformy = v_tform[1::2]
    save_tform = {}
    for ijkey in z_tform:
        i, j = ijkey.split('-')
        i, j = int(i), int(j)
        save_tform[ijkey] = {zi: [z_tform[ijkey][zi], 0, 0] for zi in range(seg_shape[0])}
        if j == 0: continue
        for zi in range(seg_shape[0]):
            z_aligned = zi + z_tform[ijkey][zi]
            z_aligned = max(0, z_aligned)
            z_aligned = min(z_aligned, h_tform.shape[1]-1)
            
            x = h_tformx[(i*(ncol-1)+j)-1, z_aligned] + v_tformx[i-1, z_aligned] if i > 0 else 0# + tile_lt_loc[ijkey]['x']
            y = h_tformy[(i*(ncol-1)+j)-1, z_aligned] + v_tformy[i-1, z_aligned] if i > 0 else 0# + tile_lt_loc[ijkey]['y']
            save_tform[ijkey][zi] = [z_tform[ijkey][zi], x, y]
    print(save_fn, save_tform.keys())
    with open(save_fn, 'w') as file:
        json.dump(save_tform, file, indent=4)

def sort_stackname(stack_names):
    stack_z = []
    for stack_name in stack_names:
        stack_z.append(int(stack_name.split('zmin')[1].split('_')[0]))

    argsort = np.argsort(stack_z)
    return [stack_names[i] for i in argsort]

if __name__ == '__main__': main()