import numpy as np
import scipy.ndimage
import pandas as pd
import torch, os, json, random, tifffile, sys
import nibabel as nib
from datetime import datetime
from tqdm import tqdm
from skimage import exposure, img_as_ubyte
import cv2, glob
from convert_p4_tileNIS2numorph import parse_stitch_tform
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
import argparse
saver = '/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/P4_ann_patches_layer23_bgrm_round2/tilewise'
# saver = '/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/P4_ann_patches_layer23_bgrm/tilewise'
# saver = '/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/P4_ann_patches_bgrm/tilewise'

def main():
    # generate_mask_layer23('P4')
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtag')
    parser.add_argument('--ptag')
    parser.add_argument('--btag')
    # parser.add_argument('--tileij', nargs="+", type=int)
    parser.add_argument('--savetag')
    parser.add_argument('--patch_num', default=100, type=int)
    

    args = parser.parse_args()
    patch_num = args.patch_num

    # btag = 'L35D719P1'
    # gtag = 'P4'
    gtag = args.gtag
    # ptag = 'pair11'
    # btag = 'L66D764P8'
    # ptag = 'pair14'
    # btag = 'L73D766P7'
    ptag = args.ptag
    btag = args.btag
    # gtag = 'P14'
    # ptag = 'female'
    # btag = 'L106P3'
    # tgti = args.tileij[0]
    # tgtj = args.tileij[1]
    savetag = args.savetag
    for fn in glob.glob(f'{saver}/{btag}*'):
        os.remove(fn)
    # generate_mask_path = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{gtag}_ann_mask/{btag}_ann_mask.nii.gz'
    generate_mask_path = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{gtag}_ann_mask_layer23/{btag}_ann_mask.nii.gz'
    for tgti in range(4):
        for tgtj in range(5):
            generate(gtag, ptag, btag, tgti, tgtj, generate_mask_path, patch_num=patch_num)

def generate_mask_p14(fn):
    gtag = 'P14'
    coloc_need_traisn_tab = '''L82D711P2	frp23	mos23	orbvl5	rspv6a	
    L82D711P3	acad1	orbvl5	rspv6a	
    L86P4	acad1	frp23	
    L87D868P2	rspd1	
    L87P1	visa6b	
    L88P1	
    L88P2	rspd1	vispor1
    L88P3	
    L88P4	rspd1	
    L90P3	audpo5	orbl5	rspd1	sspul6a	visa6b	vispor1
    L92P2	rspd1	
    L92P3	rspd1	vispor1
    L92P4	
    L94P1	
    L94P3sox9toproneun	rspd1	
    L94P4	rspd1	
    L95P2	
    L95P3	visa6b	
    L95P4	audpo5	rspd1	visa6b	
    L96P3	
    L97P1	orbl5	rspd1	visa6b	
    L97P2	audpo5	orbl5	rspd1	sspul6a	visa6b	vispor1
    L106P3	rspd1	
    L106P5	rspd1	visa6b	vispo1'''
    coloc_need_train_tab = coloc_need_train_tab.split('\n')
    coloc_need_train_tab = [i.split('\t') for i in coloc_need_train_tab]
    for items in coloc_need_train_tab:
        btag = items[0]
        chunklist = os.listdir(f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/P14_LS-NIS_chunk/{btag}')
        rnamelist = [i.split('_')[0].replace('-', '').lower() for i in chunklist]
        chunkfns = []
        for rname in items[1:]:
            if rname not in rnamelist: continue
            chunkfn = chunklist[rnamelist.index(rname)]
            chunkfns.append(chunkfn)
        print(chunkfns)
        psz_wh = 128
        psz_d = 60
        wh_r = .75/25
        z_r = 4/25
        bmap = nib.load(f'/scheibel/ACMUSERS/ziquanw/Lightsheet/nis_simple/{gtag}_raw_densitymap_res25um3/{btag}_raw_densitymap.nii.gz')
        bshape = bmap.shape
        mask = np.zeros(bshape,dtype=bool)
        for fn in chunkfns:
            x, y, z = fn.split('_')[1].split('-')[1:]
            x, y, z = int(x[1:]), int(y[1:]), int(z[1:])
            x1 = int((x - psz_wh) * wh_r)
            x2 = int((x + psz_wh) * wh_r)
            y1 = int((y - psz_wh) * wh_r)
            y2 = int((y + psz_wh) * wh_r)
            z1 = int((z - psz_d) * z_r)
            z2 = int((z + psz_d) * z_r)
            mask[z1:z2, x1:x2, y1:y2] = True
        savefn = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{gtag}_ann_mask/{btag}_ann_mask.nii.gz'
        print(datetime.now(), 'mask', mask.shape, mask.sum())
        print(datetime.now(), 'saving', savefn)
        nib.save(nib.Nifti1Image(mask, affine=bmap.affine, header=bmap.header), savefn) 

    
def generate_mask_layer23(gtag):
    atlaslabel = pd.read_csv('/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/allen_atlas_ccfv3_label_table.csv')
    rid2rname = {}
    for i, row in atlaslabel.iterrows():
        rid2rname[row['structure ID']] = row['abbreviation']
    # print(rid2rname)
    dobrains = [
        'L57D855P1',
    ]
    # exit()
    remap = json.load(open('/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/stats/roza_atlas_label_remap.json', 'r'))
    # device = 'cuda:5' 
    bmaproot = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/nis_simple/{gtag}_raw_densitymap'
    if gtag == 'P4':
        atlasroot = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/nis_simple/P4_atlas_for-raw-nis'
    for fn in tqdm(os.listdir(bmaproot)):
        if '.nii.gz' not in fn: continue
        btag = fn.split('_')[0]
        if btag not in dobrains: continue
        m1 = nib.load(f'{atlasroot}/{btag}_CCFv3_Right.nii.gz').get_fdata()
        m2 = nib.load(f'{atlasroot}/{btag}_CCFv3_Left.nii.gz').get_fdata()
        m1[m2>0] = m2[m2>0]
        # m1 = m1.to(device)
        mask = np.zeros_like(m1)
        for _rid in np.unique(m1):
            if _rid == 0: continue
            rid = remap[str(int(_rid.item()))]
            assert rid in atlaslabel['structure ID'].tolist(), f'{rid} {_rid}'
            if not rid2rname[rid].endswith('2/3'): continue
            xs, ys, zs = np.where(m1==_rid)
            if _rid in [582, 304, 211, 969, 288, 667, 962, 412, 328, 694]: continue
            mask[m1==_rid] = 1
            
        if gtag == 'P4':
            bmap = nib.load(f'/scheibel/ACMUSERS/ziquanw/Lightsheet/nis_simple/{gtag}_raw_densitymap/{btag}_raw_densitymap.nii.gz')
        else:
            bmap = nib.load(f'/scheibel/ACMUSERS/ziquanw/Lightsheet/nis_simple/{gtag}_raw_densitymap_res25um3/{btag}_raw_densitymap.nii.gz')
        bshape = bmap.shape
        assert (np.array(mask.shape) == np.array(bshape)).all(), f'{mask.shape} {bshape}'
        savefn = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{gtag}_ann_mask_layer23/{btag}_ann_mask.nii.gz'
        print(datetime.now(), 'mask', mask.shape, mask.sum())
        print(datetime.now(), 'saving', savefn)
        nib.save(nib.Nifti1Image(mask, affine=bmap.affine, header=bmap.header), savefn) 

def generate_mask_p4():
    gtag = 'P4'
    coloc_need_train_tab ='L59D878P2	ect6b	gu4	visa23'
#     L79D769P9	ect6b	gu4	orbl5	visa23
# L91D814P2	ect6b	gu4	orbl5	visa23
# L91D814P6	ect6b	gu4	orbl5	visa23'''
    # coloc_need_train_tab = '''L35D719P5	ect6b	gu4	orbl5	visa23
    # L35D719P3	ect6b	gu4	orbl5	visa23
    # L57D855P4	ect6b	gu4	orbl5	visa23
    # L57D855P5	ect6b	gu4	orbl5	visa23
    # L57D855P1	ect6b	gu4	visa23
    # L57D855P2	ect6b	gu4	visa23
    # L59D878P2	ect6b	gu4	visa23
    # L59D878P5	ect6b	gu4	visa23
    # L64D804P4	ect6b	gu4	
    # L64D804P6	ect6b	gu4	
    # L64D804P3	gu4	orbl5	visa23
    # L64D804P9	gu4	orbl5	visa23
    # L66D764P3	ect6b	gu4	orbl5	visa23
    # L66D764P8	ect6b	gu4	orbl5	visa23
    # L66D764P5	gu4	orbl5	visa23
    # L66D764P6	gu4	orbl5	visa23
    # L69D764P6	ect6b	gu4	orbl5	visa23
    # L69D764P9	ect6b	gu4	orbl5	visa23
    # L73D766P5	ect6b	gu4	orbl5	visa23
    # L73D766P7	ect6b	gu4	orbl5	visa23
    # L73D766P4	ect6b	gu4	visa23
    # L73D766P9	ect6b	gu4	visa23
    # L74D769P4	ect6b	gu4	orbl5	
    # L74D769P8	ect6b	gu4	orbl5	
    # L77D764P2	ect6b	gu4	orbl5	visa23
    # L77D764P9	ect6b	gu4	orbl5	visa23
    # L79D769P7	ect6b	gu4	orbl5	visa23
    # L79D769P9	ect6b	gu4	orbl5	visa23
    # L91D814P2	ect6b	gu4	orbl5	visa23
    # L91D814P6	ect6b	gu4	orbl5	visa23
    # L91D814P3	ect6b	gu4	orbl5	visa23
    # L91D814P4	ect6b	gu4	orbl5	visa23'''
    coloc_need_train_tab = coloc_need_train_tab.split('\n')
    coloc_need_train_tab = [i.split('\t') for i in coloc_need_train_tab]
    for items in coloc_need_train_tab:
        btag = items[0]
        chunklist = os.listdir(f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/P4_LS-NIS_chunk_correct-atlas/{btag}')
        rnamelist = [i.split('_')[0].replace('-', '').lower() for i in chunklist]
        chunkfns = []
        for rname in items[1:]:
            if rname not in rnamelist: continue
            chunkfn = chunklist[rnamelist.index(rname)]
            chunkfns.append(chunkfn)
        print(chunkfns)
        psz_wh = 128
        psz_d = 60
        wh_r = .75/25
        z_r = 4/25
        bmap = nib.load(f'/scheibel/ACMUSERS/ziquanw/Lightsheet/nis_simple/{gtag}_raw_densitymap/{btag}_raw_densitymap.nii.gz')
        bshape = bmap.shape
        mask = np.zeros(bshape,dtype=bool)
        for fn in chunkfns:
            x, y, z = fn.split('_')[1].split('-')[1:]
            x, y, z = int(x[1:]), int(y[1:]), int(z[1:])
            x1 = int((x - psz_wh) * wh_r)
            x2 = int((x + psz_wh) * wh_r)
            y1 = int((y - psz_wh) * wh_r)
            y2 = int((y + psz_wh) * wh_r)
            z1 = int((z - psz_d) * z_r)
            z2 = int((z + psz_d) * z_r)
            mask[z1:z2, x1:x2, y1:y2] = True
        savefn = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{gtag}_ann_mask/{btag}_ann_mask.nii.gz'
        print(datetime.now(), 'mask', mask.shape, mask.sum())
        print(datetime.now(), 'saving', savefn)
        nib.save(nib.Nifti1Image(mask, affine=bmap.affine, header=bmap.header), savefn) 
        
def adjust_intensity(patch):
    """
    Adjust intensity for each patch in the list `patches`, analogous to MATLAB imadjustn.
    """
    patch = patch.astype(np.float64)
    # compute percentiles
    p_low = np.percentile(patch, 0.02) / 65535.0
    p_high = np.percentile(patch, 99.98) / 65535.0
    
    # normalize between 0–1
    low, high = p_low, p_high
    patch_norm = np.clip((patch/65535.0 - low) / (high - low), 0, 1)
    
    # convert to uint8
    patch_uint8 = img_as_ubyte(patch_norm)
    return patch_uint8
# from scipy.ndimage import 
def generate(gtag, ptag, btag, tgti, tgtj, generate_mask_path, patch_num=1000, patch_sz=101):
    res = [4, 0.75, 0.75]
    down_res = 25
    down_zratio = res[0]/down_res
    down_xratio = res[1]/down_res
    down_yratio = res[2]/down_res
    if gtag == 'P4':
        density_map = nib.load(f'/scheibel/ACMUSERS/ziquanw/Lightsheet/nis_simple/{gtag}_raw_densitymap/{btag}_raw_densitymap.nii.gz').get_fdata()
    else:
        density_map = nib.load(f'/scheibel/ACMUSERS/ziquanw/Lightsheet/nis_simple/{gtag}_raw_densitymap_res25um3/{btag}_raw_densitymap.nii.gz').get_fdata()
    ## Generate patches according to a whole-brain mask
    # generate_mask = nib.load(f'L102P1corpuscallosummask_v2.nii.gz').get_fdata() > 0
    # mask_down_res = 12.5
    # mask_down_ratio = [resi/mask_down_res for resi in res]
    # print(generate_mask.sum(), generate_mask.shape[0]*generate_mask.shape[1]*generate_mask.shape[2])
    # generate_mask = nib.load(f'L74D769P4_coloc_mask_v1.nii').get_fdata() > 0
    # mask_down_res = 25
    # mask_down_ratio = [resi/mask_down_res for resi in res]
    # print(generate_mask.sum(), generate_mask.shape[0]*generate_mask.shape[1]*generate_mask.shape[2])
    # generate_mask = nib.load(f'L74D769P8_coloc_mask_v1.nii').get_fdata() > 0
    # generate_mask = nib.load(f'L73D766P5_coloc_mask_v1.nii.gz').get_fdata() > 0
    # generate_mask = nib.load(f'L73D766P4_coloc_mask_v1.nii.gz').get_fdata() > 0
    # generate_mask = nib.load(f'L73D766P9_coloc_mask_v1.nii.gz').get_fdata() > 0
    # generate_mask = nib.load(f'L69D764P6_coloc_mask_v1.nii.gz').get_fdata() > 0
    # generate_mask = nib.load(f'L69D764P9_coloc_mask_v1.nii.gz').get_fdata() > 0
    # generate_mask = nib.load(f'L77D764P9_coloc_mask_v1.nii.gz').get_fdata() > 0
    # generate_mask = nib.load(f'L77D764P2_coloc_mask_v1.nii.gz').get_fdata() > 0
    # generate_mask = nib.load(f'L64D804P3_coloc_mask_v1.nii.gz').get_fdata() > 0
    # generate_mask = nib.load(f'L66D764P6_coloc_mask_v1.nii.gz').get_fdata() > 0
    generate_mask = nib.load(generate_mask_path).get_fdata() > 0
    print(generate_mask.sum(), generate_mask.shape[0]*generate_mask.shape[1]*generate_mask.shape[2])
    # generate_mask = generate_mask[::-1, ::-1, ::-1]
    # generate_mask = scipy.ndimage.binary_erosion(generate_mask, iterations=1)
    # generate_mask = scipy.ndimage.binary_dilation(generate_mask, iterations=1)
    # generate_mask = np.roll(generate_mask, 5, 0)
    # generate_mask = np.roll(generate_mask, 5, 1)
    mask_down_res = 25
    mask_down_ratio = [resi/mask_down_res for resi in res]
    print(generate_mask.sum(), generate_mask.shape[0]*generate_mask.shape[1]*generate_mask.shape[2])
    # generate_mask = Nonee
    # exit()
    ##

    stitched_info, _, _ = parse_stitch_tform(ptag, gtag, btag)
    # tgti = int(sys.argv[4])
    # tgtj = int(sys.argv[5])
    channel_sort = ['C01', 'C00', 'C02']
    offset = 2
    down_time = 3
    pair_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/{gtag}/{ptag}'
    for dir in os.listdir(pair_path):
        if len(dir.split('_')) < 2 or '.' in dir: continue
        if btag == dir.split('_')[1]: 
            result_path = f'{pair_path}/{dir}'
            break
    tile_tag = 'UltraII[%02d x %02d]'
    result_root = result_path + '/' + tile_tag
    raw_img_root = result_root.replace('results', 'image_before_stitch_all_channel')
    stack_names = [f for f in os.listdir(result_root % (0, 0)) if f.endswith('instance_bbox.zip')]
    stack_names = sort_stackname(stack_names)
    meta_name = stack_names[0].replace('instance_bbox', 'seg_meta')
    tile_shape = torch.load(f'{result_root % (0, 0)}/{meta_name}')
    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path) if 'UltraII' in fn])
    ncol, nrow = tile_loc.max(0)+1
    tile_shape = [s.item() for s in tile_shape]
    
    chunk_num = len([_ for _ in stack_names for j in range(nrow) for i in range(ncol)])
    patch_per_chunk = int(np.floor(patch_num / chunk_num)*(5000 if generate_mask is not None else 5))
    patch_info = []
    patches = []
    clean_patches = []
    clean_patches3d = []
    chunks_done = 0
    chtop_ratio = 1
    for i in range(ncol):
        if i != tgti: continue
        for j in range(nrow):
            if j != tgtj: continue
            print(datetime.now(), tile_tag % (i, j), f'Loading random {len(stack_names)*patch_per_chunk} centers')
            chunks_done += 1
            topleft_loc = {
                stitched_info['z'][stitchi]: [stitched_info['x1'][stitchi], stitched_info['y1'][stitchi]] 
                for stitchi in range(len(stitched_info['z'])) 
                if stitched_info['ijkey'][stitchi] == tile_tag % (i, j)
            }
            centers = []
            index = []
            for stack_name in stack_names:
                ct_path = f"{result_root % (i, j)}/{stack_name}"
                if not os.path.exists(ct_path): continue
                tile_shape = torch.load(f"{result_root % (i, j)}/{stack_name.replace('instance_bbox', 'seg_meta')}")
                nisidx = torch.load(f"{result_root % (i, j)}/{stack_name.replace('instance_bbox', 'instance_label')}")
                tilew, tileh = tile_shape[1], tile_shape[2]
                stack_z = int(stack_name.split('zmin')[1].split('_')[0])
                ct = torch.load(ct_path).float()
                middlemask = torch.logical_and(
                    torch.logical_and(ct[:, 1] >= patch_sz//2, ct[:, 2] >= patch_sz//2),
                    torch.logical_and(ct[:, 4] <= tilew - patch_sz//2, ct[:, 5] <= tileh - patch_sz//2)
                ).numpy()
                idx = np.arange(ct.shape[0])[middlemask].tolist()
                random.shuffle(idx)
                idx = idx[:patch_per_chunk]
                index.extend(nisidx[idx].tolist())
                ct = ct[idx, :]
                ct[:, ::3] = (ct[:, ::3] + stack_z) * (2.5/res[0])
                centers.append(ct)
            centers = torch.cat(centers).round()
            print(datetime.now(), tile_tag % (i, j), f'Loaded {len(centers)} centers')
            ct_allz = ((centers[:, 0] + centers[:, 3]) // 2).long()
            ct_zdepth = (centers[:, 3] - centers[:, 0]).long()
            loadrange_img3d = ct_zdepth.max() // 2
            imgfns = os.listdir(raw_img_root % (i, j))
            # imgfns = np.array([fn for fn in imgfns if fn.startswith(btag)])
            # imgz = np.array([int(fn.split('_')[1]) for fn in imgfns])
            imgfns = np.array([fn for fn in imgfns if btag in fn])
            imgz = np.array([int(fn.split(' Z')[-1].split('.')[0]) for fn in imgfns])
            imgchs = np.array([fn.split('_')[-2] for fn in imgfns])
            # patch = np.zeros([patch_sz*2, patch_sz*2, 3, len(centers)], dtype=np.uint8).astype(np.uint8)
            # clean_patch = np.zeros([patch_sz, patch_sz, 3, len(centers)], dtype=np.uint8).astype(np.uint8)
            # clean_patch3d = np.zeros([patch_sz, patch_sz, ct_zdepth.max(), 3, len(centers)], dtype=np.uint8).astype(np.uint8)
            # print(clean_patch3d.shape)
            # info = np.ones([len(centers), len(channel_sort) + 5 + 3])
            # info[:, 0] = index
            # keep_index = []
            patch = []
            clean_patch = []
            clean_patch3d = []
            info = []
            header = ['NIS index', 'center_x', 'center_y', 'center_z', 'atlas_annotation'] + channel_sort + ['NIS_shape_x', 'NIS_shape_y', 'NIS_shape_z']
            img_buffer = {}
            for ctz in tqdm(ct_allz.unique().tolist(), desc=f'{datetime.now()} Patch generating zrange=[{ct_allz.min().item()}, {ct_allz.max().item()}]'):
                if generate_mask is not None: 
                    if int(ctz*mask_down_ratio[0]) >= generate_mask.shape[0]: continue
                    if not generate_mask[int(ctz*mask_down_ratio[0])].any(): continue
                if ctz < 850: continue ## Z conditions
                # if not ((ctz > 400 and ctz <= 600) or (ctz < 1000 and ctz >= 900)): continue ## Z conditions
                # if not ((ctz < 400 and ctz >= 600) or (ctz < 1000 and ctz >= 900)): continue ## Z conditions
                ctzmask = ct_allz==ctz
                bz1, bx1, by1, bz2, bx2, by2 = centers[ctzmask].T.long()
                x = (bx1 + bx2) // 2
                y = (by1 + by2) // 2
                depth = bz2 - bz1
                w = bx2 - bx1
                h = by2 - by1
                ### Global NIS center
                if int(ctz) in topleft_loc:
                    tx, ty = topleft_loc[int(ctz)]
                else: 
                    tx, ty = 0, 0
                # print("len(x)", len(x))
                if len(x) <= 10: continue
                # print(len(x))
                ###########
                if generate_mask is not None:
                    maskz, maskx, masky = int(ctz*mask_down_ratio[0]), (x+tx)*mask_down_ratio[1], (y+ty)*mask_down_ratio[2]
                    # print(maskz, maskx.max().item(), masky.max().item(), generate_mask.shape)
                    ct_mask = generate_mask[torch.LongTensor([maskz]).repeat(len(maskx)), maskx.long().clip(max=generate_mask.shape[1]-1), masky.long().clip(max=generate_mask.shape[2]-1)]
                    # if len(ct_mask.shape) == 0: ct_mask = torch.from_numpy(ct_mask)
                    x, y, w, h =  x[ct_mask], y[ct_mask], w[ct_mask], h[ct_mask]
                    depth = depth[ct_mask]
                    ctzmask[torch.where(ctzmask)[0][~ct_mask]] = False

                if len(x) == 0: continue
                if int(len(x)*chtop_ratio) == 0: continue
                # print("len(x)", len(x))
                ###########
                imgfn = imgfns[imgz == ctz]
                imgch = imgchs[imgz == ctz]
                img = []
                for ch in channel_sort:
                    fn = imgfn[imgch==ch]
                    assert len(fn) == 1, f"{imgch}, {ch}, {np.unique(imgz)}, {ctz}, {len(np.unique(imgz))}, {tile_tag % (i, j)}"
                    if f'{raw_img_root % (i, j)}/{fn[0]}' not in img_buffer: 
                        im = tifffile.imread(f'{raw_img_root % (i, j)}/{fn[0]}')
                        img_buffer[f'{raw_img_root % (i, j)}/{fn[0]}'] = im
                    else:
                        im = img_buffer[f'{raw_img_root % (i, j)}/{fn[0]}']
                    img.append(im)
                    
                ref_img = img[0]
                for channeli in range(1, len(img)):
                    mov_img = img[channeli]
                    shift, error, diffphase = phase_cross_correlation(ref_img[::down_time,::down_time], mov_img[::down_time,::down_time])
                    shift = shift * down_time
                    img[channeli] = scipy.ndimage.shift(mov_img, shift)

                img = np.stack(img, -1)
                img = np.pad(img, ((patch_sz, patch_sz), (patch_sz, patch_sz), (0, 0)))
                ###########
                sortch = 2 if gtag == 'P4' else 1
                xoffset = torch.stack([x+offset for offset in torch.meshgrid(torch.arange(-w.max()//2, w.max()//2+1), torch.arange(-h.max()//2, h.max()//2+1))[0].reshape(-1)], 1).long()
                yoffset = torch.stack([y+offset for offset in torch.meshgrid(torch.arange(-w.max()//2, w.max()//2+1), torch.arange(-h.max()//2, h.max()//2+1))[1].reshape(-1)], 1).long()
                chpix = img[xoffset, yoffset, sortch]
                k = 10 if chpix.shape[1] >= 10 else chpix.shape[1]
                topch = torch.from_numpy(
                    np.argsort(
                        np.take_along_axis(chpix, np.argpartition(chpix, -k, axis=1)[:, -k:], axis=1).mean(1)
                        *-1)[:int(len(x)*chtop_ratio)])
                ct_mask = torch.zeros(len(x)).bool()
                ct_mask[topch] = True
                x, y, w, h =  x[ct_mask], y[ct_mask], w[ct_mask], h[ct_mask]
                depth = depth[ct_mask]
                ctzmask[torch.where(ctzmask)[0][~ct_mask]] = False
                assert len(x) > 0, len(x)
                ###########
                ########### Load 3D image ###########
                # img3d = []
                # for ctz3d in range(ctz-loadrange_img3d, ctz+loadrange_img3d+1):
                #     imgfn = imgfns[imgz == ctz3d]
                #     imgch = imgchs[imgz == ctz3d]
                #     imgzi = []
                #     for ch in channel_sort:
                #         fn = imgfn[imgch==ch]
                #         assert len(fn) == 1, f"{imgch}, {ch}, {np.unique(imgz)}, {ctz}, {len(np.unique(imgz))}, {tile_tag % (i, j)}"
                #         if f'{raw_img_root % (i, j)}/{fn[0]}' not in img_buffer: 
                #             im = tifffile.imread(f'{raw_img_root % (i, j)}/{fn[0]}')
                #             img_buffer[f'{raw_img_root % (i, j)}/{fn[0]}'] = im
                #         else:
                #             im = img_buffer[f'{raw_img_root % (i, j)}/{fn[0]}']
                #         imgzi.append(im)
                        
                #     ref_img = imgzi[0]
                #     for channeli in range(1, len(imgzi)):
                #         mov_img = imgzi[channeli]
                #         shift, error, diffphase = phase_cross_correlation(ref_img[::down_time,::down_time], mov_img[::down_time,::down_time])
                #         shift = shift * down_time
                #         imgzi[channeli] = scipy.ndimage.shift(mov_img, shift)
                #     img3d.append(np.stack(imgzi, -1))
                # img3d = np.stack(img3d, 2)
                # img3d = np.pad(img3d, ((patch_sz, patch_sz), (patch_sz, patch_sz), (0, 0), (0, 0)))
                ########### 
                imgs = []
                clean_imgs = []
                clean_imgs3d = []
                for xi, yi, wi, hi in zip(x, y, w, h):
                    x1 = xi - patch_sz // 2
                    x2 = x1 + patch_sz
                    y1 = yi - patch_sz // 2
                    y2 = y1 + patch_sz
                    im = img[x1+patch_sz:x2+patch_sz, y1+patch_sz:y2+patch_sz, :].copy()
                    im[..., 0] = adjust_intensity(im[..., 0])
                    im[..., 1] = adjust_intensity(im[..., 1])
                    im[..., 2] = adjust_intensity(im[..., 2])  
                    clean_imgs.append(im.copy())
                    # im3d = img3d[x1+patch_sz:x2+patch_sz, y1+patch_sz:y2+patch_sz, :, :].copy()
                    # im3d[..., 0] = adjust_intensity(im3d[..., 0])
                    # im3d[..., 1] = adjust_intensity(im3d[..., 1])
                    # im3d[..., 2] = adjust_intensity(im3d[..., 2])  
                    # clean_imgs3d.append(im3d.copy())
                    # imprev = im3d[:, :, loadrange_img3d-1].copy()
                    # imnext = im3d[:, :, loadrange_img3d+1].copy()
                    im = cv2.rectangle(im,
                        (int(patch_sz // 2 - wi//2 - offset), int(patch_sz // 2 - hi//2 - offset)),
                        (int(patch_sz // 2 + wi//2 + offset), int(patch_sz // 2 + hi//2 + offset)),
                        (255, 255, 255), 1
                    )                 
                    # imprev = cv2.rectangle(imprev,
                    #     (int(patch_sz // 2 - wi//2 - offset), int(patch_sz // 2 - hi//2 - offset)),
                    #     (int(patch_sz // 2 + wi//2 + offset), int(patch_sz // 2 + hi//2 + offset)),
                    #     (255, 255, 255), 1
                    # )                 
                    # imnext = cv2.rectangle(imnext,
                    #     (int(patch_sz // 2 - wi//2 - offset), int(patch_sz // 2 - hi//2 - offset)),
                    #     (int(patch_sz // 2 + wi//2 + offset), int(patch_sz // 2 + hi//2 + offset)),
                    #     (255, 255, 255), 1
                    # )                 

                    patch_loc = density_map[int(ctz*down_zratio)]
                    locx, locy = int((xi+tx)*down_xratio), int((yi+ty)*down_yratio)
                    assert locx <= patch_loc.shape[0], f'{locx} {xi} {tx} {down_xratio}, {patch_loc.shape}'
                    assert locy <= patch_loc.shape[1], f'{locy} {yi} {ty} {down_yratio}, {patch_loc.shape}'
                    patch_loc = adjust_intensity(patch_loc)
                    densityw, densityh = patch_loc.shape
                    patch_loc = cv2.resize(patch_loc, im.shape[:2])[:, :, None].repeat(3, axis=2)
                    patch_loc = cv2.drawMarker(patch_loc, 
                        (int(locy * im.shape[1] / densityh), int(locx * im.shape[0] / densityw)), 
                        (255, 255, 255), markerType=cv2.MARKER_STAR,
                        markerSize=7, thickness=1)
                    imgs.append(np.concatenate([
                        np.concatenate([im, patch_loc]),
                        # np.concatenate([imprev, patch_loc]),
                        np.concatenate([np.zeros_like(im), np.zeros_like(im)])
                    ], 1))
                    # imgs.append(np.concatenate([im, patch_loc]))
                    # imgs.append(im)
                    # patch_locs.append(patch_loc)
            
                # clean_patch[..., ctzmask] = np.stack(clean_imgs, -1)   
                # clean_patch3d[..., ctzmask] = np.stack(clean_imgs3d, -1)   
                # patch[:imgs[0].shape[0], :imgs[0].shape[1], :, ctzmask] = np.stack(imgs, -1)    
                # info[ctzmask, 5:8] = img[x+patch_sz, y+patch_sz, :]
                # info[ctzmask, 1] = x + tx
                # info[ctzmask, 2] = y + ty
                # info[ctzmask, 3] = ctz
                # info[ctzmask, 8] = w
                # info[ctzmask, 9] = h
                # info[ctzmask, 10] = depth
                # keep_index.append(torch.where(ctzmask)[0])
                patch.append(np.stack(imgs, -1))
                clean_patch.append(np.stack(clean_imgs, -1))
                # clean_patch3d.append(np.stack(clean_imgs3d, -1))
                oneinfo = np.ones([clean_patch[-1].shape[-1], len(channel_sort) + 5 + 3])
                oneinfo[:, 5:8] = img[x+patch_sz, y+patch_sz, :]
                oneinfo[:, 0] = np.array(index)[ctzmask]
                oneinfo[:, 1] = x + tx
                oneinfo[:, 2] = y + ty
                oneinfo[:, 3] = ctz
                oneinfo[:, 8] = w
                oneinfo[:, 9] = h
                oneinfo[:, 10] = depth 
                info.append(oneinfo)

            # if len(keep_index) == 0: continue
            # keep_index = torch.cat(keep_index)
            # patch_info.append(info[keep_index, :])
            # patches.append(patch[..., keep_index])
            # clean_patches.append(clean_patch[..., keep_index])
            # clean_patches3d.append(clean_patch3d[..., keep_index])
            if len(info) == 0: continue
            patch = np.concatenate(patch, -1).astype(np.uint8)
            patch_info.append(np.concatenate(info, 0))
            # patches.append(np.concatenate([patch, np.zeros(patch.shape).astype(np.uint8)], 1).astype(np.uint8))
            patches.append(patch.astype(np.uint8))
            clean_patches.append(np.concatenate(clean_patch, -1))
            # clean_patches3d.append(np.concatenate(clean_patch3d, -1))

        #     break
        # break
    if len(patch_info) == 0: return
    patch_info = np.concatenate(patch_info).astype(np.int32)
    patches = np.concatenate(patches, -1)
    clean_patches = np.concatenate(clean_patches, -1)
    # clean_patches3d = np.concatenate(clean_patches3d, -1)
    print(patch_info.shape, patches.shape, patches.dtype)
    # np.savetxt(f'coloc_ann_patch/tile_wise/{btag}_patch_info_tile{tgti:02d}-{tgtj:02d}.csv', patch_info, delimiter=',', fmt='%d', header=','.join(header))
    # tifffile.imwrite(f'coloc_ann_patch/tile_wise/{btag}_patches_with_bbox_tile{tgti:02d}-{tgtj:02d}.tif', patches.transpose(3, 2, 1, 0), photometric='rgb')
    # tifffile.imwrite(f'coloc_ann_patch/tile_wise/{btag}_patches_tile{tgti:02d}-{tgtj:02d}.tif', clean_patches.transpose(3, 2, 1, 0), photometric='rgb')
    ## Restrict to isocortex ####################
    if generate_mask is None:
        ctx, cty, ctz = patch_info[:, 1:4].T
        center = torch.from_numpy(np.stack([ctz, ctx, cty], -1))
        annotations = get_annotations(ptag, btag, center)
        patch_info[:, 4] = annotations.numpy()
        print(annotations.unique())
        annmask = np.where(patch_info[:, 4] == 16001)[0].tolist()
        random.shuffle(annmask)
        annmask = annmask[:patch_per_chunk*chunks_done]
        patch_info = patch_info[annmask]
        patches = patches[:, :, :, annmask]
        clean_patches = clean_patches[:, :, :, annmask] # x, y, ch, patchi
        # clean_patches3d = clean_patches3d[:, :, :, :, annmask] # x, y, z, ch, patchi
    ## Restrict to isocortex ####################
    print(datetime.now(), 'patch num generated:', clean_patches.shape[-1])
    np.savetxt(f'{saver}/{btag}_isocortex_patch_info_tile{tgti:02d}-{tgtj:02d}.csv', patch_info, delimiter=',', fmt='%d', header=','.join(header))
    tifffile.imwrite(f'{saver}/{btag}_isocortex_patches_with_bbox_tile{tgti:02d}-{tgtj:02d}.tif', patches.transpose(3, 2, 1, 0), photometric='rgb')
    tifffile.imwrite(f'{saver}/{btag}_isocortex_patches_tile{tgti:02d}-{tgtj:02d}.tif', clean_patches.transpose(3, 2, 1, 0), photometric='rgb')
    # tifffile.imwrite(f'{saver}/{btag}_isocortex_patches3d_tile{tgti:02d}-{tgtj:02d}.tif', clean_patches3d.transpose(4, 3, 2, 1, 0), photometric='rgb')
    print(datetime.now(), 'patch saved:', saver)
    
def get_annotations(ptag, btag, center):
    atlas_path = f'/lichtman/Felix/Lightsheet/P4/{ptag}/output_{btag}/registered/{btag}_MASK_topro_25_all.nii'
    if not os.path.exists(atlas_path): return torch.zeros(len(center)).long() + 16001
    brain_mask_downr = torch.FloatTensor([4/25, 0.75/25, 0.75/25])#.to(device)
    brain_mask1 = interested_brain_mask = torch.from_numpy(np.array(nib.load(atlas_path).get_fdata()).transpose(2,0,1).copy())
    down_center = center * brain_mask_downr[None]
    down_center = down_center.clip(max=torch.FloatTensor(list(brain_mask1.shape))[None]-1, min=torch.zeros(1,3)).long()
    annotations = interested_brain_mask[down_center[:, 0], down_center[:, 1], down_center[:, 2]]
    return annotations

def sort_stackname(stack_names):
    stack_z = []
    for stack_name in stack_names:
        stack_z.append(int(stack_name.split('zmin')[1].split('_')[0]))

    argsort = np.argsort(stack_z)
    return [stack_names[i] for i in argsort]



if __name__ == '__main__': main()