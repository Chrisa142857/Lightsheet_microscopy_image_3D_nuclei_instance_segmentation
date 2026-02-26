import os, argparse
import sys
import pandas as pd
import tifffile
import numpy as np
import random
from lmsf_torch import lmsf_2d_cumulative
from datetime import datetime
from tqdm import tqdm, trange
import torch.fft

parser = argparse.ArgumentParser()
parser.add_argument('--gtag')
parser.add_argument('--ptag')
parser.add_argument('--btag')
# parser.add_argument('--tileij', nargs="+", type=int)
parser.add_argument('--savetag', default='', type=str)
parser.add_argument('--patch_num', default=100, type=int)
args = parser.parse_args()

gtag = args.gtag
ptag = args.ptag
btag = args.btag
savetag = args.savetag
patch_num = args.patch_num
device = 'cpu'
# gtag = sys.argv[1]
# btag = sys.argv[2]
# savetag = sys.argv[3]

# patch_num = 100


def fft_denoise(images, cutoff=25):

    """
    Apply Gaussian Low Pass Filter to a batch of images using PyTorch.
    
    Parameters
    ----------
    images : torch.Tensor
        Input images. Shape (N, H, W) or (N, C, H, W).
    cutoff : float
        The cutoff frequency (sigma) for the Gaussian mask.
        
    Returns
    -------
    img_denoised : torch.Tensor
        Filtered images with same shape as input.
    """
    # ensure input is on the correct device and is float
    if not torch.is_floating_point(images):
        images = images.float()
        
    # Get dimensions (H, W are the last two dimensions)
    h, w = images.shape[-2:]
    device = images.device

    # --- 1. Create Gaussian Mask (Batch-friendly) ---
    # Create a grid of coordinates centered at (0,0)
    y = torch.arange(h, device=device) - h // 2
    x = torch.arange(w, device=device) - w // 2
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Calculate squared distance from center
    # D^2 = x^2 + y^2
    dist_sq = X**2 + Y**2
    
    # Gaussian Mask: exp( -D^2 / (2 * cutoff^2) )
    # Shape: (H, W). PyTorch will broadcast this to (N, H, W) automatically.
    mask = torch.exp(-dist_sq / (2 * cutoff**2))
    
    # --- 2. FFT Denoising ---
    
    # Compute 2D FFT
    # dim=(-2, -1) ensures we transform only spatial dimensions, keeping batch dims
    dft = torch.fft.fft2(images, dim=(-2, -1))
    
    # Shift zero frequency to center
    dft_shift = torch.fft.fftshift(dft, dim=(-2, -1))
    
    # Apply Mask (Broadcasting happens here)
    fshift = dft_shift * mask
    
    # Inverse Shift
    f_ishift = torch.fft.ifftshift(fshift, dim=(-2, -1))
    
    # Inverse FFT
    img_back = torch.fft.ifft2(f_ishift, dim=(-2, -1))
    
    # Take magnitude (abs) to get real part
    img_denoised = torch.abs(img_back)
    
    return img_denoised

ref_patch_info = None
# ref_patch_info = pd.read_csv(f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_classification_wholebrain/P4_mapRes25/{ptag}_{btag}_resnet50ChAlign_nis_coloc.csv', usecols=['NIS_ID', 'NIS_inside_tile'])
# ref_patch_info = pd.read_csv('/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L73D766P5_coloc-mask-v1_isocortex_patch_info.csv')
# tgtr = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{gtag}_ann_patches/tilewise'
# saver = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{gtag}_ann_patches/brainwise'
# tgtr = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{gtag}_ann_patches_bgrm/tilewise'
# saver = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{gtag}_ann_patches_bgrm/brainwise'
# tgtr = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{gtag}_ann_patches_layer23_bgrm/tilewise'
# saver = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{gtag}_ann_patches_layer23_bgrm/brainwise'
tgtr = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{gtag}_ann_patches_layer23_bgrm_round2/tilewise'
saver = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{gtag}_ann_patches_layer23_bgrm_round2/brainwise'
os.makedirs(saver, exist_ok=True)
fns = os.listdir(tgtr)
patches_fns = [fn for fn in fns if fn.startswith(f'{btag}_isocortex_patch')]
tilekey = patches_fns[0].split('_')[-1].split('.')[0]
if gtag == 'P4' or btag.split('_')[0] == 'L102P2':
    ncol = 4
    nrow = 5
elif gtag == 'P14':
    ncol = 4
    nrow = 4

merge_fns = [fn.replace('_'+tilekey, '') for fn in patches_fns if tilekey in fn]
merge_data = {fn: [] for fn in merge_fns}
for savefn in merge_fns:
    for i in range(ncol):
        for j in range(nrow):
            fn, tail = savefn.split('.')
            tile_fn = f'{fn}_tile{i:02d}-{j:02d}.{tail}'
            if tile_fn not in patches_fns: continue
            # assert tile_fn in patches_fns, f'{tile_fn} {patches_fns}'
            merge_data[savefn].append(f'{tgtr}/{tile_fn}')
        #     break
        # break
# print(merge_data)
# merge_fns = [fn for fn in merge_fns if fn.endswith('.csv')] + [fn for fn in merge_fns if not fn.endswith('.csv')]
# assert merge_fns[0].endswith('.csv') and len(merge_data[merge_fns[0]])>0, merge_fns
merge_fns = [fn for fn in merge_fns if not fn.endswith('.csv') and 'with_bbox' not in fn] + [fn for fn in merge_fns if not fn.endswith('.csv') and 'with_bbox' in fn]
assert merge_fns[0].endswith('.tif') and len(merge_data[merge_fns[0]])>0, merge_fns
patch_idx = None
bsz = 30
# for savefn in merge_fns:
for savefn in merge_fns:
    if savefn.endswith('.csv'): continue
    if 'patches3d' in savefn: continue
    print(savefn)
    images = [tifffile.imread(fn) for fn in tqdm(merge_data[savefn], desc='preload patch candicates')]
    raw_dtype = images[0].dtype
    
    if 'patches3d' in savefn:
        maxdepth = np.array([image.shape[2] for image in images]).max()
        images = [
            np.pad(image, ((0, 0), (0, 0), ((maxdepth - image.shape[2])//2, (maxdepth - image.shape[2])//2), (0, 0), (0, 0)))
        for image in images]

    image = np.concatenate(images, axis=0)#[:patch_num]
    if 'with_bbox' in savefn: 
        image = image.astype(float)
        print(datetime.now(), 'BG remove start', image.shape)
        batchsz = 10000
        for chi in range(image.shape[1]):
            input_img = image[:, chi:chi+1, :101, :101]
            output = []
            for batchi in range(0, len(input_img), batchsz):
                output.append(fft_denoise(torch.from_numpy(image[batchi:batchi+batchsz, chi:chi+1, :101, :101]).contiguous().to(device)).cpu().numpy())
            image[:, chi:chi+1, :101, :101] = np.concatenate(output)
            # image[:, chi:chi+1, :101, :101] = lmsf_2d_cumulative(image[:, chi:chi+1, :101, :101], device=device).numpy()
        print(datetime.now(), 'BG remove done')
        image = image.astype(raw_dtype)
    if patch_idx is None: 
        info_savefn = f'/{btag}_isocortex_patch_info.csv'
        df = pd.concat([pd.read_csv('/'.join(fn.split('/')[:-1])+'/'+info_savefn.replace('.csv', f'_{fn.split("/")[-1].split("_")[-1][:-4]}.csv')) for fn in merge_data[savefn]], axis=0)#.iloc[:patch_num]
        
        all_patch_num = image.shape[0]
        idx = np.arange(all_patch_num)
        # if ref_patch_info is not None:
        # idxmask1 = df['# NIS index'].isin(ref_patch_info['NIS_ID'])
        idxmask = ~(image[:, :, (50-bsz//2):(50+bsz//2), (50-bsz//2):(50+bsz//2)]==0).any(-1).any(-1).any(-1)
        
        print('prev idxmask', len(idx))
        idx = idx[idxmask]
        # assert (image[idx, :, :100, :100]!=0).all()
        print('after idxmask', len(idx))
        idx = idx[np.argsort((image[idxmask, :, (50-bsz//2):(50+bsz//2), (50-bsz//2):(50+bsz//2)].reshape(idx.shape[0], image.shape[1], -1).std(-1))[:, 1]*-1)]
        patch_idx = idx[:patch_num]#.tolist()
        
        df = df.iloc[patch_idx]

        if gtag == 'P14':
            info_savefn = info_savefn.replace('_isocortex', '') 
        info_savefn = info_savefn.replace('.csv', f'{savetag}.csv')
        df.to_csv(f'{saver}/{info_savefn}', float_format='%d', index=False)
        print('Saved', info_savefn, df.shape)

    image = image[patch_idx]
    # assert (image[:, :, :100, :100]!=0).all()

    if gtag == 'P14':
        savefn = savefn.replace('_isocortex', '') 
    savefn = savefn.replace('.tif', f'{savetag}.tif')
    tifffile.imwrite(f'{saver}/{savefn}', image, photometric='rgb')
    print('Saved', savefn, image.shape)


# for savefn in merge_fns:
#     if not savefn.endswith('.csv'): continue
#     if 'patches3d' in savefn: continue
#     print(savefn)
#     df = pd.concat([pd.read_csv(fn) for fn in merge_data[savefn]], axis=0)#.iloc[:patch_num]
#     assert patch_idx is not None
#     # if all_patch_num is None: 
#     #     all_patch_num = df.shape[0]
#     #     patch_idx = np.arange(all_patch_num)
#     #     print(len(patch_idx))
#     #     if ref_patch_info is not None:
#     #         patch_idx = patch_idx[~df['# NIS index'].isin(ref_patch_info['# NIS index'])]
#     #     patch_idx = patch_idx.tolist()
#     #     random.shuffle(patch_idx)
#     #     patch_idx = patch_idx[:patch_num]
#     df = df.iloc[patch_idx]

#     if gtag == 'P14':
#         savefn = savefn.replace('_isocortex', '') 
#     savefn = savefn.replace('.csv', f'{savetag}.csv')
#     df.to_csv(f'{saver}/{savefn}', float_format='%d', index=False)
#     print('Saved', savefn, df.shape)