from datetime import datetime
import numpy as np
import torch, torchist
import h5py, os, fastremap
import multiprocessing as mp
from tqdm import tqdm

import utils
from unet import NISModel

def main():
    save_ftail = '_NIS_results.h5'
    chunk_depth = 48
    ## prev steps setting #############################
    dir_n = 'flow_3d'
    pair_tag = 'pair15'
    brain_tag = 'L73D766P4' # L73D766P9
    # brain_tag = sys.argv[1]
    save_r = '/lichtman/ziquanw/Lightsheet/results/P4'
    save_r = '%s/%s' % (save_r, pair_tag)
    save_r = '%s/%s' % (save_r, brain_tag)
    flow_r = '%s/%s' % (save_r, dir_n)
    eval_tag = 'resample'
    fs = [os.path.join(flow_r, f) for f in os.listdir(flow_r) if eval_tag in f and f.endswith('.npy')]
    orig_fs = fs.copy()
    depth_start = 0
    for f in orig_fs:
        fs[filename_to_depth(f, depth_start)] = f
    depth = len(fs) 
    x = np.load(fs[0])
    whole_brain_shape = (depth, x.shape[1], x.shape[2])
    trained_model = 'downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_22_42_54.153497_epoch_21'
    model = NISModel(device=torch.device('cpu'), pretrained_model=trained_model)
    rescale = model.diam_mean / model.diam_labels
    niter = np.uint32(1 / rescale * 200)
    save_path = os.path.join(save_r, brain_tag+save_ftail)
    f = h5py.File(save_path, 'w')
    seg_dset = f.create_dataset('nuclei_segmentation', whole_brain_shape, dtype='int', chunks=True, compression='gzip')
    coord_dset = f.create_dataset('coordinate', (0, 4), dtype='int', chunks=True, maxshape=(None, 4))
    label_dset = f.create_dataset('label', (0,), dtype='int', chunks=True, maxshape=(None,))
    idbase_dset = f.create_dataset('idbase', (0,), dtype='int', chunks=True, maxshape=(None,))
    stitch_range = f.create_dataset('wait_for_stitch', (0, 6), dtype='int', maxshape=(None, 6))
    idbase = 0
    for z in range(0, depth, chunk_depth):
        zmin, zmax = z, min(depth, z + chunk_depth)
        masks, coords, labels = compute_one_chunk(fs, zmin, zmax, niter)
        if masks is None: continue
        seg_dset[zmin:zmax, :, :] = masks
        old_contour_n = coord_dset.shape[0]
        new_contour_n = old_contour_n + coords.shape[0]
        coord_dset.resize((new_contour_n, coord_dset.shape[1]))
        coord_dset[old_contour_n:] = coords
        old_instance_n = label_dset.shape[0]
        new_instance_n = old_instance_n + labels.shape[0]
        label_dset.resize((new_instance_n,))
        label_dset[old_instance_n:] = labels
        idbase_dset.resize((new_instance_n,))
        idbase_dset[old_instance_n:] = idbase
        idbase = idbase + max(labels)
    
def compute_one_chunk(all_paths, zmin, zmax, niter):
    with mp.Pool(processes=4) as pool:
        paths = [path for path in all_paths if zmin <= filename_to_depth(path) < zmax]
        data = list(tqdm(pool.imap(np.load, [path for path in paths]), total=len(paths), desc=f"Load from slice {zmin} to {zmax}")) # [3 x Y x X]
    data = torch.from_numpy(np.stack(data, axis=1))
    dP = data[:3]
    cellprob = data[3]
    print(datetime.now(), 'Start compute masks from grad')
    masks, coords, labels = compute_masks(dP, cellprob, niter=niter)
    print(datetime.now(), f'Done, {labels.shape} masks from grad')
    return masks, coords, labels

def compute_masks(dP, cellprob, niter, 
                   cellprob_threshold=0.0,
                   min_size=15):
    """ 
        dP: torch.FloatTensor
        cellprob: torch.FloatTensor
    """
    
    cp_mask = cellprob > cellprob_threshold 

    if cp_mask.any(): #mask at this point is a cell cluster binary map, not labels     
        # follow flows
        print(datetime.now(), "Start to track flow map")
        p, inds = follow_flows(dP * cp_mask / 5., niter=niter)
        if inds is None:
            print(datetime.now(), "No nuclei in this chunk, return None")
            return None, None, None
        
        #calculate masks
        mask, coord, labels = get_masks(p, iscell=cp_mask)
        mask, coord, labels = mask.numpy(), coord.numpy(), labels.numpy()
            
        if mask.max() < 2**16:
            mask = mask.astype(np.uint16)

    else: # nothing to compute, just make it compatible
        print(datetime.now(), "No nuclei in this chunk, return None")
        return None, None, None

    return mask, coord, labels

def follow_flows(dP, niter=200):
    shape = dP.shape[1:]
    p = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]),
            torch.arange(shape[2]), indexing='ij')
    p = torch.stack(p).float()
    inds = torch.nonzero(dP[0].abs()>1e-3)
    ## 
    p = steps3D(p, dP, inds, niter)#.numpy()
    return p, inds

def steps3D(p, dP, inds, niter):
    ## Original code
    # shape = p.shape[1:]
    # for t in trange(niter):
    #     #pi = p.astype(np.int32)
    #     for j in trange(inds.shape[0]):
    #         z = inds[j,0]
    #         y = inds[j,1]
    #         x = inds[j,2]
    #         p0, p1, p2 = int(p[0,z,y,x]), int(p[1,z,y,x]), int(p[2,z,y,x])
    #         p[0,z,y,x] = min(shape[0]-1, max(0, p[0,z,y,x] + dP[0,p0,p1,p2]))
    #         p[1,z,y,x] = min(shape[1]-1, max(0, p[1,z,y,x] + dP[1,p0,p1,p2]))
    #         p[2,z,y,x] = min(shape[2]-1, max(0, p[2,z,y,x] + dP[2,p0,p1,p2]))
    # return p
    ## ChatGPT optimization
    shape = p.shape[1:]
    inds = inds.long()
    for t in range(niter):
        z = inds[:, 0]
        y = inds[:, 1]
        x = inds[:, 2]
        p0, p1, p2 = p[:, z, y, x].long()
        p[0, z, y, x] = torch.clip(p[0, z, y, x] + dP[0, p0, p1, p2], 0, shape[0] - 1)
        p[1, z, y, x] = torch.clip(p[1, z, y, x] + dP[1, p0, p1, p2], 0, shape[1] - 1)
        p[2, z, y, x] = torch.clip(p[2, z, y, x] + dP[2, p0, p1, p2], 0, shape[2] - 1)
        
    return p

def filename_to_depth(f, depth_start=0):
    return int(f.split('/')[-1].split('_resample')[1][:-4]) - depth_start

def get_masks(p, iscell=None, rpad=20):
    ## Original code #######################################################
    # pflows = []
    # edges = []
    # shape0 = p.shape[1:]
    # dims = len(p)
    # if iscell is not None:
    #     if dims==3:
    #         inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
    #             np.arange(shape0[2]), indexing='ij')
    #     elif dims==2:
    #         inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
    #                  indexing='ij')
    #     for i in range(dims):
    #         p[i, ~iscell] = inds[i][~iscell]

    # for i in range(dims):
    #     pflows.append(p[i].flatten().astype('int32'))
    #     edges.append(np.arange(-.5-rpad, shape0[i]+.5+rpad, 1))

    # h,_ = np.histogramdd(tuple(pflows), bins=edges)
    # hmax = h.copy()
    # for i in range(dims):
    #     hmax = maximum_filter1d(hmax, 5, axis=i)

    # seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
    # Nmax = h[seeds]
    # isort = np.argsort(Nmax)[::-1]
    # for s in seeds:
    #     s = s[isort]

    # # pix = list(np.array(seeds).T)
    # pix = np.array(seeds).T

    # shape = h.shape
    # if dims==3:
    #     expand = np.nonzero(np.ones((3,3,3)))
    # else:
    #     expand = np.nonzero(np.ones((3,3)))
    # for e in expand:
    #     e = np.expand_dims(e,1)

    # for iter in range(5):
    #     for k in range(len(pix)):
    #         if iter==0:
    #             pix[k] = list(pix[k])
    #         newpix = []
    #         iin = []
    #         for i,e in enumerate(expand):
    #             epix = e[:,np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
    #             epix = epix.flatten()
    #             iin.append(np.logical_and(epix>=0, epix<shape[i]))
    #             newpix.append(epix)
    #         iin = np.all(tuple(iin), axis=0)
    #         for p in newpix:
    #             p = p[iin]
    #         newpix = tuple(newpix)
    #         igood = h[newpix]>2
    #         for i in range(dims):
    #             pix[k][i] = newpix[i][igood]
    #         if iter==4:
    #             pix[k] = tuple(pix[k])
    
    # M = np.zeros(h.shape, np.uint32)
    # for k in range(len(pix)):
    #     M[pix[k]] = 1+k
        
    # for i in range(dims):
    #     pflows[i] = pflows[i] + rpad
    # M0 = M[tuple(pflows)]

    # # remove big masks
    # uniq, counts = fastremap.unique(M0, return_counts=True)
    # big = np.prod(shape0) * 0.4
    # bigc = uniq[counts > big]
    # if len(bigc) > 0 and (len(bigc)>1 or bigc[0]!=0):
    #     M0 = fastremap.mask(M0, bigc)
    # fastremap.renumber(M0, in_place=True) #convenient to guarantee non-skipped labels
    # M0 = np.reshape(M0, shape0)
    # return M0

    ## Optimized #####################################################
    iter_num = 3 # needs to be odd
    # p = torch.from_numpy(p)#.double()
    # iscell = torch.from_numpy(iscell)
    print(datetime.now(), 'Get cell center')
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims == 3:
            inds = torch.meshgrid(torch.arange(shape0[0]), torch.arange(shape0[1]),
                                  torch.arange(shape0[2]), indexing='ij')
        elif dims == 2:
            inds = torch.meshgrid(torch.arange(shape0[0]), torch.arange(shape0[1]),
                                  indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell].float()

    for i in range(dims):
        pflows.append(p[i].flatten().long())
        edges.append(torch.arange(-.5 - rpad, shape0[i] + .5 + rpad, 1))

    h = torchist.histogramdd(torch.stack(pflows, dim=1).double(), edges=edges)
    shape = h.shape
    hmax = h.clone().float()
    max_filter = torch.nn.MaxPool1d(kernel_size=iter_num, stride=1, padding=iter_num//2)
    for i in range(dims):
        hmax = max_filter(hmax.transpose(i, -1)).transpose(i, -1)
    seeds = torch.nonzero(torch.logical_and(h - hmax > -1e-6, h > 8), as_tuple=True)
    Nmax = h[seeds]
    isort = torch.argsort(Nmax, descending=True)
    seeds = tuple(seeds[i][isort] for i in range(len(seeds)))
    pix = torch.stack(seeds, dim=1)
    print(datetime.now(), 'Extend nuclei')
    #####################################
    pix = list(pix)
    if dims==3: # expand coordinates, center is (1, 1)
        expand = torch.nonzero(torch.ones((3,3,3))).T
    else:
        expand = torch.nonzero(torch.ones((3,3))).T
    expand = tuple(e.unsqueeze(1) for e in expand)
    for iter in range(iter_num): # expand all seeds four times
        for k in range(len(pix)): # expand each seed
            if iter==0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i,e in enumerate(expand): # for x, y, z of a 3x3 grid
                epix = e.unsqueeze(1) + pix[k][i].unsqueeze(0) - 1
                epix = epix.flatten()
                iin.append(torch.logical_and(epix>=0, epix<shape[i]))
                newpix.append(epix)
            iin = torch.stack(iin).all(0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix]>2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter==iter_num-1:
                pix[k] = tuple(pix[k])
                
    print(datetime.now(), 'Get coordinates and label mask')
    #####################################   
    coords = []
    M = torch.zeros(*shape, dtype=torch.long)
    remove_c = 0
    big = np.prod(shape0) * 0.4
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    fg = torch.zeros(*shape, dtype=bool)
    fg[tuple(pflows)] = True
    labels = []
    for k in range(len(pix)):
        if len(pix[k][0]) > big:
            remove_c += 1
            continue
        is_fg = fg[pix[k]]
        if not is_fg.any():
            remove_c += 1
            continue
        coord = (pix[k][0][is_fg], pix[k][1][is_fg], pix[k][2][is_fg])
        M[coord] = 1+k-remove_c
        coords.append(
            torch.cat([torch.stack(coord, -1), torch.zeros(len(coord[0]), 1, dtype=coord[0].dtype)+1+k-remove_c], -1)[is_fg, :]
        )
        labels.append(1+k-remove_c)
    M0 = M[tuple(pflows)]
    coords = torch.cat(coords)
    labels = torch.LongTensor(labels)
    # ## Code optimization end
    # remove big masks
    # print(datetime.now(), 'Remap mask label to [1, N]')
    # uniq, counts = fastremap.unique(M0, return_counts=True)
    # big = np.prod(shape0) * 0.4
    # bigc = uniq[counts > big]
    # if len(bigc) > 0 and (len(bigc)>1 or bigc[0]!=0):
    #     M0 = fastremap.mask(M0, bigc)
    # fastremap.renumber(M0, in_place=True) #convenient to guarantee non-skipped labels
    M0 = torch.reshape(M0, shape0)
    return M0, coords, labels

if __name__ == '__main__':
    main()