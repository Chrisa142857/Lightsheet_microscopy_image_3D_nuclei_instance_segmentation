import time, os
from scipy.ndimage.filters import maximum_filter1d
# import torch
import scipy.ndimage
import numpy as np
import tifffile
from tqdm import trange
from numba import njit, float32, int32, vectorize
import cv2
import fastremap

import logging
dynamics_logger = logging.getLogger(__name__)

import core, metrics, transforms

import torch, torchist

TORCH_ENABLED = True 
torch_GPU = torch.device('cuda')
torch_CPU = torch.device('cpu')

@njit('(float64[:], int32[:], int32[:], int32, int32, int32, int32)', nogil=True)
def _extend_centers(T,y,x,ymed,xmed,Lx, niter):
    """ run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)
    Parameters
    --------------
    T: float64, array
        _ x Lx array that diffusion is run in
    y: int32, array
        pixels in y inside mask
    x: int32, array
        pixels in x inside mask
    ymed: int32
        center of mask in y
    xmed: int32
        center of mask in x
    Lx: int32
        size of x-dimension of masks
    niter: int32
        number of iterations to run diffusion
    Returns
    ---------------
    T: float64, array
        amount of diffused particles at each pixel
    """

    for t in range(niter):
        T[ymed*Lx + xmed] += 1
        T[y*Lx + x] = 1/9. * (T[y*Lx + x] + T[(y-1)*Lx + x]   + T[(y+1)*Lx + x] +
                                            T[y*Lx + x-1]     + T[y*Lx + x+1] +
                                            T[(y-1)*Lx + x-1] + T[(y-1)*Lx + x+1] +
                                            T[(y+1)*Lx + x-1] + T[(y+1)*Lx + x+1])
    return T



def _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, n_iter=200, device=torch.device('cuda')):
    """ runs diffusion on GPU to generate flows for training images or quality control
    
    neighbors is 9 x pixels in masks, 
    centers are mask centers, 
    isneighbor is valid neighbor boolean 9 x pixels
    
    """
    if device is not None:
        device = device
    nimg = neighbors.shape[0] // 9
    pt = torch.from_numpy(neighbors).to(device)
    
    T = torch.zeros((nimg,Ly,Lx), dtype=torch.double, device=device)
    meds = torch.from_numpy(centers.astype(int)).to(device).long()
    isneigh = torch.from_numpy(isneighbor).to(device)
    for i in range(n_iter):
        T[:, meds[:,0], meds[:,1]] +=1
        Tneigh = T[:, pt[:,:,0], pt[:,:,1]]
        Tneigh *= isneigh
        T[:, pt[0,:,0], pt[0,:,1]] = Tneigh.mean(axis=1)
    del meds, isneigh, Tneigh
    T = torch.log(1.+ T)
    # gradient positions
    grads = T[:, pt[[2,1,4,3],:,0], pt[[2,1,4,3],:,1]]
    del pt
    dy = grads[:,0] - grads[:,1]
    dx = grads[:,2] - grads[:,3]
    del grads
    mu_torch = np.stack((dy.cpu().squeeze(), dx.cpu().squeeze()), axis=-2)
    return mu_torch


def masks_to_flows_gpu(masks, device=None):
    """ convert masks to flows using diffusion from center pixel
    Center of masks where diffusion starts is defined using COM
    Parameters
    -------------
    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    Returns
    -------------
    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].
    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 
    """
    if device is None:
        device = torch.device('cuda')

    
    Ly0,Lx0 = masks.shape
    Ly, Lx = Ly0+2, Lx0+2

    masks_padded = np.zeros((Ly, Lx), np.int64)
    masks_padded[1:-1, 1:-1] = masks

    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)
    neighborsY = np.stack((y, y-1, y+1, 
                           y, y, y-1, 
                           y-1, y+1, y+1), axis=0)
    neighborsX = np.stack((x, x, x, 
                           x-1, x+1, x-1, 
                           x+1, x-1, x+1), axis=0)
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)

    # get mask centers
    slices = scipy.ndimage.find_objects(masks)
    
    centers = np.zeros((masks.max(), 2), 'int')
    for i,si in enumerate(slices):
        if si is not None:
            sr,sc = si
            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            yi,xi = np.nonzero(masks[sr, sc] == (i+1))
            yi = yi.astype(np.int32) + 1 # add padding
            xi = xi.astype(np.int32) + 1 # add padding
            ymed = np.median(yi)
            xmed = np.median(xi)
            imin = np.argmin((xi-xmed)**2 + (yi-ymed)**2)
            xmed = xi[imin]
            ymed = yi[imin]
            centers[i,0] = ymed + sr.start 
            centers[i,1] = xmed + sc.start

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:,:,0], neighbors[:,:,1]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = np.array([[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices])
    n_iter = 2 * (ext.sum(axis=1)).max()
    # run diffusion
    mu = _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, 
                             n_iter=n_iter, device=device)

    # normalize
    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y-1, x-1] = mu
    mu_c = np.zeros_like(mu0)
    return mu0, mu_c



def masks_to_flows(masks, use_gpu=False, device=None):
    """ convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map. 

    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].

    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 

    """
    if masks.max() == 0:
        dynamics_logger.warning('empty masks!')
        return np.zeros((2, *masks.shape), 'float32')

    if use_gpu:
        if use_gpu and device is None:
            device = torch_GPU
        elif device is None:
            device = torch_CPU
        masks_to_flows_device = masks_to_flows_gpu
    # else:
    #     masks_to_flows_device = masks_to_flows_cpu
        
    if masks.ndim==3:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_device(masks[z], device=device)[0]
            mu[[1,2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_device(masks[:,y], device=device)[0]
            mu[[0,2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_device(masks[:,:,x], device=device)[0]
            mu[[0,1], :, :, x] += mu0
        return mu
    elif masks.ndim==2:
        mu, mu_c = masks_to_flows_device(masks, device=device)
        return mu

    else:
        raise ValueError('masks_to_flows only takes 2D or 3D arrays')

def labels_to_flows(labels, files=None, use_gpu=False, device=None, redo_flows=False):
    """ convert labels (list of masks or flows) to flows for training model 

    if files is not None, flows are saved to files to be reused

    Parameters
    --------------

    labels: list of ND-arrays
        labels[k] can be 2D or 3D, if [3 x Ly x Lx] then it is assumed that flows were precomputed.
        Otherwise labels[k][0] or labels[k] (if 2D) is used to create flows and cell probabilities.

    Returns
    --------------

    flows: list of [4 x Ly x Lx] arrays
        flows[k][0] is labels[k], flows[k][1] is cell distance transform, flows[k][2] is Y flow,
        flows[k][3] is X flow, and flows[k][4] is heat distribution

    """
    nimg = len(labels)
    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis,:,:] for n in range(nimg)]

    if labels[0].shape[0] == 1 or labels[0].ndim < 3 or redo_flows: # flows need to be recomputed
        
        dynamics_logger.info('computing flows for labels')
        
        # compute flows; labels are fixed here to be unique, so they need to be passed back
        # make sure labels are unique!
        labels = [fastremap.renumber(label, in_place=True)[0] for label in labels]
        veci = [masks_to_flows(labels[n][0],use_gpu=use_gpu, device=device) for n in trange(nimg)]
        
        # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
        flows = [np.concatenate((labels[n], labels[n]>0.5, veci[n]), axis=0).astype(np.float32)
                    for n in range(nimg)]
        if files is not None:
            for flow, file in zip(flows, files):
                file_name = os.path.splitext(file)[0]
                tifffile.imsave(file_name+'_flows.tif', flow)
    else:
        dynamics_logger.info('flows precomputed')
        flows = [labels[n].astype(np.float32) for n in range(nimg)]
    return flows


@njit(['(int16[:,:,:], float32[:], float32[:], float32[:,:])', 
        '(float32[:,:,:], float32[:], float32[:], float32[:,:])'], cache=True)
def map_coordinates(I, yc, xc, Y):
    """
    bilinear interpolation of image 'I' in-place with ycoordinates yc and xcoordinates xc to Y
    
    Parameters
    -------------
    I : C x Ly x Lx
    yc : ni
        new y coordinates
    xc : ni
        new x coordinates
    Y : C x ni
        I sampled at (yc,xc)
    """
    C,Ly,Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        yf = min(Ly-1, max(0, yc_floor[i]))
        xf = min(Lx-1, max(0, xc_floor[i]))
        yf1= min(Ly-1, yf+1)
        xf1= min(Lx-1, xf+1)
        y = yc[i]
        x = xc[i]
        for c in range(C):
            Y[c,i] = (np.float32(I[c, yf, xf]) * (1 - y) * (1 - x) +
                      np.float32(I[c, yf, xf1]) * (1 - y) * x +
                      np.float32(I[c, yf1, xf]) * y * (1 - x) +
                      np.float32(I[c, yf1, xf1]) * y * x )


def steps2D_interp(p, dP, niter, use_gpu=False, device=None):
    shape = dP.shape[1:]
    if use_gpu:
        if device is None:
            device = torch_GPU
        shape = np.array(shape)[[1,0]].astype('float')-1  # Y and X dimensions (dP is 2.Ly.Lx), flipped X-1, Y-1
        pt = torch.from_numpy(p[[1,0]].T).float().to(device).unsqueeze(0).unsqueeze(0) # p is n_points by 2, so pt is [1 1 2 n_points]
        im = torch.from_numpy(dP[[1,0]]).float().to(device).unsqueeze(0) #covert flow numpy array to tensor on GPU, add dimension 
        # normalize pt between  0 and  1, normalize the flow
        for k in range(2): 
            im[:,k,:,:] *= 2./shape[k]
            pt[:,:,:,k] /= shape[k]
            
        # normalize to between -1 and 1
        pt = pt*2-1 
        
        #here is where the stepping happens
        for t in range(niter):
            # align_corners default is False, just added to suppress warning
            dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
            
            for k in range(2): #clamp the final pixel locations
                pt[:,:,:,k] = torch.clamp(pt[:,:,:,k] + dPt[:,k,:,:], -1., 1.)
            

        #undo the normalization from before, reverse order of operations 
        pt = (pt+1)*0.5
        for k in range(2): 
            pt[:,:,:,k] *= shape[k]        
        
        p =  pt[:,:,:,[1,0]].cpu().numpy().squeeze().T
        return p

    else:
        dPt = np.zeros(p.shape, np.float32)
            
        for t in range(niter):
            map_coordinates(dP.astype(np.float32), p[0], p[1], dPt)
            for k in range(len(p)):
                p[k] = np.minimum(shape[k]-1, np.maximum(0, p[k] + dPt[k]))
        return p


# @njit('(float32[:,:,:,:],float32[:,:,:,:], int32[:,:], int32)', nogil=True)
def steps3D(p, dP, inds, niter):
    """ run dynamics of pixels to recover masks in 3D
    
    Euler integration of dynamics dP for niter steps

    Parameters
    ----------------

    p: float32, 4D array
        pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)

    dP: float32, 4D array
        flows [axis x Lz x Ly x Lx]

    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 3]

    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------

    p: float32, 4D array
        final locations of each pixel after dynamics

    """
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
        # p0 = p[0, z, y, x].long()
        # p1 = p[1, z, y, x].long()
        # p2 = p[2, z, y, x].long()
        
        p[0, z, y, x] = torch.clip(p[0, z, y, x] + dP[0, p0, p1, p2], 0, shape[0] - 1)
        p[1, z, y, x] = torch.clip(p[1, z, y, x] + dP[1, p0, p1, p2], 0, shape[1] - 1)
        p[2, z, y, x] = torch.clip(p[2, z, y, x] + dP[2, p0, p1, p2], 0, shape[2] - 1)
        
    return p

@njit('(float32[:,:,:], float32[:,:,:], int32[:,:], int32)', nogil=True)
def steps2D(p, dP, inds, niter):
    """ run dynamics of pixels to recover masks in 2D
    
    Euler integration of dynamics dP for niter steps

    Parameters
    ----------------

    p: float32, 3D array
        pixel locations [axis x Ly x Lx] (start at initial meshgrid)

    dP: float32, 3D array
        flows [axis x Ly x Lx]

    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 2]

    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------

    p: float32, 3D array
        final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        for j in range(inds.shape[0]):
            # starting coordinates
            y = inds[j,0]
            x = inds[j,1]
            p0, p1 = int(p[0,y,x]), int(p[1,y,x])
            step = dP[:,p0,p1]
            for k in range(p.shape[0]):
                p[k,y,x] = min(shape[k]-1, max(0, p[k,y,x] + step[k]))
    return p

def follow_flows(dP, mask=None, niter=200, interp=True, use_gpu=True, device=None):
    """ define pixels and run dynamics to recover masks in 2D
    
    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------

    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
    
    mask: (optional, default None)
        pixel mask to seed masks. Useful when flows have low magnitudes.

    niter: int (optional, default 200)
        number of iterations of dynamics to run

    interp: bool (optional, default True)
        interpolate during 2D dynamics (not available in 3D) 
        (in previous versions + paper it was False)

    use_gpu: bool (optional, default False)
        use GPU to run interpolated dynamics (faster than CPU)


    Returns
    ---------------

    p: float32, 3D or 4D array
        final locations of each pixel after dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    inds: int32, 3D or 4D array
        indices of pixels used for dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)
    if len(shape)>2:
        ## orig code
        # p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
        #         np.arange(shape[2]), indexing='ij')
        # p = np.array(p).astype(np.float32)
        # # run dynamics on subset of pixels
        # inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T\
        ## optimized 
        shape = dP.shape[1:]
        p = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]),
                torch.arange(shape[2]), indexing='ij')
        p = torch.stack(p).float()
        inds = torch.nonzero(dP[0].abs()>1e-3)
        ## 
        p = steps3D(p, dP, inds, niter)#.numpy()
        # inds = inds.numpy().astype(np.int32)
    else:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        p = np.array(p).astype(np.float32)

        inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
        
        if inds.ndim < 2 or inds.shape[0] < 5:
            dynamics_logger.warning('WARNING: no mask pixels found')
            return p, None
        
        if not interp:
            p = steps2D(p, dP.astype(np.float32), inds, niter)
            
        else:
            p_interp = steps2D_interp(p[:,inds[:,0], inds[:,1]], dP, niter, use_gpu=use_gpu, device=device)            
            p[:,inds[:,0],inds[:,1]] = p_interp
    return p, inds

def remove_bad_flow_masks(masks, flows, threshold=0.4, use_gpu=False, device=None):
    """ remove masks which have inconsistent flows 
    
    Uses metrics.flow_error to compute flows from predicted masks 
    and compare flows to predicted flows from network. Discards 
    masks with flow errors greater than the threshold.

    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    flows: float, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded.

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    merrors, _ = metrics.flow_error(masks, flows, use_gpu, device)
    badi = 1+(merrors>threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks

def get_masks(p, iscell=None, rpad=20):
    """ create masks using pixel convergence after running dynamics
    
    Makes a histogram of final pixel locations p, initializes masks 
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards 
    masks with flow errors greater than the threshold. 
    Parameters
    ----------------
    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are 
        iscell False to stay in their original location.
    rpad: int (optional, default 20)
        histogram edge padding
    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded 
        (if flows is not None)
    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using 
        `remove_bad_flow_masks`.
    Returns
    ---------------
    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
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
    # print(datetime.now(), 'Get cell center')
    iter_num = 3 # needs to be odd
    # p = torch.from_numpy(p)#.double()
    # iscell = torch.from_numpy(iscell)
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
    #####################################
    # h = h.cpu().numpy()
    # pix = list(pix.cpu().numpy())
    # if dims==3: # expand coordinates, center is (1, 1)
    #     expand = np.nonzero(np.ones((3,3,3)))
    # else:
    #     expand = np.nonzero(np.ones((3,3)))
    # for e in expand:
    #     e = np.expand_dims(e,1)
    # for iter in range(iter_num): # expand all seeds four times
    #     for k in range(len(pix)): # expand each seed
    #         if iter==0:
    #             pix[k] = list(pix[k])
    #         newpix = []
    #         iin = []
    #         for i,e in enumerate(expand): # for x, y, z of a 3x3 grid
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
    #         if iter==iter_num-1:
    #             # pix[k] = torch.from_numpy(np.array(pix[k])).T
    #             pix[k] = tuple(pix[k])
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
########################################

    M = torch.zeros(*shape, dtype=torch.long)
    # M = np.zeros(h.shape, np.uint32)
    remove_bigc = 0
    big = np.prod(shape0) * 0.4
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    fg = torch.zeros(*shape, dtype=bool)
    fg[tuple(pflows)] = True
    for k in range(len(pix)):
        if len(pix[k][0]) > big:
            remove_bigc += 1
            continue
        is_fg = fg[pix[k]]
        if is_fg.any():
            coord = (pix[k][0][is_fg], pix[k][1][is_fg], pix[k][2][is_fg])
            M[coord] = 1+k-remove_bigc
        # M[pix[k]] = 1+k-remove_bigc
        
    M0 = M[tuple(pflows)]
    # M0 = M.numpy()
    ###########################################
    # pix = pix.numpy()
    # expand_s = iter_num * 2 + 1 # total expand size = iter * 2 + 1
    # M = np.zeros(h.shape, np.uint32)
    # if dims==3: # expand coordinates, center is (1, 1)
    #     expand = np.nonzero(np.ones((expand_s, expand_s, expand_s)))
    # else:
    #     expand = np.nonzero(np.ones((expand_s, expand_s)))
    # for e in expand:
    #     e = np.expand_dims(e,1)
    
    # for k in range(len(pix)): # expand each seed
    #     epix = []
    #     iin = []
    #     for i in range(dims):
    #         epix.append(pix[k, i] + expand[i] - iter_num)
    #         iin.append(np.logical_and(epix[-1]>=0, epix[-1]<shape[i]))
        
    #     iin = np.all(tuple(iin), axis=0)
    #     for i in range(dims):
    #         epix[i] = epix[i][iin]

    #     igood = h[tuple(epix)]>2
    #     for i in range(dims):
    #         epix[i] = epix[i][igood]
    
    #     M[tuple(epix)] = 1+k
        
    # for i in range(dims):
    #     pflows[i] = pflows[i] + rpad
    # M0 = M[tuple(pflows)]
    ########################################
    # expand_s = iter_num * 2 + 1 # total expand size = iter * 2 + 1
    # M = torch.zeros(*shape, dtype=torch.int64)
    # if dims==3: # expand coordinates, center is (1, 1)
    #     expand = torch.nonzero(torch.ones(expand_s, expand_s, expand_s))
    # else:
    #     expand = torch.nonzero(torch.ones(expand_s, expand_s))
    # for e in expand:
    #     e = torch.unsqueeze(e,1)
    
    # for k in range(len(pix)): # expand each seed
    #     epix = []
    #     iin = []
    #     for i in range(dims):
    #         epix.append(pix[k, i] + expand[i] - iter_num)
    #         iin.append(torch.logical_and(epix[-1]>=0, epix[-1]<shape[i]))
        
    #     iin = torch.all(torch.stack(iin), dim=0)
    #     for i in range(dims):
    #         epix[i] = epix[i][iin]

    #     igood = h[tuple(epix)]>2
    #     for i in range(dims):
    #         epix[i] = epix[i][igood]
    
    #     M[tuple(epix)] = 1+k
        
    # for i in range(dims):
    #     pflows[i] = pflows[i] + rpad
    # M0 = M[tuple(pflows)].numpy()
    # ## Code optimization end
    
    # remove big masks
    # uniq, counts = fastremap.unique(M0, return_counts=True)
    # print(counts.max())
    # big = np.prod(shape0) * 0.4
    # bigc = uniq[counts > big]
    # if len(bigc) > 0 and (len(bigc)>1 or bigc[0]!=0):
    #     M0 = fastremap.mask(M0, bigc)
    # fastremap.renumber(M0, in_place=True) #convenient to guarantee non-skipped labels
    M0 = torch.reshape(M0, shape0).numpy()
    return M0

def compute_masks(dP, cellprob, p=None, niter=200, 
                   cellprob_threshold=0.0,
                   flow_threshold=0.4, interp=True, do_3D=False, 
                   min_size=15, resize=None, 
                   use_gpu=False,device=None):
    """ compute masks using dynamics from dP, cellprob, and boundary """
    
    cp_mask = cellprob > cellprob_threshold 

    if np.any(cp_mask): #mask at this point is a cell cluster binary map, not labels     
        # follow flows
        if do_3D:
            dP = torch.from_numpy(dP)
            cp_mask = torch.from_numpy(cp_mask)
        if p is None:
            if dP.shape[1] != cp_mask.shape[0]:
                cp_mask = torch.nn.functional.interpolate(cp_mask.unsqueeze(0).unsqueeze(0) * 1.0, size=(dP.shape[1], dP.shape[2], dP.shape[3]), mode='nearest-exact').squeeze().numpy() > 0
            p, inds = follow_flows(dP * cp_mask / 5., niter=niter, interp=interp, 
                                            use_gpu=use_gpu, device=device)
            if inds is None:
                dynamics_logger.info('No cell pixels found.')
                shape = resize if resize is not None else cellprob.shape
                mask = np.zeros(shape, np.uint16)
                p = np.zeros((len(shape), *shape), np.uint16)
                return mask, p
        
        #calculate masks
        mask = get_masks(p, iscell=cp_mask)
            
        # flow thresholding factored out of get_masks
        if not do_3D:
            shape0 = p.shape[1:]
            if mask.max()>0 and flow_threshold is not None and flow_threshold > 0:
                # make sure labels are unique at output of get_masks
                mask = remove_bad_flow_masks(mask, dP, threshold=flow_threshold, use_gpu=use_gpu, device=device)
        
        if resize is not None:
            #if verbose:
            #    dynamics_logger.info(f'resizing output with resize = {resize}')
            if mask.max() > 2**16-1:
                recast = True
                mask = mask.astype(np.float32)
            else:
                recast = False
                mask = mask.astype(np.uint16)
            mask = transforms.resize_image(mask, resize[0], resize[1], interpolation=cv2.INTER_NEAREST)
            if recast:
                mask = mask.astype(np.uint32)
            Ly,Lx = mask.shape
        elif mask.max() < 2**16:
            mask = mask.astype(np.uint16)

    else: # nothing to compute, just make it compatible
        dynamics_logger.info('No cell pixels found.')
        shape = resize if resize is not None else cellprob.shape
        mask = np.zeros(shape, np.uint16)
        p = np.zeros((len(shape), *shape), np.uint16)
        return mask, p


    # moving the cleanup to the end helps avoid some bugs arising from scaling...
    # maybe better would be to rescale the min_size and hole_size parameters to do the
    # cleanup at the prediction scale, or switch depending on which one is bigger... 
    # mask = core.fill_holes_and_remove_small_masks(mask, min_size=min_size)

    if mask.dtype==np.uint32:
        dynamics_logger.warning('more than 65535 masks in image, masks returned as np.uint32')

    return mask, p

