import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import logging
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import tempfile
import cv2
from scipy.stats import mode
import fastremap
import transforms, dynamics, metrics

import torch
#     from GPUtil import showUtilization as gpu_usage #for gpu memory debugging 
from torch import nn
from torch.utils import mkldnn as mkldnn_utils
import resnet_torch
TORCH_ENABLED = True
from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label, maximum_filter1d, binary_fill_holes

core_logger = logging.getLogger(__name__)

def get_masks_unet(output, cell_threshold=0, boundary_threshold=0):
    """ create masks using cell probability and cell boundary """
    cells = (output[...,1] - output[...,0])>cell_threshold
    selem = generate_binary_structure(cells.ndim, connectivity=1)
    labels, nlabels = label(cells, selem)

    if output.shape[-1]>2:
        slices = find_objects(labels)
        dists = 10000*np.ones(labels.shape, np.float32)
        mins = np.zeros(labels.shape, np.int32)
        borders = np.logical_and(~(labels>0), output[...,2]>boundary_threshold)
        pad = 10
        for i,slc in enumerate(slices):
            if slc is not None:
                slc_pad = tuple([slice(max(0,sli.start-pad), min(labels.shape[j], sli.stop+pad))
                                    for j,sli in enumerate(slc)])
                msk = (labels[slc_pad] == (i+1)).astype(np.float32)
                msk = 1 - gaussian_filter(msk, 5)
                dists[slc_pad] = np.minimum(dists[slc_pad], msk)
                mins[slc_pad][dists[slc_pad]==msk] = (i+1)
        labels[labels==0] = borders[labels==0] * mins[labels==0]
        
    masks = labels
    shape0 = masks.shape
    _,masks = np.unique(masks, return_inverse=True)
    masks = np.reshape(masks, shape0)
    return masks

def fill_holes_and_remove_small_masks(masks, min_size=15):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)
    
    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes

    (might have issues at borders between cells, todo: check and fix)
    
    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
        
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
    
    slices = find_objects(masks)
    j = 0
    for i,slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i+1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            elif npix > 0:   
                if msk.ndim==3:
                    for k in range(msk.shape[0]):
                        msk[k] = binary_fill_holes(msk[k])
                else:          
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = (j+1)
                j+=1
    return masks

def parse_model_string(pretrained_model):
    if isinstance(pretrained_model, list):
        model_str = os.path.split(pretrained_model[0])[-1]
    else:
        model_str = os.path.split(pretrained_model)[-1]
    if len(model_str)>3 and model_str[:4]=='unet':
        cp = False
        nclasses = max(2, int(model_str[4]))
    elif len(model_str)>7 and model_str[:8]=='cellpose':
        cp = True
        nclasses = 3
    else:
        return 3, True, True, False
    
    if 'residual' in model_str and 'style' in model_str and 'concatentation' in model_str:
        ostrs = model_str.split('_')[2::2]
        residual_on = ostrs[0]=='on'
        style_on = ostrs[1]=='on'
        concatenation = ostrs[2]=='on'
        return nclasses, residual_on, style_on, concatenation
    else:
        if cp:
            return 3, True, True, False
        else:
            return nclasses, False, False, True

def use_gpu(gpu_number=0, use_torch=True):
    """ check if gpu works """
    if use_torch:
        return _use_gpu_torch(gpu_number)
    else:
        raise ValueError('cellpose only runs with pytorch now')

def _use_gpu_torch(gpu_number=0):
    try:
        device = torch.device('cuda:' + str(gpu_number))
        _ = torch.zeros([1, 2, 3]).to(device)
        core_logger.info('** TORCH CUDA version installed and working. **')
        return True
    except:
        core_logger.info('TORCH CUDA version not installed/working.')
        return False

def assign_device(use_torch=True, gpu=False, device=0):
    if gpu and use_gpu(use_torch=True):
        device = torch.device(f'cuda:{device}')
        gpu=True
        core_logger.info('>>>> using GPU')
    else:
        device = torch.device('cpu')
        core_logger.info('>>>> using CPU')
        gpu=False
    return device, gpu

def check_mkl(use_torch=True):
    #core_logger.info('Running test snippet to check if MKL-DNN working')
    mkl_enabled = torch.backends.mkldnn.is_available()
    if mkl_enabled:
        mkl_enabled = True
        #core_logger.info('MKL version working - CPU version is sped up.')
    else:
        core_logger.info('WARNING: MKL version on torch not working/installed - CPU version will be slightly slower.')
        core_logger.info('see https://pytorch.org/docs/stable/backends.html?highlight=mkl')
    return mkl_enabled

class UnetModel():
    def __init__(self, gpu=False, pretrained_model=False,
                    diam_mean=30., net_avg=False, device=None,
                    residual_on=False, style_on=False, concatenation=True,
                    nclasses=3, nchan=2):
        self.unet = True
        self.torch = True
        self.mkldnn = None
        if device is None:
            sdevice, gpu = assign_device(self.torch, gpu)
        self.device = device if device is not None else sdevice
        if device is not None:
            device_gpu = self.device.type=='cuda'
        self.gpu = gpu if device is None else device_gpu
        if not self.gpu:
            self.mkldnn = check_mkl(True)
        self.pretrained_model = pretrained_model
        self.diam_mean = diam_mean

        ostr = ['off', 'on']
        self.net_type = 'unet{}_residual_{}_style_{}_concatenation_{}'.format(nclasses,
                                                                                ostr[residual_on],
                                                                                ostr[style_on],
                                                                                ostr[concatenation])                                             
        if pretrained_model:
            core_logger.info(f'u-net net type: {self.net_type}')
        # create network
        self.nclasses = nclasses
        self.nbase = [32,64,128,256]
        self.nchan = nchan
        self.nbase = [nchan, 32, 64, 128, 256]
        self.net = resnet_torch.CPnet(self.nbase, 
                                        self.nclasses, 
                                        sz=3,
                                        residual_on=residual_on, 
                                        style_on=style_on,
                                        concatenation=concatenation,
                                        mkldnn=self.mkldnn,
                                        diam_mean=diam_mean).to(self.device)
        
        if pretrained_model is not None and isinstance(pretrained_model, str):
            self.net.load_model(pretrained_model, cpu=(not self.gpu))

    def eval(self, x, batch_size=8, channels=None, channels_last=False, invert=False, normalize=True,
             rescale=None, do_3D=False, anisotropy=None, net_avg=False, augment=False,
             channel_axis=None, z_axis=None, nolist=False,
             tile=True, cell_threshold=None, boundary_threshold=None, min_size=15, 
             compute_masks=True):
        """ segment list of images x

            Parameters
            ----------
            x: list or array of images
                can be list of 2D/3D images, or array of 2D/3D images, or 4D image array

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            channel_axis: int (optional, default None)
                if None, channels dimension is attempted to be automatically determined

            z_axis: int (optional, default None)
                if None, z dimension is attempted to be automatically determined

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            normalize: bool (optional, default True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input

            anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

            net_avg: bool (optional, default False)
                runs the 4 built-in networks and averages them if True, runs one network if False

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            cell_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            boundary_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = flows at each pixel
                flows[k][2] = the cell distance field
                flows[k][3] = the cell boundary

            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

        """        
        x = [transforms.convert_image(xi, channels, channel_axis, z_axis, do_3D, 
                                    normalize, invert, nchan=self.nchan) for xi in x]
        nimg = len(x)
        self.batch_size = batch_size

        styles = []
        flows = []
        masks = []
        if rescale is None:
            rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)
        iterator = range(nimg)

        if isinstance(self.pretrained_model, list):
            model_path = self.pretrained_model[0]
            if not net_avg:
                self.net.load_model(self.pretrained_model[0])
        else:
            model_path = self.pretrained_model

        if cell_threshold is None or boundary_threshold is None:
            try:
                thresholds = np.load(model_path+'_cell_boundary_threshold.npy')
                cell_threshold, boundary_threshold = thresholds
                core_logger.info('>>>> found saved thresholds from validation set')
            except:
                core_logger.warning('WARNING: no thresholds found, using default / user input')

        cell_threshold = 2.0 if cell_threshold is None else cell_threshold
        boundary_threshold = 0.5 if boundary_threshold is None else boundary_threshold

        if not do_3D:
            for i in iterator:
                img = x[i].copy()
                shape = img.shape
                # rescale image for flow computation
                img = transforms.resize_image(img, rsz=rescale[i])
                y, style = self._run_nets(img, net_avg=net_avg, augment=augment, 
                                          tile=tile)
                if compute_masks:
                    maski = get_masks_unet(y, cell_threshold, boundary_threshold)
                    maski = fill_holes_and_remove_small_masks(maski, min_size=min_size)
                    maski = transforms.resize_image(maski, shape[-3], shape[-2], 
                                                        interpolation=cv2.INTER_NEAREST)
                else:
                    maski = None
                masks.append(maski)
                styles.append(style)
        else:
            for i in iterator:
                tic=time.time()
                yf, style = self._run_3D(x[i], rsz=rescale[i], anisotropy=anisotropy, 
                                         net_avg=net_avg, augment=augment, tile=tile)
                yf = yf.mean(axis=0)
                core_logger.info('probabilities computed %2.2fs'%(time.time()-tic))
                if compute_masks:
                    maski = get_masks_unet(yf.transpose((1,2,3,0)), cell_threshold, boundary_threshold)
                    maski = fill_holes_and_remove_small_masks(maski, min_size=min_size)
                else:
                    maski = None
                masks.append(maski)
                styles.append(style)
                core_logger.info('masks computed %2.2fs'%(time.time()-tic))
                flows.append(yf)

        if nolist:
            masks, flows, styles = masks[0], flows[0], styles[0]
        
        return masks, flows, styles

    def _to_device(self, x):
        X = torch.from_numpy(x).float().to(self.device)
        return X

    def _from_device(self, X):
        x = X.detach().cpu().numpy()
        return x

    def network(self, x, return_conv=False):
        """ convert imgs to torch and run network model and return numpy """
        X = self._to_device(x)
        self.net.eval()
        if self.mkldnn:
            self.net = mkldnn_utils.to_mkldnn(self.net)
        with torch.no_grad():
            y, style = self.net(X)
        del X
        y = self._from_device(y)
        style = self._from_device(style)
        if return_conv:
            conv = self._from_device(conv)
            y = np.concatenate((y, conv), axis=1)
        
        return y, style
                
    def _run_nets(self, img, net_avg=False, augment=False, tile=True, tile_overlap=0.1, bsize=224, 
                  return_conv=False, progress=None):
        """ run network (if more than one, loop over networks and average results

        Parameters
        --------------

        img: float, [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        net_avg: bool (optional, default False)
            runs the 4 built-in networks and averages them if True, runs one network if False

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU memory usage limited (recommended)

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI

        Returns
        ------------------

        y: array [3 x Ly x Lx] or [3 x Lz x Ly x Lx]
            y is output (averaged over networks);
            y[0] is Y flow; y[1] is X flow; y[2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles,
            but not averaged over networks.

        """
        if isinstance(self.pretrained_model, str) or not net_avg:  
            y, style = self._run_net(img, augment=augment, tile=tile, tile_overlap=tile_overlap,
                                     bsize=bsize, return_conv=return_conv)
        else:  
            for j in range(len(self.pretrained_model)):
                self.net.load_model(self.pretrained_model[j], cpu=(not self.gpu))
                y0, style = self._run_net(img, augment=augment, tile=tile, 
                                          tile_overlap=tile_overlap, bsize=bsize,
                                          return_conv=return_conv)

                if j==0:
                    y = y0
                else:
                    y += y0
                if progress is not None:
                    progress.setValue(10 + 10*j)
            y = y / len(self.pretrained_model)
            
        return y, style

    def _run_net(self, imgs, augment=False, tile=True, tile_overlap=0.1, bsize=224,
                 return_conv=False):
        """ run network on image or stack of images

        (faster if augment is False)

        Parameters
        --------------

        imgs: array [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient(s) for image

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended);
            cannot be turned off for 3D segmentation

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]

        Returns
        ------------------

        y: array [Ly x Lx x 3] or [Lz x Ly x Lx x 3]
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles

        """   
        if imgs.ndim==4:  
            # make image Lz x nchan x Ly x Lx for net
            imgs = np.transpose(imgs, (0,3,1,2)) 
            detranspose = (0,2,3,1)
            return_conv = False
        else:
            # make image nchan x Ly x Lx for net
            imgs = np.transpose(imgs, (2,0,1))
            detranspose = (1,2,0)

        # pad image for net so Ly and Lx are divisible by 4
        imgs, ysub, xsub = transforms.pad_image_ND(imgs)
        # slices from padding
#         slc = [slice(0, self.nclasses) for n in range(imgs.ndim)] # changed from imgs.shape[n]+1 for first slice size 
        slc = [slice(0, imgs.shape[n]+1) for n in range(imgs.ndim)]
        slc[-3] = slice(0, self.nclasses + 32*return_conv + 1)
        slc[-2] = slice(ysub[0], ysub[-1]+1)
        slc[-1] = slice(xsub[0], xsub[-1]+1)
        slc = tuple(slc)

        # run network
        if tile or augment or imgs.ndim==4:
            y, style = self._run_tiled(imgs, augment=augment, bsize=bsize, 
                                      tile_overlap=tile_overlap, 
                                      return_conv=return_conv)
        else:
            imgs = np.expand_dims(imgs, axis=0)
            y, style = self.network(imgs, return_conv=return_conv)
            y, style = y[0], style[0]
        style /= (style**2).sum()**0.5

        # slice out padding
        y = y[slc]
        # transpose so channels axis is last again
        y = np.transpose(y, detranspose)
        
        return y, style
    
    def _run_tiled(self, imgi, augment=False, bsize=224, tile_overlap=0.1, return_conv=False):
        """ run network in tiles of size [bsize x bsize]

        First image is split into overlapping tiles of size [bsize x bsize].
        If augment, tiles have 50% overlap and are flipped at overlaps.
        The average of the network output over tiles is returned.

        Parameters
        --------------

        imgi: array [nchan x Ly x Lx] or [Lz x nchan x Ly x Lx]

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]
         
        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        Returns
        ------------------

        yf: array [3 x Ly x Lx] or [Lz x 3 x Ly x Lx]
            yf is averaged over tiles
            yf[0] is Y flow; yf[1] is X flow; yf[2] is cell probability

        styles: array [64]
            1D array summarizing the style of the image, averaged over tiles

        """
        if imgi.ndim==4:
            batch_size = self.batch_size 
            Lz, nchan = imgi.shape[:2]
            IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[0], bsize=bsize, 
                                                            augment=augment, tile_overlap=tile_overlap)
            ny, nx, nchan, ly, lx = IMG.shape
            batch_size *= max(4, (bsize**2 // (ly*lx))**0.5)
            yf = np.zeros((Lz, self.nclasses, imgi.shape[-2], imgi.shape[-1]), np.float32)
            styles = []
            if ny*nx > batch_size:
                ziterator = range(Lz)
                for i in ziterator:
                    yfi, stylei = self._run_tiled(imgi[i], augment=augment, 
                                                  bsize=bsize, tile_overlap=tile_overlap)
                    yf[i] = yfi
                    styles.append(stylei)
            else:
                # run multiple slices at the same time
                ntiles = ny*nx
                nimgs = max(2, int(np.round(batch_size / ntiles)))
                niter = int(np.ceil(Lz/nimgs))
                ziterator = range(niter)
                for k in ziterator:
                    IMGa = np.zeros((ntiles*nimgs, nchan, ly, lx), np.float32)
                    for i in range(min(Lz-k*nimgs, nimgs)):
                        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[k*nimgs+i], bsize=bsize, 
                                                                        augment=augment, tile_overlap=tile_overlap)
                        IMGa[i*ntiles:(i+1)*ntiles] = np.reshape(IMG, (ny*nx, nchan, ly, lx))
                    ya, stylea = self.network(IMGa)
                    for i in range(min(Lz-k*nimgs, nimgs)):
                        y = ya[i*ntiles:(i+1)*ntiles]
                        if augment:
                            y = np.reshape(y, (ny, nx, 3, ly, lx))
                            y = transforms.unaugment_tiles(y, self.unet)
                            y = np.reshape(y, (-1, 3, ly, lx))
                        yfi = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
                        yfi = yfi[:,:imgi.shape[2],:imgi.shape[3]]
                        yf[k*nimgs+i] = yfi
                        stylei = stylea[i*ntiles:(i+1)*ntiles].sum(axis=0)
                        stylei /= (stylei**2).sum()**0.5
                        styles.append(stylei)
            return yf, np.array(styles)
        else:
            IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=bsize, 
                                                            augment=augment, tile_overlap=tile_overlap)
            ny, nx, nchan, ly, lx = IMG.shape
            IMG = np.reshape(IMG, (ny*nx, nchan, ly, lx))
            batch_size = self.batch_size
            niter = int(np.ceil(IMG.shape[0] / batch_size))
            nout = self.nclasses + 32*return_conv
            y = np.zeros((IMG.shape[0], nout, ly, lx))
            for k in range(niter):
                irange = np.arange(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
                y0, style = self.network(IMG[irange], return_conv=return_conv)
                y[irange] = y0.reshape(len(irange), y0.shape[-3], y0.shape[-2], y0.shape[-1])
                if k==0:
                    styles = style[0]
                styles += style.sum(axis=0)
            styles /= IMG.shape[0]
            if augment:
                y = np.reshape(y, (ny, nx, nout, bsize, bsize))
                y = transforms.unaugment_tiles(y, self.unet)
                y = np.reshape(y, (-1, nout, bsize, bsize))
            
            yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
            yf = yf[:,:imgi.shape[1],:imgi.shape[2]]
            styles /= (styles**2).sum()**0.5
            return yf, styles

    def _run_3D(self, imgs, rsz=1.0, anisotropy=None, net_avg=False, 
                augment=False, tile=True, tile_overlap=0.1, 
                bsize=224, progress=None, do_sim_Z=False):
        """ run network on stack of images

        (faster if augment is False)

        Parameters
        --------------

        imgs: array [Lz x Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient(s) for image

        anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

        net_avg: bool (optional, default False)
            runs the 4 built-in networks and averages them if True, runs one network if False

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended);
            cannot be turned off for 3D segmentation

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]

        progress: pyqt progress bar (optional, default None)
            to return progress bar status to GUI


        Returns
        ------------------

        yf: array [Lz x Ly x Lx x 3]
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles

        """ 
        sstr = ['YX', 'ZY', 'ZX']
        if anisotropy is not None:
            rescaling = [[rsz, rsz],
                         [rsz*anisotropy, rsz],
                         [rsz*anisotropy, rsz]]
        else:
            rescaling = [rsz] * 3
        pm = [(0,1,2,3), (1,0,2,3), (2,0,1,3)]
        ipm = [(3,0,1,2), (3,1,0,2), (3,1,2,0)]
        yf = np.zeros((3, self.nclasses, imgs.shape[0], imgs.shape[1], imgs.shape[2]), np.float32)
        for p in range(3 - 2*self.unet):
            xsl = imgs.copy().transpose(pm[p])
            # rescale image for flow computation
            shape = xsl.shape
            xsl = transforms.resize_image(xsl, rsz=rescaling[p])  
            # per image
            core_logger.info('running %s: %d planes of size (%d, %d)'%(sstr[p], shape[0], shape[1], shape[2]))
            y, style = self._run_nets(xsl, net_avg=net_avg, augment=augment, tile=tile, 
                                      bsize=bsize, tile_overlap=tile_overlap)
            y = transforms.resize_image(y, shape[1], shape[2])    
            yf[p] = y.transpose(ipm[p])
            if progress is not None:
                progress.setValue(25+15*p)
            if do_sim_Z: break
        return yf, style
