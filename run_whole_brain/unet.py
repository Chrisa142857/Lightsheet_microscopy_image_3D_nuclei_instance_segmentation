import os#, sys, time, shutil, tempfile, datetime, pathlib, subprocess
from pathlib import Path
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datetime import datetime

import transforms
import resnet
import utils


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


class UnetModel():
    def __init__(self, gpu=False, pretrained_model=False,
                    diam_mean=30., net_avg=False, device=None,
                    residual_on=False, style_on=False, concatenation=True,
                    nclasses=3, nchan=2):
        self.unet = True
        self.torch = True
        self.mkldnn = None
        if device is None:
            # sdevice, gpu = assign_device(self.torch, gpu)
            print('You need to set device')
            exit()
        self.device = device #if device is not None else sdevice
        if device is not None:
            device_gpu = self.device.type=='cuda'
        self.gpu = device_gpu
        # if not self.gpu:
        #     self.mkldnn = check_mkl(True)
        self.pretrained_model = pretrained_model
        self.diam_mean = diam_mean

        ostr = ['off', 'on']
        self.net_type = 'unet{}_residual_{}_style_{}_concatenation_{}'.format(nclasses,
                                                                                ostr[residual_on],
                                                                                ostr[style_on],
                                                                                ostr[concatenation])                                             
        # if pretrained_model:
        #     core_logger.info(f'u-net net type: {self.net_type}')
        # create network
        self.nclasses = nclasses
        self.nbase = [32,64,128,256]
        self.nchan = nchan
        self.nbase = [nchan, 32, 64, 128, 256]
        self.net = resnet.CPnet(self.nbase, 
                                        self.nclasses, 
                                        sz=3,
                                        residual_on=residual_on, 
                                        style_on=style_on,
                                        concatenation=concatenation,
                                        mkldnn=self.mkldnn,
                                        diam_mean=diam_mean).to(self.device)
        
        if pretrained_model is not None and isinstance(pretrained_model, str):
            self.net.load_model(pretrained_model, cpu=(not self.gpu))

    def _to_device(self, x):
        X = torch.from_numpy(x).float().to(self.device)
        return X

    def _from_device(self, X):
        x = X.detach().cpu().numpy()
        return x

    def network(self, x, return_conv=False):
        """ convert imgs to torch and run network model and return numpy """
        # X = self._to_device(x)
        # X  = x.to(self.device)
        # if self.mkldnn:
        #     self.net = mkldnn_utils.to_mkldnn(self.net)
        with torch.no_grad():
            y, style = self.net(x)
        # del X
        # y = self._from_device(y)
        # style = self._from_device(style)
        # if return_conv:
        #     conv = self._from_device(conv)
        #     y = np.concatenate((y, conv), axis=1)
        
        return y, style
                

    def _run_nets(self, img, net_avg=False, tile=True, tile_overlap=0.1, bsize=224, 
                  return_conv=False, progress=None):
        """ run network (if more than one, loop over networks and average results

        Parameters
        --------------

        img: float, [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        net_avg: bool (optional, default False)
            runs the 4 built-in networks and averages them if True, runs one network if False

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
            y, style = self._run_net(img, tile=tile, tile_overlap=tile_overlap,
                                     bsize=bsize, return_conv=return_conv)
        else:  
            for j in range(len(self.pretrained_model)):
                self.net.load_model(self.pretrained_model[j], cpu=(not self.gpu))
                y0, style = self._run_net(img, tile=tile, 
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


    def _run_net(self, imgs, tile=True, tile_overlap=0.1, bsize=224,
                 return_conv=False):
        """ run network on image or stack of images

        Parameters
        --------------

        imgs: array [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient(s) for image

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
        # make image nchan x Ly x Lx for net
        imgs = np.transpose(imgs, (2,0,1))
        detranspose = (1,2,0)
        
        print(datetime.now(), f"Pad img")
        # pad image for net so Ly and Lx are divisible by 4
        imgs, ysub, xsub = transforms.pad_image_ND(imgs)
        print(datetime.now(), f"Done")
        # slices from padding
#         slc = [slice(0, self.nclasses) for n in range(imgs.ndim)] # changed from imgs.shape[n]+1 for first slice size 
        slc = [slice(0, imgs.shape[n]+1) for n in range(imgs.ndim)]
        slc[-3] = slice(0, self.nclasses + 32*return_conv + 1)
        slc[-2] = slice(ysub[0], ysub[-1]+1)
        slc[-1] = slice(xsub[0], xsub[-1]+1)
        slc = tuple(slc)

        # run network
        if tile or imgs.ndim==4:
            y, style = self._run_tiled(imgs, bsize=bsize, 
                                      tile_overlap=tile_overlap, 
                                      return_conv=return_conv)
        else:
            imgs = np.expand_dims(imgs, axis=0)
            y, style = self.network(imgs, return_conv=return_conv)
            y, style = y[0], style[0]
        if style is not None:
            style /= (style**2).sum()**0.5

        # slice out padding
        y = y[slc]
        # transpose so channels axis is last again
        y = np.transpose(y, detranspose)
        
        return y, style
    
    def _run_tiled(self, imgi, bsize=224, tile_overlap=0.1, return_conv=False):
        """ run network in tiles of size [bsize x bsize]

        First image is split into overlapping tiles of size [bsize x bsize].
        The average of the network output over tiles is returned.

        Parameters
        --------------

        imgi: array [nchan x Ly x Lx] or [Lz x nchan x Ly x Lx]

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
        # IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=bsize, 
        #                                                 tile_overlap=tile_overlap)
        print(datetime.now(), f"Tile img")
        ysub, xsub, Ly, Lx, IMGshape = make_tiles(imgi, bsize=bsize, tile_overlap=tile_overlap)
        ysub, xsub = torch.LongTensor(ysub), torch.LongTensor(xsub)
        tile_mask = self.mask_tile(ysub, xsub, imgi.shape)
        batches = torch.where(tile_mask)[0].numpy()
        ny, nx, nchan, ly, lx = IMGshape
        lower_intensity = self.lower_intensity
        dset = TiledImage(imgi, ysub, xsub, lower_intensity)
        dset = Subset(dset, batches)
        # IMG = np.reshape(IMG, (ny*nx, nchan, ly, lx))
        batch_size = self.batch_size
        dloader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=16)
        # niter = int(np.ceil(batches.shape[0] / batch_size))
        nout = self.nclasses + 32*return_conv
        y = np.zeros((ny*nx, nout, ly, lx))
        # for k in range(niter):
        print(datetime.now(), f"Loop tiles")
        for k, data in enumerate(dloader):
            img = data['tiles']
            imask = data['is_foreground']
            img = img[imask]
            # irange = np.arange(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
            irange = batches[batch_size*k : min(batches.shape[0], batch_size*k+batch_size)]
            irange = irange[imask]
            # y0, style = self.network(IMG[irange], return_conv=return_conv)
            y0, style = self.network(img, return_conv=return_conv)
            y[irange] = y0.reshape(len(irange), y0.shape[-3], y0.shape[-2], y0.shape[-1])
        #     if k==0:
        #         styles = style[0]
        #     styles += style.sum(axis=0)
        # styles /= (ny*nx)
        
        print(datetime.now(), f"Avg tiles")
        # yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
        yf = average_tiles(torch.from_numpy(y), ysub, xsub, Ly, Lx).numpy()
        yf = yf[:,:imgi.shape[1],:imgi.shape[2]]
        print(datetime.now(), f"Done")
        # styles /= (styles**2).sum()**0.5
        return yf, None#, styles


class TiledImage(Dataset):
    def __init__(self, imgi, ysub, xsub, lower_intensity):
        self.imgi = imgi
        self.ysub = ysub
        self.xsub = xsub
        self.lower_intensity = lower_intensity

    def __getitem__(self, idx):
        ys, ye = self.ysub[idx]
        xs, xe = self.xsub[idx]
        out = self.imgi[:, ys:ye,  xs:xe].float()
        return {'tiles': out, 'is_foreground': out.mean() > self.lower_intensity}

    def __len__(self):
        return len(self.ysub)
    
    

def make_tiles(imgi, bsize, tile_overlap):
    nchan, Ly, Lx = imgi.shape
    tile_overlap = min(0.5, max(0.05, tile_overlap))
    bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
    bsizeY = np.int32(bsizeY)
    bsizeX = np.int32(bsizeX)
    # tiles overlap by 10% tile size
    ny = 1 if Ly<=bsize else int(np.ceil((1.+2*tile_overlap) * Ly / bsize))
    nx = 1 if Lx<=bsize else int(np.ceil((1.+2*tile_overlap) * Lx / bsize))
    ystart = np.linspace(0, Ly-bsizeY, ny).astype(int)
    xstart = np.linspace(0, Lx-bsizeX, nx).astype(int)

    ysub = []
    xsub = []
    # IMG = np.zeros((len(ystart), len(xstart), nchan,  bsizeY, bsizeX), np.float32)
    for j in range(len(ystart)):
        for i in range(len(xstart)):
            ysub.append([ystart[j], ystart[j]+bsizeY])
            xsub.append([xstart[i], xstart[i]+bsizeX])
            # IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1],  xsub[-1][0]:xsub[-1][1]]
    return ysub, xsub, Ly, Lx, (len(ystart), len(xstart), nchan,  bsizeY, bsizeX)
    
def average_tiles(y, ysub, xsub, Ly, Lx):
    ysub, xsub = ysub.to(y.device), xsub.to(y.device)
    Navg = torch.zeros((Ly,Lx)).to(y.device)
    yf = torch.zeros((y.shape[1], Ly, Lx)).float().to(y.device)
    # taper edges of tiles
    mask = _taper_mask(ly=y.shape[-2], lx=y.shape[-1]).to(y.device)
    for j in range(len(ysub)):
        yf[:, ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += mask
    yf /= Navg
    return yf

def _taper_mask(ly=224, lx=224, sig=7.5):
    bsize = max(224, max(ly, lx))
    xm = torch.arange(bsize).float()
    xm = torch.abs(xm - xm.mean())
    mask = 1/(1 + torch.exp((xm - (bsize/2-20)) / sig))
    mask = mask * mask.unsqueeze(1)
    mask = mask[bsize//2-ly//2 : bsize//2+ly//2+ly%2, 
                bsize//2-lx//2 : bsize//2+lx//2+lx%2]
    return mask


def normalize_img(img, axis=-1, invert=False, device='cuda'):
    """ normalize each channel of the image so that so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities

    optional inversion

    Parameters
    ------------

    img: ND-array (at least 3 dimensions)

    axis: channel axis to loop over for normalization

    invert: invert image (useful if cells are dark instead of bright)

    Returns
    ---------------

    img: ND-array, float32
        normalized image of same size

    """
    if img.ndim<3:
        error_message = 'Image needs to have at least 3 dimensions'
        # transforms_logger.critical(error_message)
        raise ValueError(error_message)

    img = img.float()
    img = torch.moveaxis(img, axis, 0)
    for k in range(img.shape[0]):
        # ptp can still give nan's with weird images
        i99 = percentile(img[k], 99, device)
        i1 = percentile(img[k], 1, device)
        if i99 - i1 > +1e-3: #np.ptp(img[k]) > 1e-3:
            img[k] = (img[k] - i1) / (i99 - i1)
            if invert:
                img[k] = -1*img[k] + 1   
        else:
            img[k] = 0
    img = torch.moveaxis(img, 0, axis)
    return img

def percentile(tensor, q, device):
    flattened = tensor.to(device).flatten()
    k = int((q / 100.0) * flattened.numel())
    sorted_tensor, _ = torch.sort(flattened)
    percentile_value = sorted_tensor[k-1]#.float()
    return percentile_value.cpu()

class NISModel(UnetModel):    
    def __init__(self, gpu=False, pretrained_model=False, 
                    net_avg=False,
                    diam_mean=30., device=None,
                    residual_on=True, style_on=True, concatenation=False,
                    nchan=2):
        self.torch = True
        if isinstance(pretrained_model, np.ndarray):
            pretrained_model = list(pretrained_model)
        elif isinstance(pretrained_model, str):
            pretrained_model = [pretrained_model]
    
        self.diam_mean = diam_mean
        if pretrained_model:
            pretrained_model_string = pretrained_model[0]
            params = parse_model_string(pretrained_model_string)
            if params is not None:
                _, residual_on, style_on, concatenation = params 
            # models_logger.info(f'>>>> loading model {pretrained_model_string}')
            

                
        # initialize network
        super().__init__(gpu=gpu, pretrained_model=False,
                         diam_mean=self.diam_mean, net_avg=net_avg, device=device,
                         residual_on=residual_on, style_on=style_on, concatenation=concatenation,
                        nchan=nchan)

        self.unet = False
        self.pretrained_model = pretrained_model
        if self.pretrained_model:
            self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
            self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]
            self.diam_labels = self.net.diam_labels.data.cpu().numpy()[0]

        
        ostr = ['off', 'on']
        self.net_type = 'cellpose_residual_{}_style_{}_concatenation_{}'.format(ostr[residual_on],
                                                                                   ostr[style_on],
                                                                                   ostr[concatenation]
                                                                                 ) 
    
    def get_prob(self, x, batch_size=8, channels=None, channel_axis=None, 
             z_axis=None, normalize=True, invert=False, lower_intensity=100,
             rescale=None, diameter=None, net_avg=False, tile=True, tile_overlap=0.1,
             resample=True, loop_run=False, model_loaded=False):

        
        if not model_loaded and (isinstance(self.pretrained_model, list) and not net_avg and not loop_run):
            self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
        # reshape image (normalization happens in _run_cp)
        x = transforms.convert_image(x, channels, channel_axis=channel_axis, z_axis=z_axis,
                                    normalize=False, invert=False, nchan=self.nchan)
        if x.ndim < 4:
            x = x[np.newaxis,...]
        self.batch_size = batch_size
        self.lower_intensity = lower_intensity

        if diameter is not None and diameter > 0:
            rescale = self.diam_mean / diameter
        elif rescale is None:
            diameter = self.diam_labels
            rescale = self.diam_mean / diameter

        p = self._run_cp(x, 
                        normalize=normalize,
                        invert=invert,
                        rescale=rescale, 
                        net_avg=net_avg, 
                        resample=resample,
                        tile=tile, 
                        tile_overlap=tile_overlap,
                        )
        
        # flows = [plot.dx_to_circ(dP), dP, cellprob, p] if masks is not None else None
        return p

    def _run_cp(self, x, normalize=True, invert=False,
                rescale=1.0, net_avg=False, resample=True, tile=True, tile_overlap=0.1,
                ):
        shape = x.shape if not isinstance(x[0], str) else []
        print(datetime.now(), f"x.shape={shape}")
        img = np.asarray(x[0])
        if normalize or invert:
            print(datetime.now(), f"Normalize img")
            img = normalize_img(torch.from_numpy(img), invert=invert, device=self.device).numpy()
            print(datetime.now(), f"Done")
            
        if rescale != 1.0:
            print(datetime.now(), f"Resize img")
            img = transforms.resize_image(img, rsz=rescale)
            print(datetime.now(), f"Done")

        yf, style = self._run_nets(img, net_avg=net_avg, tile=tile,
                                    tile_overlap=tile_overlap)
        if resample:
            print(datetime.now(), f"Resize output")
            yf = transforms.resize_image(yf, shape[1], shape[2])
            print(datetime.now(), f"Done")
        return yf

