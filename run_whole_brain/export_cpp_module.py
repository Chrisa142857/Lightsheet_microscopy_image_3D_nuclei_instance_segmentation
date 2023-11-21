import torch, torchist
# from run_all_in_one import preproc_flow2d, one_chunk_2d_to_3d

from unet import NISModel, percentile, make_tiles
from collections import namedtuple
import math 
from tqdm import tqdm, trange
from kornia.filters.median import median_blur

import numpy as np

device = 'cuda:0'


def init_flow3Dindex(p, iscell, zinds, yinds, xinds):
    # for i in range(dims):
    isnotcell = torch.logical_not(iscell)
    p[0, isnotcell] = zinds[isnotcell].float()
    p[1, isnotcell] = yinds[isnotcell].float()
    p[2, isnotcell] = xinds[isnotcell].float()

    pflows = torch.stack([p[i].flatten().long() for i in range(3)])
    # for i in range(dims):
    # pflows.append(p[i].flatten().long())
    # edges.append(torch.arange(-.5 - rpad, shape0[i] + .5 + rpad, 1))
    return pflows
def export_init_flow3Dindex():
    lz,ly,lx = 10,300,300
    p = torch.randn(3, lz, ly, lx)
    iscell = torch.randn(lz, ly, lx)
    zinds, yinds, xinds = torch.meshgrid(torch.arange(lz), torch.arange(ly), torch.arange(lx), indexing='ij')
    args = (p, iscell, zinds, yinds, xinds)
    traced_script_module = torch.jit.trace(init_flow3Dindex, args)
    traced_script_module.save(f'downloads/resource/init_flow3Dindex.pt')

def export_flow3D2seed():
    rpad = 20
    Lz, Ly, Lx = 14, 800, 800
    pflows = torch.randn(Lz * Ly * Lx, 3).double()
    edge0 = torch.arange(-.5 - rpad, Lz + .5 + rpad, 1)
    edge1 = torch.arange(-.5 - rpad, Ly + .5 + rpad, 1)
    edge2 = torch.arange(-.5 - rpad, Lx + .5 + rpad, 1)
    # args = (pflows, edge0, edge1, edge2)
    h = histogramdd(pflows, edges=[edge0, edge1, edge2])
    traced_script_module = torch.jit.trace(flow_3DtoSeed, h)
    traced_script_module.save(f'downloads/resource/flow_3DtoSeed.pt')
    # edge0 = torch.arange(-.5 - rpad, 10 + .5 + rpad, 1)
    # edge1 = torch.arange(-.5 - rpad, 80 + .5 + rpad, 1)
    # edge2 = torch.arange(-.5 - rpad, 80 + .5 + rpad, 1)
    # traced_script_module(pflows, edge0, edge1, edge2)


def histogramdd(
    x,
    weights = None,
    edges = None,
):

    # Preprocess
    D = x.size(-1)
    x = x.reshape(-1, D).squeeze(-1)
    edges = [e.flatten() for e in edges]
    bins = [e.numel() - 1 for e in edges]
    low = [e[0] for e in edges]
    upp = [e[-1] for e in edges]

    pack = x.new_full((D, max(bins) + 1), torch.inf)
    pack[0, :edges[0].numel()] = edges[0].to(x)
    pack[1, :edges[1].numel()] = edges[1].to(x)
    pack[2, :edges[2].numel()] = edges[2].to(x)

    edges = pack

    # assert torch.all(upp > low), "The upper bound must be strictly larger than the lower bound"

    bins = torch.as_tensor(bins).squeeze().long()
    low = torch.as_tensor(low).squeeze().to(x)
    upp = torch.as_tensor(upp).squeeze().to(x)

    print(edges.shape)
    # Filter out-of-bound values
    mask = ~out_of_bounds(x, low, upp)
    print(x.shape)
    x = x[mask]
    print(x.shape)

    print(edges.shape)
    # Indexing
    idx = torch.searchsorted(edges, x.t().contiguous(), right=True).t() - 1

    print(idx.shape)
    # Histogram
    shape = torch.Size(bins.expand(D))

    idx = ravel_multi_index(idx, shape)
    print(idx.shape)
    hist = idx.bincount(weights, minlength=shape.numel())
    print(shape, shape.numel(), hist.shape)
    hist = hist.reshape(shape)
    print(hist.shape)
    exit()
    return hist

def ravel_multi_index(coords, shape):
    shape = coords.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()
    return (coords * coefs).sum(dim=-1)

def out_of_bounds(x, low, upp):
    a, b = x < low, x > upp
    # if x.dim() > 1:
    a, b = torch.any(a, dim=-1), torch.any(b, dim=-1)
    return torch.logical_or(a, b)

# def flow_3DtoSeed(pflows, edge0, edge1, edge2):
    # iter_num = 3
    # dims = 3
    # edges = [edge0, edge1, edge2]
    # h = histogramdd(pflows, edges=edges)
    # shape = h.shape
def flow_3DtoSeed(h):
    iter_num = 3
    dims = 3
    hmax = h.clone().float()
    max_filter = torch.nn.MaxPool1d(kernel_size=iter_num, stride=1, padding=iter_num//2)
    for i in range(dims):
        hmax = max_filter(hmax.transpose(i, -1)).transpose(i, -1)
    seeds = torch.nonzero(torch.logical_and(h - hmax > -1e-6, h > 8), as_tuple=True)
    Nmax = h[seeds]
    isort = torch.argsort(Nmax, descending=True)
    seeds = tuple(seeds[i][isort] for i in range(dims))
    pix = torch.stack(seeds, dim=1)
    return pix


def export_index_flow3D():
    trained_model = 'downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_22_42_54.153497_epoch_21'
    state_dict = torch.load(trained_model, map_location=torch.device('cpu'))
    rescale = state_dict['diam_mean'] / state_dict['diam_labels']
    niter = np.uint32(1 / rescale * 200).item()
    print(niter)
    dP = torch.randn(3, 10, 3000, 3000) * (torch.randn(10, 3000, 3000)>0) / 5.
    print(dP.shape)
    shape = dP.shape[1:]
    Lz, Ly, Lx = torch.LongTensor([shape[0]])[0], torch.LongTensor([shape[1]])[0], torch.LongTensor([shape[2]])[0]
    # index_flow3D(dP, torch.arange(shape[0]), torch.arange(shape[1]), torch.arange(shape[2]), Lz, Ly, Lx)
    args = (dP, torch.arange(shape[0]), torch.arange(shape[1]), torch.arange(shape[2]), Lz, Ly, Lx)
    traced_script_module = torch.jit.trace(index_flow3D, args)
    traced_script_module.save(f'downloads/resource/index_flow3D.pt')
    
    # cp_mask = torch.randn(14, 8000, 8000) > 0
    # compute_masks(dP, cp_mask, niter)

def index_flow3D(dP, x, y, z, Lz, Ly, Lx):
    niter=139
    p = torch.meshgrid(x, y, z, indexing='ij')
    p = torch.stack(p).float()
    inds = torch.nonzero(dP[0].abs()>1e-3)#.long()
    ## 
    p = steps3D(p, dP, inds, Lz, Ly, Lx, niter)#.numpy()
    return p

def compute_masks(dP, cp_mask, niter):#, Lz, Ly, Lx
    Lz, Ly, Lx = dP.shape[1:]
    p = follow_flows(dP * cp_mask / 5., Lz, Ly, Lx, niter)
    out = get_masks(p, cp_mask, Lz, Ly, Lx)
    return out

def follow_flows(dP, Lz, Ly, Lx, niter):
    shape = [Lz, Ly, Lx]
    p = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]),
            torch.arange(shape[2]), indexing='ij')
    p = torch.stack(p).float()
    inds = torch.nonzero(dP[0].abs()>1e-3).long()
    ## 
    p = steps3D(p, dP, inds, Lz, Ly, Lx, niter)#.numpy()
    return p

def steps3D(p, dP, inds, Lz, Ly, Lx, niter):
    z = inds[:, 0]
    y = inds[:, 1]
    x = inds[:, 2]
    pmin = torch.zeros(3)
    pmax = torch.stack([Lz-1,Ly-1,Lx-1])
    pp = p[:, z, y, x]
    for _ in range(niter):
        pz = torch.floor(pp[0]).long()
        py = torch.floor(pp[1]).long()
        px = torch.floor(pp[2]).long()
        pdP = dP[:, pz, py, px]
        pp = (pp + pdP).transpose(0, -1)
        pp = pp.clip(min=pmin, max=pmax).transpose(0, -1)
    p[:, z, y, x] = pp
    return p


# def get_masks(p, iscell, Lz, Ly, Lx, rpad=20):
#     ## Optimized #####################################################
#     iter_num = 3 # needs to be odd
#     pflows = []
#     edges = []
#     # shape0 = p.shape[1:]
#     shape0 = [Lz, Ly, Lx]
#     # dims = len(p)
#     dims = 3
#     inds = torch.meshgrid(torch.arange(shape0[0]), torch.arange(shape0[1]),
#                             torch.arange(shape0[2]), indexing='ij')

#     for i in range(dims):
#         p[i, ~iscell] = inds[i][~iscell].float()

#     for i in range(dims):
#         pflows.append(p[i].flatten().long())
#         edges.append(torch.arange(-.5 - rpad, shape0[i] + .5 + rpad, 1))

#     h = torchist.histogramdd(torch.stack(pflows, dim=1).double(), edges=edges)
#     shape = h.shape
#     hmax = h.clone().float()
#     max_filter = torch.nn.MaxPool1d(kernel_size=iter_num, stride=1, padding=iter_num//2)
#     for i in range(dims):
#         hmax = max_filter(hmax.transpose(i, -1)).transpose(i, -1)
#     seeds = torch.nonzero(torch.logical_and(h - hmax > -1e-6, h > 8), as_tuple=True)
#     Nmax = h[seeds]
#     isort = torch.argsort(Nmax, descending=True)
#     seeds = tuple(seeds[i][isort] for i in range(dims))
#     pix = torch.stack(seeds, dim=1)
    #####################################
def expand_seed(seed_num, pflows, pix, h, Lz, Ly, Lx, rpads):
    shape0 = [Lz, Ly, Lx]
    shape = h.shape
    iter_num = 3
    dims = 3
    pix = list(pix)
    # if len(pix) == 0: return None
    expand = torch.nonzero(torch.ones((3,3,3))).T
    expand = tuple(e.unsqueeze(1) for e in expand)
    for iter in range(iter_num): # expand all seeds four times
        for k in range(seed_num): # expand each seed
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
    #####################################   
    M = torch.zeros(*shape, dtype=torch.long)
    remove_c = 0
    big = Lz * Ly * Lx * 1e-3
    for i in range(dims):
        pflows[i] = pflows[i] + rpads[i]
    fg = torch.zeros(*shape, dtype=bool)
    fg[tuple(pflows)] = True
    coords = []
    labels = []
    vols = []
    centers = []
    for k in range(len(pix)):
        if len(pix[k][0]) > big:
            remove_c += 1
            continue
        is_fg = fg[pix[k]]
        if not is_fg.any():
            remove_c += 1
            continue
        coord = (pix[k][0][is_fg], pix[k][1][is_fg], pix[k][2][is_fg])
        vols.append(len(coord[0]))
        centers.append(torch.stack([(c.max()+c.min())/2 for c in coord]))
        M[coord] = 1+k-remove_c
        coords.append(
            torch.cat([torch.stack(coord, -1), torch.zeros(len(coord[0]), 1, dtype=coord[0].dtype)+1+k-remove_c], -1)[is_fg, :]
        )
        labels.append(1+k-remove_c)
    M0 = M[tuple(pflows)]
    coords = torch.cat(coords)
    labels = torch.LongTensor(labels)
    centers = torch.stack(centers)
    vols = torch.LongTensor(vols)
    # ## Code optimization end
    M0 = torch.reshape(M0, shape0)
    return M0, coords, labels, vols, centers

# def grad_2d_to_3d(flow_2d: torch.Tensor, pre_final_yx_flow: torch.Tensor, pre_last_second: torch.Tensor):
#     # flow_2d, pre_final_yx_flow, pre_last_second = args
#     std_resolution = (2.5, .75, .75)
#     in_resolution = (4, .75, .75)
#     scale_r = [i/s for i, s in zip(in_resolution, std_resolution)]
#     flow_2d = preproc_flow2d(flow_2d, pre_final_yx_flow, scale_r)
#     grad3d = one_chunk_2d_to_3d(flow_2d, pre_last_second, 'cuda:0')
#     return grad3d
    # sim_grad_z(i, yx_flow, cellprob, pre_yx_flow, next_yx_flow, device=device)

def export_interpolate():
    flow_2d = torch.randn(3, 40, 8000, 9000).to('cuda:0')
    traced_script_module = torch.jit.trace(interpolate_wrap, flow_2d)
    traced_script_module.save(f'downloads/resource/interpolate_ratio_1.6x1x1.pt')
    

def interpolate_wrap(flow_2d):
    scale_r = (4/2.5, 1, 1)
    return torch.nn.functional.interpolate(flow_2d.unsqueeze(0), scale_factor=scale_r, mode='nearest-exact').squeeze()

def export_sim_gradz():
    device='cuda:0'
    yx_flow = torch.randn(2, 5000, 5000).to(device)
    cellprob = torch.randn(5000, 5000).to(device)
    pre_yx_flow = torch.randn(2, 5000, 5000).to(device)
    next_yx_flow = torch.randn(2, 5000, 5000).to(device)
    # sim_grad_z(yx_flow, cellprob, pre_yx_flow, next_yx_flow)
    args = (yx_flow, cellprob, pre_yx_flow, next_yx_flow)
    # grad_2d_to_3d(args)
    traced_script_module = torch.jit.trace(sim_grad_z, args)
    device = device.replace(':', '')
    traced_script_module.save(f'downloads/resource/grad_2Dto3D.pt')

def sim_grad_z(yx_flow, cellprob, pre_yx_flow, next_yx_flow):
    stagen = 7
    filter_size = 3
    filter = median_blur
    grad_Z = torch.zeros_like(cellprob)
    # print(datetime.now(), "Do median filter pyramid for flow map %d" % i)
    # pre_yx_flow = pre_yx_flow.to(device)
    # next_yx_flow = next_yx_flow.to(device)
    inside = (cellprob.unsqueeze(0).unsqueeze(0) > 0)#.to(device)
    pre = ((pre_yx_flow**2).sum(0)**0.5).unsqueeze(0).unsqueeze(0)
    next = ((next_yx_flow**2).sum(0)**0.5).unsqueeze(0).unsqueeze(0)
    grad_Z = next - pre
    for _ in range(stagen):
        pre_nextstage = filter(pre, filter_size)
        next_nextstage = filter(next, filter_size)
        grad_Z = grad_Z + (next - pre_nextstage)
        grad_Z = grad_Z + (next_nextstage - pre)
        pre = pre_nextstage
        next = next_nextstage
    grad_Z = grad_Z / (1 + stagen*2)
    grad_Z = grad_Z * inside
    grad_Z = grad_Z.squeeze()#.cpu()
    dP = torch.stack((grad_Z, yx_flow[0], yx_flow[1], cellprob),
                dim=0) # (4, dZ, dY, dX))
    return dP#.numpy()



# def export_grad_2Dto3D(device):
#     input_size = (10,512,512)
#     flow_2d = torch.stack([torch.randn(*input_size[1:],3) for _ in range(input_size[0])])
#     pre_final_yx_flow = torch.randn(3,*input_size[1:])
#     pre_last_second = torch.randn(2,*input_size[1:])
#     # device = 'cuda:1'
#     args = (flow_2d, pre_final_yx_flow, pre_last_second)
#     # grad_2d_to_3d(args)
#     traced_script_module = torch.jit.trace(grad_2d_to_3d, args)
#     device = device.replace(':', '')
#     traced_script_module.save(f'downloads/resource/grad_2Dto3D_{device}.pt')

def export_nis_model(device):
    trained_model = 'downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_22_42_54.153497_epoch_21'
    model = NISModel(device=torch.device('cpu'), pretrained_model=trained_model)
    x = torch.randn(10, 2, 224, 224)
    model.net.eval()
    traced_script_module = torch.jit.trace(model.net, x)
    # device = device.replace(':', '')
    traced_script_module.save(f'downloads/resource/nis_unet_cpu.pt')
    

def export_get_model_param():
    img = torch.randn(1, 8000, 9000)
    bsize = torch.FloatTensor([224])[0]
    overlap = torch.FloatTensor([0.1])[0]
    Ly = torch.LongTensor([img.shape[1]])[0]
    Lx = torch.LongTensor([img.shape[2]])[0]
    mask = torch.randn(250, 250)
    # z = torch.FloatTensor([100])[0]
    args = (Ly, Lx, bsize, overlap, mask)
    # y1, x1, _, _, _, _ = make_tiles_torch(*args)
    # y2, x2, _, _, _ = make_tiles(img, bsize, overlap)
    # y2, x2 = torch.LongTensor(y2), torch.LongTensor(x2)
    # print(y1.shape)
    # print(y2.shape)
    # print((y1==y2).all(), (x1==x2).all())
    traced_script_module = torch.jit.trace(make_tiles_torch, args)
    # device = device.replace(':', '')
    traced_script_module.save(f'downloads/resource/get_model_tileparam_cpu.pt')
    print(traced_script_module.code)


def make_tiles_torch(Ly: torch.Tensor, Lx: torch.Tensor, bsize: torch.Tensor, tile_overlap: torch.Tensor, mask: torch.Tensor):
    # bsize = bsize.long()
    tile_overlap = tile_overlap.clip(min=0.05, max=0.5)
    bsizeY = bsize
    bsizeX = bsize
    # tile_overlap = min(torch.FloatTensor([0.5])[0], max(torch.FloatTensor([0.05])[0], tile_overlap))
    # bsizeY, bsizeX = min(bsize, torch.LongTensor([Ly])[0]), min(bsize, torch.LongTensor([Lx])[0])
    # tiles overlap by 10% tile size
    ny = torch.ceil((1.+2*tile_overlap) * Ly / bsize).long()#.item()
    nx = torch.ceil((1.+2*tile_overlap) * Lx / bsize).long()#.item()
    # print(ny, nx)
    ystart = linspace(torch.FloatTensor([0])[0], Ly-bsizeY, ny).long()#[0]
    xstart = linspace(torch.FloatTensor([0])[0], Lx-bsizeX, nx).long()#[0]
    # print(ystart.shape, xstart.shape)
    ystarts = ystart[:, None].repeat(1,xstart.shape[0]).reshape(-1)
    xstarts = xstart.repeat(ystart.shape[0])
    yends = ystarts + bsizeY
    xends = xstarts + bsizeX
    ysub = torch.stack([ystarts, yends], -1)
    xsub = torch.stack([xstarts, xends], -1)
    ny, nx = ystart.shape[0], xstart.shape[0]
    
    # mask_zres=25
    # img_zres=2.5 # 4 ?
    yratio = mask.shape[0] / Ly
    xratio = mask.shape[1] / Lx
    # zratio = mask_zres / img_zres
    # z = (z / zratio).long()
    # mask = mask[z]
    cy = (ysub[:, 0] + ysub[:, 1])/2
    cx = (xsub[:, 0] + xsub[:, 1])/2
    cy = (cy * yratio).long()
    cx = (cx * xratio).long()
    Tile_param = namedtuple('Tile_param', ['ysub', 'xsub', 'mask'])
    out = Tile_param(ysub, xsub, torch.where(mask[cy, cx] > 0)[0])
    # out.ysub = ysub
    # out.xsub = xsub
    # out.mask = mask[cy, cx] > 0
    return out

# @torch.jit.script
def linspace(start: torch.Tensor, stop: torch.Tensor, num: torch.Tensor):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out

def export_preproc_img(device):
    img = torch.randn(1, 8113, 8452).float()
    trained_model = 'downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_22_42_54.153497_epoch_21'
    model = NISModel(device=torch.device('cpu'), pretrained_model=trained_model)
    rescale = model.diam_mean / model.diam_labels # 1.437293
    # print(rescale)
    # assert rescale == 1.437293
    area = torch.LongTensor([img.shape[1]*img.shape[2]])[0]
    args = (img.to('cuda:0'), area)
    # print(preproc(*args).shape)
    # exit()
    traced_script_module = torch.jit.trace(preproc, args)
    # device = device.replace(':', '')
    traced_script_module.save(f'downloads/resource/preproc_img1xLyxLx.pt')
    out = traced_script_module(torch.randn(1, 800, 800).float().to('cuda:0'), torch.LongTensor([800*800])[0])
    print(out.shape)

def preproc(img, area):
    img = torch.cat([img, torch.zeros_like(img)], dim=0) # 2 x Ly x Lx
    # print(datetime.now(), f"Normalize img")
    # print(img.shape)
    # area = Lx * Ly
    img = normalize_img(img, area) # 2 x Ly x Lx
    # print(img.shape, rescale)
    # print(datetime.now(), f"Resize img")
    img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=1.437293, mode='bilinear')[0]
    return img
    # print(img.shape)
    # Pad
    # img, pysub, pxsub = pad_image_ND(img)
    # print(img.shape)
    # Preproc_img = namedtuple('Preproc_img', ['pad_ysub', 'pad_xsub', 'img'])
    # out = Preproc_img(pysub, pxsub, img)
    # return out

def pad_image_ND(img0, div=16, extra = 1):
    Ly, Lx = img0.shape[1:]
    Lpad = (div * math.ceil(Ly/div) - Ly)#.long()
    xpad1 = extra*div//2 + Lpad//2
    xpad2 = extra*div//2 + Lpad - Lpad//2
    Lpad = (div * math.ceil(Lx/div) - Lx)#.long()
    ypad1 = extra*div//2 + Lpad//2
    ypad2 = extra*div//2+Lpad - Lpad//2
    pads = (ypad1, ypad2, xpad1, xpad2, 0, 0) # last, 2nd to last, 3rd to last
    I = torch.nn.functional.pad(img0, pads, mode='constant')
    Ly, Lx = img0.shape[-2:]
    ysub = torch.arange(xpad1, xpad1+Ly)
    xsub = torch.arange(ypad1, ypad1+Lx)
    return I, ysub, xsub

def normalize_img(img, area, chan=0):
    img = img.float()
    k = chan
    # Lx, Ly = img[k].shape
    i99 = percentile(img[k], 99, area)
    i1 = percentile(img[k], 1, area)
    img[k] = (img[k] - i1).clip(min=1e-3) / (i99 - i1).clip(min=1e-3)
    return img

def percentile(tensor, q, area):
    flattened = tensor.flatten()#.to(device)
    k = torch.ceil((q / 100.0) * area).long()
    sorted_tensor, _ = torch.sort(flattened)
    percentile_value = sorted_tensor[k-1]#.float()
    return percentile_value#.cpu()

def average_tiles(y, ysub, xsub, Ly, Lx):
    Navg = torch.zeros((Ly,Lx), device=y.device, requires_grad=False)
    yf = torch.zeros((y.shape[1], Ly, Lx), device=y.device, requires_grad=False).float()
    # taper edges of tiles
    mask = _taper_mask(bsize=y.shape[-2], device=y.device)
    avg_mask = torch.zeros((Ly,Lx), device=y.device, requires_grad=False).bool()
    for j in range(ysub.shape[0]):
        yf[:, ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] = yf[:, ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] + y[j] * mask
        Navg[ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] = Navg[ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] + mask
        avg_mask[ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] = True
    
    yf[:, avg_mask] = yf[:, avg_mask] / Navg[avg_mask]
    return yf
    
def _taper_mask(device, bsize=224, sig=7.5):
    xm = torch.arange(bsize, device=device).float()
    xm = torch.abs(xm - xm.mean())
    mask = 1/(1 + torch.exp((xm - (bsize/2-20)) / sig))
    mask = mask * mask.unsqueeze(1)
    return mask

# def postproc(y, ysub, xsub, preproc_img, pad_ysub, pad_xsub, org_img):
    # ysub, xsub = ysub.long(), xsub.long()
    # yf = average_tiles(y, ysub, xsub, preproc_img.shape[1], preproc_img.shape[2])
    # yf = yf[:, :preproc_img.shape[1], :preproc_img.shape[2]]
    # yf = yf[:, pad_ysub[0]:pad_ysub[-1]+1, pad_xsub[0]:pad_xsub[-1]+1]
def postproc(yf):
    yf = torch.nn.functional.interpolate(yf.unsqueeze(0), scale_factor=1/1.437293, mode='bilinear')[0]
    yf = torch.permute(yf, (1,2,0))#.detach().cpu()
    return yf

def export_postproc_img():
    yf = torch.randn(3, 500, 500)
    ly = torch.LongTensor([300])[0]
    lx = torch.LongTensor([300])[0]
    args = (yf, ly, lx)
    traced_script_module = torch.jit.trace(postproc, yf)
    # device = device.replace(':', '')
    # traced_script_module.save(f'downloads/resource/postproc_Unet{batchlen}x3x224x224.pt')
    traced_script_module.save(f'downloads/resource/postproc_unet.pt')

    # device = 'cuda:0'
    # trained_model = 'downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_22_42_54.153497_epoch_21'
    # model = NISModel(device=torch.device('cpu'), pretrained_model=trained_model)
    # x = torch.randn(10, 2, 224, 224)
    # model.net.eval()
    # model.net.to(device)
    # org_img = torch.randn(1, 8113, 8452).float()
    # print('Original image size', org_img.shape)
    # area = torch.LongTensor([org_img.shape[1]*org_img.shape[2]])[0]
    # img = preproc(org_img.to(device), area)
    # img,pad_ysub,pad_xsub = pad_image_ND(img)
    # print('Image size after preproc', img.shape)
    # bsize = torch.FloatTensor([224])[0]
    # overlap = torch.FloatTensor([0.1])[0]
    # Ly = torch.LongTensor([img.shape[1]])[0]
    # Lx = torch.LongTensor([img.shape[2]])[0]
    # mask = torch.randn(250, 250)
    # tile_param = make_tiles_torch(Ly, Lx, bsize, overlap, mask)
    # idx = torch.where(tile_param.mask)[0]
    # ysub = tile_param.ysub
    # xsub = tile_param.xsub
    # x = []
    # batchlen = 200
    # for i in tqdm(idx[:batchlen]):
    #     ys, ye = ysub[i].long()
    #     xs, xe = xsub[i].long()
    #     # print(ysub[i], xsub[i])
    #     x.append(img[:, ys:ye, xs:xe])
    # x = torch.stack(x)
    # print('Input size to Unet', x.shape)
    # with torch.no_grad():
    #     y, _ = model.net(x)
    # print('Output size from Unet', y.shape, y.mean(), y.max())
    # args = (y, ysub[:batchlen].to(device), xsub[:batchlen].to(device), img, pad_ysub, pad_xsub, org_img)
    # yf = postproc(*args)
    # print(yf.mean(), yf.max())
    # # exit()
    # traced_script_module = torch.jit.trace(postproc, args)
    # # device = device.replace(':', '')
    # traced_script_module.save(f'downloads/resource/postproc_Unet{batchlen}x3x224x224.pt')
    print('Output size after postproc', traced_script_module(yf).shape)

def export_fg_indexer():
    lz = 10
    ly = 300
    lx = 300
    plen = int(1e+9)
    zid = torch.randint(lz-1, (plen,))
    yid = torch.randint(ly-1, (plen,))
    xid = torch.randint(lx-1, (plen,))
    args = (torch.LongTensor([lz])[0],torch.LongTensor([ly])[0],torch.LongTensor([lx])[0],zid,yid,xid)
    # traced_script_module = torch.jit.trace(fg_indexer, args)
    traced_script_module = torch.jit.script(fg_indexer)
    traced_script_module.save(f'downloads/resource/fg_indexer.pt')

def fg_indexer(fg,zid,yid,xid):
    # fg = torch.zeros((lz.item(),ly.item(),lx.item())).bool()
    # fg[zid, yid, xid] = True
    fg = torch.index_put_(fg, (zid[:1000000], yid[:1000000], xid[:1000000]), torch.ones(zid[:1000000].shape[0], dtype=torch.bool), accumulate=True)
    return fg


if __name__ == '__main__':
    # export_init_flow3Dindex()
    # export_index_flow3D()
    # export_flow3D2seed()
    export_fg_indexer()
    # export_interpolate()
    # export_sim_gradz()
    # export_grad_2Dto3D(device)
    # export_nis_model(device)
    # export_get_model_param()
    # export_preproc_img(device)
    # export_postproc_img()
    