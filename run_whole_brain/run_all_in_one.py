# import tifffile as tif
import os, torch, h5py, json
from tqdm import tqdm, trange
import sys#, time
import numpy as np
# from torch.utils.data import DataLoader,Dataset,Subset
from unet import NISModel, make_tiles, average_tiles, normalize_img
import utils
# from multiprocessing import Pool
from datetime import datetime
# from torch.multiprocessing import Process
import multiprocessing as mp
from kornia.filters.median import median_blur

from run_cpu_step import compute_masks
from run_stitch_step import stitch_one_gap, NpEncoder
from two_slice_stitch import StitchModel
import transforms

# mp.set_start_method('spawn')
# num_cpus = mp.cpu_count()-1
# print(datetime.now(), "Pipeline start with %d CPUs" % (num_cpus+1))
async def main():
    global TILER_PARAM
    TILER_PARAM = None
    print(datetime.now(), f"Start python program {sys.argv}", flush=True)
    # 2D to 3D: one chunk has [slicen_3d] slices, based on the RAM limit. Different number has no effect to results
    slicen_3d = 30
    ##
    batch_size = 400
    ##
    brain_tag = 'L73D766P4' # L73D766P9
    pair_tag = 'pair15'
    # brain_tag = 'L94P4' # L73D766P9
    # pair_tag = 'female'
    device = torch.device('cuda:2')
    # brain_tag = sys.argv[1]
    # pair_tag = sys.argv[2] # 'pair15'
    # device = torch.device('cuda:%d' % int(sys.argv[3]))
    # save_r = '/lichtman/ziquanw/Lightsheet/results/P4'
    save_r = '/cajal/ACMUSERS/ziquanw/Lightsheet/results/temp'
    # save_r = '/cajal/ACMUSERS/ziquanw/Lightsheet/results/P14'
    global std_resolution, in_resolution
    std_resolution = (2.5, .75, .75)
    in_resolution = (4, .75, .75)
    scale_r = [i/s for i, s in zip(in_resolution, std_resolution)]
    save_r = '%s/%s' % (save_r, pair_tag)
    os.makedirs(save_r, exist_ok=True)
    save_r = '%s/%s' % (save_r, brain_tag)
    os.makedirs(save_r, exist_ok=True)
    eval_tag = '_C1_'
    img_r = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/stitched' % (pair_tag, brain_tag)
    # img_r = '/lichtman/Ian/Lightsheet/P14/stitched/%s/%s/stitched' % (pair_tag, brain_tag)
    maskn = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/registered/%s_MASK_topro_25_all.nii' % (pair_tag, brain_tag, brain_tag)
    # maskn = None
    
    trained_model = 'downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_22_42_54.153497_epoch_21'
    model = NISModel(device=torch.device('cpu'), pretrained_model=trained_model)
    lower_intensity = 0.001 if maskn is None else 0 # 0~1
    model.lower_intensity = lower_intensity
    mask_tile = utils.MaskTiledImg(maskn=maskn, img_zres=std_resolution[0])
    model.mask_tile = mask_tile
    state_dict = torch.load(trained_model, map_location=torch.device('cpu'))
    rescale = state_dict['diam_mean'] / state_dict['diam_labels']
    niter = np.uint32(1 / rescale * 200).item()
    # print(niter.item())
    # exit()
    fs = [os.path.join(img_r, f) for f in os.listdir(img_r) if eval_tag in f and f.endswith('.tif')]
    orig_fs = fs.copy()
    depth_start = 1
    for f in orig_fs:
        fs[filename_to_depth(f, depth_start)] = f
    start = 0
    end = -1
    fs = fs[start:end] if end > 0 else fs[start:]
    # dset = BrainSliceDataset(fs)
    depth = len(fs) 
    x = np.asarray(utils.imread(fs[0]))[np.newaxis,...]
    shapex = [x.shape[0], int(x.shape[1]*rescale), int(x.shape[2]*rescale)]
    padsize = get_pad_size(shapex)
    shapex[-2] += padsize[-2]
    shapex[-1] += padsize[-1]
    # tiler_param = make_tiles(shapex, bsize=224, tile_overlap=0.1)
    whole_brain_shape = (depth, x.shape[1], x.shape[2])
    save_ftail = '_NIS_results.h5'
    save_path = os.path.join(save_r, brain_tag+save_ftail)
    datasets = init_h5database(save_path, whole_brain_shape)
    # flow_2d = []
    graph_model = StitchModel(device)
    graph_model.to(device)
    graph_model.load_state_dict(torch.load('downloads/resource/tss_weight.pth', map_location=device))
    graph_model.eval()

    pre_final_yx_flow, pre_last_second = None, None
    # resampled_si = 0
    # si = 0
    idbase = 0
    # print(datetime.now(), "Load slice %d" % (start + 1))
    z = 0
    
    ####################################################################################
    chunk_fs = [fs[i:i+slicen_3d] for i in range(0, len(fs), slicen_3d)]
    starts = [i for i in range(0, len(fs), slicen_3d)]
    ends = [i+slicen_3d for i in range(0, len(fs), slicen_3d)]
    gpu_args = [chunk_fs[0], starts[0], ends[0], pre_final_yx_flow, pre_last_second, maskn, batch_size, depth_start, scale_r, device, model]
    gpuout = gpu_process(*gpu_args)
    grad3d, pre_final_yx_flow, pre_last_second, first_img, last_img = gpuout
    assert not grad3d.isnan().any(), grad3d.max()
    # grad3d = torch.randn(4, 79, 8113, 8452)
    masks = None
    out_json = []
    # for fs, start, end in zip(chunk_fs[1:], starts[1:], ends[1:]):
    for ci in range(len(chunk_fs)):
        if ci + 1 < len(chunk_fs):
            fs, start, end = chunk_fs[ci+1], starts[ci+1], ends[ci+1]
            gpu_args[0] = fs
            gpu_args[1] = start
            gpu_args[2] = end
            gpu_args[3] = pre_final_yx_flow
            gpu_args[4] = pre_last_second
        cpu_args = (grad3d[:3], grad3d[3], niter)
        zmin, zmax = z, z + grad3d.shape[1]
        z = zmax
        print(datetime.now(), f'Compute 3D NIS from one chunk & GPU-compute one chunk in parallel')
        # nis_out = await compute_masks(*cpu_args)
        # nis_out = asyncio.run(compute_masks(*cpu_args))
        asyncio.new_event_loop()
        loop = asyncio.get_event_loop()
        # nis_out = compute_masks(cpu_args)
        # gpuout = gpu_process(gpu_args)
        cpu_task = loop.create_task(async_wrap(cpu_args, compute_masks))
        print(datetime.now(), "CPU task created")
        if ci + 1 < len(chunk_fs):
            gpu_task = loop.create_task(async_wrap(gpu_args, gpu_process))
            print(datetime.now(), "GPU task created")
        # loop.run_until_complete([cpu_task, gpu_task])
        # gpuout = gpu_process(*gpu_args)
        # with mp.Pool(processes=2) as pool:
        #     gpuout, nis_out = pool.starmap(mp_wrap, [(gpu_args, gpu_process), (cpu_args, compute_masks)])
        #     pool.close()
        #     pool.join()
        await cpu_task
        nis_out = cpu_task.result()
        if ci + 1 < len(chunk_fs):
            await gpu_task
            gpuout = gpu_task.result()
        if nis_out is not None:
            print(datetime.now(), f'Done, {nis_out[2].shape} masks from grad, zmin={zmin} zmax={zmax}')
            if zmin > 0:
                pre_last_mask = masks[-1]
            idbase, masks = save_to_h5(nis_out, datasets, idbase, zmin, zmax, whole_brain_shape)
            if zmin > 0:
                remap_dict = stitch_process(graph_model, pre_last_mask, masks, pre_last_img, first_img, pre_last_flow3d, grad3d[:, -1])
                if remap_dict is not None: out_json = save_stitch(remap_dict, out_json, os.path.join(save_r, brain_tag+"_remap.json"))
        else:
            print(datetime.now(), 'Done, No mask')
        pre_last_img = last_img
        pre_last_flow3d = grad3d[:, -1]
        if ci + 1 < len(chunk_fs):
            grad3d, pre_final_yx_flow, pre_last_second, first_img, last_img = gpuout

    print(datetime.now(), "All Done")
    # exit()

def stitch_process(graph_model, pre_slice_mask, masks, pre_slice_image, next_slice_image, pre_slice_flow3d, next_slice_flow3d):
    pre_slice_image = norm_img(pre_slice_image)
    next_slice_image = norm_img(next_slice_image)
    pre_slice_flow3d = pre_slice_flow3d.detach().numpy()
    next_slice_flow3d = next_slice_flow3d.detach().numpy()
    return stitch_one_gap(graph_model, pre_slice_mask, masks, pre_slice_image, next_slice_image, pre_slice_flow3d, next_slice_flow3d, 0)

def save_stitch(remap_dict, out_json, remap_save_path):
    out_json.append({
        'remap': remap_dict,
    })
    with open(remap_save_path,'w') as file:
        file.write(json.dumps(out_json, indent=4, cls=NpEncoder))
    return out_json

def mp_wrap(args, processer):
    return processer(*args)

async def async_wrap(args, processer):
    return processer(*args)

def gpu_process(fs, start, end, pre_final_yx_flow, pre_last_second, maskn, batch_size, depth_start, scale_r, device, model):
    print(datetime.now(), 'Loading model for chunk compute by GPU')
    # trained_model = 'downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_29_22_42_54.153497_epoch_21'
    # model = NISModel(device=device, pretrained_model=trained_model)
    # lower_intensity = 0.001 if mask_tile.mask is None else 0 # 0~1
    # model.lower_intensity = lower_intensity
    # model.mask_tile = mask_tile
    TILER_PARAM = None
    model.net.to(device)
    model.device = device
    # rescale = model.diam_mean / model.diam_labels
    first_img = None
    last_img = None
    flow_2d = []
    imgi = 0
    for fn in tqdm(fs, desc="Run 2D Unet from slice %d to %d" % (start, end)):
        # img, fn = data
        if maskn is not None: 
            model.mask_tile.z = filename_to_depth(fn, depth_start)
            if int(model.mask_tile.z/model.mask_tile.zratio) >= len(model.mask_tile.mask): 
                print(datetime.now(), "Brain mask end, break out")
                break
        print(datetime.now(), f"Input {fn.split('/')[-1]} to model")
        prob, TILER_PARAM, org_img = forward_nismodel(model=model, imgpath=fn, batch_size=batch_size, TILER_PARAM=TILER_PARAM)
        flow_2d.append(prob)
        if imgi == 0: first_img = org_img
        if imgi == len(fs) - 1: last_img = org_img
        imgi += 1

    flow_2d = preproc_flow2d(torch.stack(flow_2d, dim=0), pre_final_yx_flow, scale_r)
    grad3d = one_chunk_2d_to_3d(flow_2d, pre_last_second, device)
    pre_final_yx_flow = flow_2d[:, -1]
    pre_last_second = flow_2d[:2, -2]
    model.net.cpu()
    return (grad3d, pre_final_yx_flow, pre_last_second, first_img, last_img)

def init_h5database(save_path, whole_brain_shape):
    f = h5py.File(save_path, 'w')
    seg_dset = f.create_dataset('nuclei_segmentation', whole_brain_shape, dtype='int', chunks=True, maxshape=(None, whole_brain_shape[1], whole_brain_shape[2]))
    bmask_dset = f.create_dataset('binary_mask', whole_brain_shape, dtype='bool', chunks=True, maxshape=(None, whole_brain_shape[1], whole_brain_shape[2]))
    coord_dset = f.create_dataset('coordinate', (0, 4), dtype='int', chunks=True, maxshape=(None, 4)) # z, y, x, label
    ilabel_dset = f.create_dataset('instance_label', (0,), dtype='int', chunks=True, maxshape=(None,))
    ivolume_dset = f.create_dataset('instance_volume', (0,), dtype='int', chunks=True, maxshape=(None,))
    icenter_dset = f.create_dataset('instance_center', (0, 3), dtype='float', chunks=True, maxshape=(None, 3)) # z, y, x
    stitch_range = f.create_dataset('wait_for_stitch', (0, 6), dtype='int', maxshape=(None, 6))
    return seg_dset, bmask_dset, coord_dset, ilabel_dset, ivolume_dset, icenter_dset, stitch_range

def save_to_h5(output, datasets, idbase, zmin, zmax, whole_brain_shape):
    masks, coords, labels, vols, centers = output
    seg_dset, bmask_dset, coord_dset, ilabel_dset, ivolume_dset, icenter_dset, stitch_range = datasets
    if zmax >= whole_brain_shape[0]:
        seg_dset.resize((zmax, whole_brain_shape[1], whole_brain_shape[2]))
        bmask_dset.resize((zmax, whole_brain_shape[1], whole_brain_shape[2]))
    bmask = masks > 0
    masks[bmask] = masks[bmask] + idbase
    labels = labels + idbase
    coords[:, 0] = coords[:, 0] + zmin
    centers[:, 0] = centers[:, 0] + zmin
    coords[:, 3] = coords[:, 3] + idbase
    masks, coords, labels, vols, centers, bmask = masks.numpy(), coords.numpy(), labels.numpy(), vols.numpy(), centers.numpy(), bmask.numpy()
    seg_dset[zmin:zmax, :, :] = masks
    bmask_dset[zmin:zmax, :, :] = bmask
    old_contour_n = coord_dset.shape[0]
    new_contour_n = old_contour_n + coords.shape[0]
    h5dataset_append(coord_dset, coords, old_contour_n, new_contour_n)
    old_instance_n = ilabel_dset.shape[0]
    new_instance_n = old_instance_n + labels.shape[0]
    h5dataset_append(ilabel_dset, labels, old_instance_n , new_instance_n)
    h5dataset_append(ivolume_dset, vols, old_instance_n , new_instance_n)
    h5dataset_append(icenter_dset, centers, old_instance_n , new_instance_n)
    idbase = idbase + max(labels) + 1
    srange = np.zeros((1, 6))
    srange[0, 0] = zmin
    srange[0, 1] = zmin
    for d in range(1, 3):     
        srange[0, (d*2)] = 0
        srange[0, (d*2)+1] = whole_brain_shape[d]
    h5dataset_append(stitch_range, srange)
    return idbase, masks

def h5dataset_append(dataset, new_data, old_n=None, new_n=None):
    if old_n is None: old_n = dataset.shape[0]
    if new_n is None: new_n= old_n + new_data.shape[0]
    new_shape = (new_n, dataset.shape[1]) if len(dataset.shape) == 2 else (new_n,)
    dataset.resize(new_shape)
    dataset[old_n:] = new_data

def get_pad_size(shape, div=16, extra = 1):
    Lpad = int(div * np.ceil(shape[-2]/div) - shape[-2])
    xpad1 = extra*div//2 + Lpad//2
    xpad2 = extra*div//2 + Lpad - Lpad//2
    Lpad = int(div * np.ceil(shape[-1]/div) - shape[-1])
    ypad1 = extra*div//2 + Lpad//2
    ypad2 = extra*div//2+Lpad - Lpad//2
    return ypad1+ypad2, xpad1+xpad2

def norm_img(img):
    return (img-img.min())/(img.max()-img.min()) 

def preproc_flow2d(flow_2d: torch.Tensor, pre_final_yx_flow, scale_r):
    flow_2d = flow_2d.permute((3, 0, 1, 2))
    print(datetime.now(), f"Resample to standard resolution from shape {flow_2d.shape}")
    flow_2d = torch.nn.functional.interpolate(flow_2d.unsqueeze(0), scale_factor=scale_r, mode='nearest-exact').squeeze()#.numpy()
    if pre_final_yx_flow is not None: 
        # pre_final_yx_flow = pre_final_yx_flow.permute((2, 0, 1))
        # pre_last_second = pre_last_second.permute((2, 0, 1))
        flow_2d = torch.cat([pre_final_yx_flow.unsqueeze(1), flow_2d], dim=1)
    print(datetime.now(), f"Resampled to shape {flow_2d.shape}")
    return flow_2d

def one_chunk_2d_to_3d(flow_2d, pre_last_second, device):
    '''
        flow_2d: Size(3, z, y, x)
        return: Size(4, z, y, x)
    '''
    # print(datetime.now(), "Start 2D to 3D at resampled slice %d" % resampled_si)
    grad3d = []
    for i in trange(flow_2d.shape[1]-1, desc="Do median filter pyramid for flow map"):
        yx_flow = flow_2d[:2, i]
        cellprob = flow_2d[2, i]
        next_yx_flow = flow_2d[:2, i+1]
        if i > 0:
            pre_yx_flow = flow_2d[:2, i-1]
        else:
            pre_yx_flow = pre_last_second
        ##########################################################################
        dP = sim_grad_z(i, yx_flow, cellprob, pre_yx_flow, next_yx_flow, device=device)
        ##########################################################################
        grad3d.append(dP)
        # resampled_si += 1
    grad3d = torch.stack(grad3d, dim=1)
    print(datetime.now(), "Done 2D to 3D, Grad3D.shape=", grad3d.shape)
    return grad3d

def sim_grad_z(i, yx_flow, cellprob, pre_yx_flow, next_yx_flow, device='cpu'):
    stagen = 7
    filter_size = 3
    filter = median_blur
    grad_Z = torch.zeros_like(cellprob)
    if pre_yx_flow is None: pass
    elif next_yx_flow is None: pass
    else: 
        # print(datetime.now(), "Do median filter pyramid for flow map %d" % i)
        pre_yx_flow = pre_yx_flow.to(device)
        next_yx_flow = next_yx_flow.to(device)
        inside = (cellprob.unsqueeze(0).unsqueeze(0) > 0).to(device)
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
        grad_Z = grad_Z.squeeze().cpu()
    dP = torch.stack((grad_Z, yx_flow[0], yx_flow[1], cellprob),
                dim=0) # (4, dZ, dY, dX))
    return dP#.numpy()


def filename_to_depth(f, depth_start):
    return int(f.split('/')[-1].split('_')[1]) - depth_start

class TiledImageFromPath:#(Dataset):
    def __init__(self, imgpath, lower_intensity, model: NISModel, return_conv, TILER_PARAM=None):
        
        print(datetime.now(), f"Loading img from disk")
        self.org_img = utils.imread(imgpath)
        x = np.asarray(self.org_img)[np.newaxis,...]
        # print(datetime.now(), f"Done")
                
        print(datetime.now(), f"Get IMG {x.shape}, Tranpose img axis")
        x = transforms.convert_image(x, [0,0],
                                    normalize=False, invert=False, nchan=model.nchan)
        if x.ndim < 4:
            x = x[np.newaxis,...]
        # print(datetime.now(), f"Done")
        # self.batch_size = batch_size
        # self.lower_intensity = lower_intensity

        diameter = model.diam_labels
        rescale = model.diam_mean / diameter
        
        shape = x.shape
        print(datetime.now(), f"Now x.shape={shape}")
        # img = np.asarray(x[0])
        img = x[0]
        print(datetime.now(), f"Normalize img")
        img = normalize_img(torch.from_numpy(img), device=model.device).numpy()
        # print(datetime.now(), f"Done")
            
        if rescale != 1.0:
            print(datetime.now(), f"Resize img")
            img = transforms.resize_image(img, rsz=rescale)
            # print(datetime.now(), f"Done")

        img = np.transpose(img, (2,0,1))
        detranspose = (1,2,0)
        
        print(datetime.now(), f"Pad img")
        # pad image for net so Ly and Lx are divisible by 4
        img, pysub, pxsub = transforms.pad_image_ND(img)
        # print(datetime.now(), f"Done")
        if TILER_PARAM is None: 
            TILER_PARAM = make_tiles(img.shape, bsize=224, tile_overlap=0.1)    
        ysub, xsub, Ly, Lx, IMGshape = TILER_PARAM
        ysub, xsub = torch.LongTensor(ysub), torch.LongTensor(xsub)
        self.TILER_PARAM = TILER_PARAM
        # slices from padding
#         slc = [slice(0, self.nclasses) for n in range(imgs.ndim)] # changed from imgs.shape[n]+1 for first slice size 
        slc = [slice(0, img.shape[n]+1) for n in range(img.ndim)]
        slc[-3] = slice(0, model.nclasses + 32*return_conv + 1)
        print(pysub[0], pysub[-1]+1)
        print(pxsub[0], pxsub[-1]+1)
        exit()
        slc[-2] = slice(pysub[0], pysub[-1]+1)
        slc[-1] = slice(pxsub[0], pxsub[-1]+1)
        slc = tuple(slc)

        self.slc = slc
        self.detranspose = detranspose
        self.IMGshape = IMGshape
        self.ysub = ysub
        self.xsub = xsub
        self.lower_intensity = lower_intensity
        self.imgpath = imgpath
        self.imgi = torch.from_numpy(img)
        self.org_shape = shape

        # self.is_foreground = None
        tile_mask = model.mask_tile(ysub, xsub, self.imgi.shape)
        batches = torch.where(tile_mask)[0].numpy()
        # dset = Subset(tileset, batches)
        # self.is_fore = [None for _ in range(len(self.ysub))]
        if model.mask_tile.mask is not None: 
            self.is_foreground = batches
            # self.ysub = self.ysub[batches]
            # self.xsub = self.xsub[batches]
        else:
            self.is_foreground = torch.zeros(len(self.ysub)).bool()
            for i in range(len(self.ysub)):
                ys, ye = self.ysub[i]
                xs, xe = self.xsub[i]
                self.is_foreground[i] = self.imgi[:, ys:ye,  xs:xe].mean() > self.lower_intensity

            self.is_foreground = torch.where(self.is_foreground)[0]
            # self.ysub = self.ysub[self.is_foreground]
            # self.xsub = self.xsub[self.is_foreground]

    def __getitem__(self, i):
        idx = self.is_foreground[i]
        ys, ye = self.ysub[idx]
        xs, xe = self.xsub[idx]
        out = self.imgi[:, ys:ye,  xs:xe]
        # if self.is_foreground is None: 
        #     is_fore = out.mean() > self.lower_intensity
        #     self.is_fore = is_fore
            # return {'tiles': out, 'is_foreground': is_fore}
        return out
        # else:
        #     # return {'tiles': out, 'is_foreground': True}
        #     return out

    def __len__(self):
        return len(self.is_foreground)
    

def forward_nismodel(model: NISModel, imgpath, batch_size, num_workers=16, return_conv=False, TILER_PARAM=None):
    print(datetime.now(), f"Build img tiler")
    # _, _, Ly, Lx, IMGshape = tiler_param
    lower_intensity = model.lower_intensity
    tileset = TiledImageFromPath(imgpath, lower_intensity, model, return_conv=False, TILER_PARAM=TILER_PARAM)
    ny, nx, nchan, ly, lx = tileset.IMGshape
    ysub, xsub = tileset.ysub, tileset.xsub
    dloader = [torch.stack([tileset[j] for j in range(i, min(len(tileset), i+batch_size))]) for i in range(0, len(tileset), batch_size)]
    # dloader = DataLoader(tileset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    nout = model.nclasses + 32*return_conv
    print(datetime.now(), f"Done, {len(tileset)} tiles into {len(dloader)} batches")

    y = torch.zeros((ny*nx, nout, ly, lx)).to(model.device)
    print(datetime.now(), f"Loop tiles")
    model.net.eval()
    for k, data in enumerate(dloader):
        # img = data['tiles'].float()
        # imask = data['is_foreground']
        # img = img[imask]
        img = data.to(model.device)
        # if tileset.is_foreground is None:
        #     irange = torch.arange(batch_size*k, min(ny*nx, batch_size*k+batch_size))
        # else:
        irange = tileset.is_foreground[batch_size*k : min(tileset.is_foreground.shape[0], batch_size*k+batch_size)]

        y0, _ = model.network(img, return_conv=return_conv)
        # y0 = y0.detach().cpu()
        assert not y0.isnan().any(), y0.max()
        y[irange] = y0.reshape(len(irange), y0.shape[-3], y0.shape[-2], y0.shape[-1])

    print(datetime.now(), f"Avg tiles")
    yf = average_tiles(y, ysub, xsub, tileset.imgi.shape[1], tileset.imgi.shape[2])#.numpy()
    yf = yf[:,:tileset.imgi.shape[1],:tileset.imgi.shape[2]]
    # print(datetime.now(), f"Done")
    # slice out padding
    yf = yf[tileset.slc]
    # transpose so channels axis is last again
    
    print(datetime.now(), f"Resize output from size {yf.shape} to size {(yf.shape[0], tileset.org_shape[1], tileset.org_shape[2])}")
    # yf = transforms.resize_image(yf, tileset.org_shape[1], tileset.org_shape[2])

    yf = torch.nn.functional.interpolate(yf.unsqueeze(0), size=(tileset.org_shape[1], tileset.org_shape[2]), mode='bilinear')[0]
    yf = torch.permute(yf, tileset.detranspose).detach().cpu()
    print(datetime.now(), f"Done")
    return yf, tileset.TILER_PARAM, tileset.org_img


if __name__ == "__main__":
    import asyncio
    # main()
    asyncio.run(main())
