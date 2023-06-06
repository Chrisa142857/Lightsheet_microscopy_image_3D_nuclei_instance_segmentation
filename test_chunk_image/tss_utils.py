from torch_geometric.data import Data
import torch
import fastremap
import numpy as np
import tifffile as tif
from scipy.ndimage import find_objects, median_filter


def build_graph_from_ram(imgs, masks, builder, device='cpu', istrain=False):
    inputs = []
    imgs = [img.astype(np.float32) for img in imgs]
    imgs = [(img-img.min())/(img.max()-img.min()) for img in imgs]
    data_list = []
    remaps = []
    for i in range(len(masks)):
        img, mask = imgs[i], masks[i]
        mask, remap = fastremap.renumber(mask)
        masks[i] = mask
        assert mask.max() == (len(np.unique(mask))-1), "%d, %d" % (mask.max(), len(np.unique(mask))-1)
        bbox, slices = get_bbox(mask)
        data_list.append([img, mask, bbox, slices, remap, None])
    assert len(data_list) % 2 == 0
    if istrain:
        gt_list = gen_gt(data_list)
    for i in range(len(data_list)):
        remaps.append(data_list[i][4])
        data_list[i][4] = None
    for i in range(len(data_list)):
        if i == 0:
            cur_data = data_list[i]
            next_data = data_list[i+1]
            pre = np.zeros(1)
            ZYXflow, YXflow, next_YXflow, pre_YXflow = simulate_ZYXflow(cur_data[1:-1], [pre, None, None, None], next_data[1:-1], device)
            pre_data = cur_data
            pre_data[-2:] = [YXflow, ZYXflow]
            cur_data = next_data
            cur_data[-2] = next_YXflow
            continue
        elif i == len(data_list)-1:
            next = np.zeros(1)
            ZYXflow, YXflow, next_YXflow, pre_YXflow = simulate_ZYXflow(cur_data[1:-1], pre_data[1:-1], [next, None, None, None], device)
            cur_data[-1] = ZYXflow
        else:
            next_data = data_list[i+1]
            ZYXflow, YXflow, next_YXflow, pre_YXflow = simulate_ZYXflow(cur_data[1:-1], pre_data[1:-1], next_data[1:-1], device)
            cur_data[-1] = ZYXflow
            next_data[-2] = next_YXflow
            
        assert cur_data[-1] is not None and pre_data[-1] is not None
        input = [(pre_data[0], pre_data[1], pre_data[2], pre_data[-1]), (cur_data[0], cur_data[1], cur_data[2], cur_data[-1])]
        x, edge, edge_attr = builder(input)
        if istrain:
            gt_edge = gt_list[i-1]
            node0 = edge[0].unique()
            # gt_edge, edge_index = gt_edge.T.tolist(), edge.T.tolist()
            gt_edge = gt_edge.T.tolist()
            labels = []
            for nid in node0:
                labels.append([1 if one_edge in gt_edge else 0 for one_edge in edge[:, edge[0]==nid].T.tolist()])
            labels = torch.LongTensor(labels).to(device)
            labels = torch.cat([torch.zeros(len(labels), dtype=torch.long).to(device).unsqueeze(-1), labels], -1)
            labels[labels.sum(1)==0, 0] = 1
            assert not (labels.sum(1) != 1).any()
            labels = torch.argmax(labels, dim=1)
            # label = torch.FloatTensor([1 if ind in gt_edge else 0 for ind in edge_index]).to(device)
            one_data = Data(
                x=x, 
                edge_index=edge,
                edge_attr=edge_attr,
                y=labels
            )
        else:
            one_data = Data(
                x=x, 
                edge_index=edge,
                edge_attr=edge_attr
            )
        inputs.append(one_data)
        pre_data = cur_data
        cur_data = next_data
        
    return inputs, masks, remaps
        

def build_graph_from_disk(imgs, masks, builder, device='cpu'):
    inputs = []
    assert len(masks) == len(imgs)
    for i in range(len(imgs)):
        if i == 0:
            cur_data = load_data_from_disk(imgs[i], masks[i]) + [None, None]
            next_data = load_data_from_disk(imgs[i+1], masks[i+1]) + [None, None]
            pre = np.zeros(1)
            ZYXflow, YXflow, next_YXflow, pre_YXflow = simulate_ZYXflow(cur_data[1:-1], [pre, None, None, None], next_data[1:-1], device)
            pre_data = cur_data
            pre_data[-2:] = [YXflow, ZYXflow]
            cur_data = next_data
            cur_data[-2] = next_YXflow
            continue
        elif i == len(imgs)-1:
            next = np.zeros(1)
            ZYXflow, YXflow, next_YXflow, pre_YXflow = simulate_ZYXflow(cur_data[1:-1], pre_data[1:-1], [next, None, None, None], device)
            cur_data[-1] = ZYXflow
        else:
            next_data = load_data_from_disk(imgs[i+1], masks[i+1]) + [None, None]
            ZYXflow, YXflow, next_YXflow, pre_YXflow = simulate_ZYXflow(cur_data[1:-1], pre_data[1:-1], next_data[1:-1], device)
            cur_data[-1] = ZYXflow
            next_data[-2] = next_YXflow
            
        assert cur_data[-1] is not None and pre_data[-1] is not None
        input = [(pre_data[0], pre_data[1], pre_data[2], pre_data[-1]), (cur_data[0], cur_data[1], cur_data[2], cur_data[-1])]
        x, edge, edge_attr = builder(input)
        one_data = Data(
            x=x, 
            edge_index=edge,
            edge_attr=edge_attr
        )
        inputs.append(one_data)
        pre_data = cur_data
        cur_data = next_data
    return inputs


def load_data_from_disk(ifn, mfn):
    mask = tif.imread(mfn).astype(np.int32)
    img = tif.imread(ifn).astype(np.float32)
    img = (img-img.min())/(img.max()-img.min())
    mask, _ = fastremap.renumber(mask)
    assert mask.max() == (len(np.unique(mask))-1), "%d, %d" % (mask.max(), len(np.unique(mask))-1)
    bbox, slices = get_bbox(mask)
    return [img, mask, bbox, slices]

def simulate_ZYXflow(input, pre, next, device):
    mask, bbox, slices, xyflow = input
    pre_mask, pre_bbox, pre_slices, pre_xyflow = pre
    next_mask, next_bbox, next_slices, next_xyflow = next
    inside = mask > 0
    filter_size = 3
    stagen = 7
    filter = median_filter
    if xyflow is None:
        xyflow = masks_to_flows_gpu(mask, bbox, slices, device) if mask.max() > 0 else None
    
    if next_xyflow is None:
        next_xyflow = masks_to_flows_gpu(next_mask, next_bbox, next_slices, device) if next_mask.max() > 0 else None
    
    if pre_xyflow is None:
        pre_xyflow = masks_to_flows_gpu(pre_mask, pre_bbox, pre_slices, device) if pre_mask.max() > 0 else None

    if pre_mask.max() == 0:
        grad_Z = (next_xyflow**2).sum(axis=0)**0.5
    elif next_mask.max() == 0:
        grad_Z = -1 * (pre_xyflow**2).sum(axis=0)**0.5
    elif mask.max() == 0:
        grad_Z = np.zeros_like(mask)
    else:
        grad_Z = np.zeros(mask.shape)
        pre = (pre_xyflow**2).sum(axis=0)**0.5
        next = (next_xyflow**2).sum(axis=0)**0.5
        grad_Z += (next - pre)
        for _ in range(stagen):
            pre_nextstage = filter(pre, size=filter_size)
            next_nextstage = filter(next, size=filter_size)
            grad_Z += (next - pre_nextstage)
            grad_Z += (next_nextstage - pre)
            pre = pre_nextstage
            next = next_nextstage
        grad_Z /= (1 + stagen*2)
    grad_Z *= inside
    xyzflow = np.concatenate((grad_Z[np.newaxis], xyflow),
                axis=0) # (dZ, dY, dX)
    return xyzflow, xyflow, next_xyflow, pre_xyflow

            
def gen_gt(data_list):
    gt_list = []
    for di in range(len(data_list)-1):
        map1, map2 = data_list[di][4], data_list[di+1][4]
        N1 = data_list[di][1].max()
        gt_edge_index = [[], []]
        for key in map1.keys():
            if key == 0: continue
            if key in map2: # add two edges
                gt_edge_index[0].extend([map1[key]-1])#, map2[key]+N1])
                gt_edge_index[1].extend([map2[key]+N1-1])#, map1[key]])
        gt_list.append(torch.LongTensor(gt_edge_index))
    return gt_list


def masks_to_flows_gpu(masks, bbox, slices, device):
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
    padded_centers = bbox[:, :2] + 1
    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:,:,0], neighbors[:,:,1]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = np.array([[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices])
    n_iter = 2 * (ext.sum(axis=1)).max()
    # run diffusion
    mu = _extend_centers_gpu(neighbors, padded_centers, isneighbor, Ly, Lx, 
                             n_iter=n_iter, device=device)
    # normalize
    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)
    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y-1, x-1] = mu
    return mu0


def get_bbox(masks):
    '''
        input array H x W 

        return [x, y, w, h]
    '''
    # get mask centers
    slices = find_objects(masks)
    bboxes = np.array([[(sr.stop + sr.start)/2, (sc.stop + sc.start)/2, sr.stop - sr.start, sc.stop - sc.start] for sr, sc in slices])
    return bboxes, slices


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

