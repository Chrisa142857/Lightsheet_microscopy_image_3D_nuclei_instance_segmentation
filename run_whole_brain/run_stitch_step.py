import h5py, torch, os, json
from tqdm import trange, tqdm
import numpy as np
from torch_geometric.data import Data
import fastremap
from scipy.ndimage import find_objects
from multiprocessing import Pool
import datetime, sys

import utils
from two_slice_stitch import StitchModel

# brain_tag = sys.argv[1]
brain_tag = 'L73D766P9'
pair_tag = 'pair15'
r = '/lichtman/ziquanw/Lightsheet/results/P4/%s/%s' % (pair_tag, brain_tag)
img_r = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/stitched' % (pair_tag, brain_tag)
brain_result_path = '%s/%s_NIS_results.h5' % (r, brain_tag)
# brain_image_path = 'wholebrain_results/P4_weights/%s_images.h5' % brain_tag
# brain_flow_path = 'wholebrain_results/P4_weights/%s_flow.h5' % brain_tag
brain_flow_dir = '%s/flow_3d' % r
# out_result_path = 'wholebrain_results/P4_weights/%s_stitched.h5' % brain_tag
remap_save_path = '%s/%s_remap.json' % (r, brain_tag)

def main():    
    max_nuclei_size = (10, 30, 30) # 
    device = 1
    graph_model = StitchModel('cuda:%d'%device)
    graph_model.to('cuda:%d'%device)
    graph_model.load_state_dict(torch.load('downloads/resource/tss_weight.pth'))
    graph_model.eval()
    flow_fnlist = sort_fs([fn for fn in os.listdir(brain_flow_dir) if fn.endswith('.npy')], get_i_xy)
    maskf = h5py.File(brain_result_path, 'r')
    seg = maskf['nuclei_segmentation']
    stitch_pts = maskf['wait_for_stitch']
    stitch_pts = stitch_pts[:]
    stitch_pts = stitch_pts[stitch_pts.sum(1)>0]
    next_id, dims = match_stitch_pt(stitch_pts) # only stitch pre slice to the next
    N = stitch_pts.shape[0]
    next_id = [None] + [stitch_pts[i][0]+1 for i in range(1, N-1)] + [None]
    # total = len(next_id[next_id!=None])
    total = len(dims[(next_id!=None) & (dims==0)])
    stitching_i = 0
    out_json = []
    for i in range(N):
        if next_id[i] is None: continue
        if dims[i] != 0: continue
        stitching_i += 1
        pt = stitch_pts[i]
        dim = dims[i]
        pre_slice_id = pt[dim]
        next_slice_id = next_id[i]
        next_slice_id = [next_slice_id.item()+j for j in range(max_nuclei_size[dim//2])]
        indices = [pre_slice_id] + next_slice_id
        print(datetime.datetime.now(), "[%03d/%d] Start stitch slices %s" % (stitching_i, total, indices))
        zs, ze, ys, ye, xs, xe = pt # ~s and ~e are the same in the dim of stitching plane, e.g., zs = ze = 64
        print(datetime.datetime.now(), "Load mask, image, and nuclei probability flow")
        assert dim == 0, "Not implemented"
        slice = seg[indices, ys:ye, xs:xe]
        pre_img_fn = flow_fnlist[pre_slice_id].split('_resample')[0]+'.tif'
        next_img_fn = flow_fnlist[next_slice_id[0]].split('_resample')[0]+'.tif'
        image = np.stack([np.asarray(utils.imread(os.path.join(img_r, pre_img_fn))), 
                          np.asarray(utils.imread(os.path.join(img_r, next_img_fn)))])
        flow = load_flow(brain_flow_dir, pre_slice_id, next_slice_id[0]+1, ys, ye, xs, xe)
        dim_vol= dim // 2
        pre_mask, next_stack = np.take(slice, 0, axis=dim_vol), np.take(slice, [ii for ii in range(1, len(indices))], axis=dim_vol)
        pre_slice_image, next_slice_image = np.take(image, 0, axis=dim_vol), np.take(image, 1, axis=dim_vol)
        pre_slice_image, next_slice_image = norm_img(pre_slice_image), norm_img(next_slice_image)
        pre_flow, next_flow = np.take(flow, 0, axis=dim_vol+1), np.take(flow, 1, axis=dim_vol+1)
        remap_dict = stitch_one_gap(graph_model, pre_mask, next_stack, pre_slice_image, next_slice_image, pre_flow, next_flow, dim_vol)
        if remap_dict is None: continue
        out_json.append({
            'stitching_slices': [pre_slice_id, next_slice_id[0]],
            'stitching_dim': dim_vol,
            'plane_range': [zs, ze, ys, ye, xs, xe],
            'remap': remap_dict
        })
        with open(remap_save_path,'w') as file:
            file.write(json.dumps(out_json, indent=4, cls=NpEncoder))


def stitch_one_gap(model, pre_mask, next_stack, pre_slice, next_slice, pre_flow, next_flow, axis):
    next_mask = np.take(next_stack, 0, axis=axis)
    print(datetime.datetime.now(), 'Build graph of nuclei between two slices')
    input, orig_remaps = build_graph(pre_slice, next_slice, pre_mask, next_mask, pre_flow, next_flow, model.preprocess)
    if input is None: 
        print(datetime.datetime.now(), 'Skip, no nuclei in one slice')
        return None
    print(datetime.datetime.now(), 'Run stitching model')
    edge_pred = model(input)
    print(datetime.datetime.now(), 'Decode the output as a ID remap dict')
    pred = edge_pred.argmax(1).detach().cpu()
    is_new_cell = pred == 0
    edge_index = input.edge_index.detach().cpu()
    node0 = edge_index[0].unique() # N0
    edge_index[1] = edge_index[1] - edge_index[0].max() - 1 # recover mask1 node id
    node1 = torch.stack([edge_index[1, edge_index[0]==nid] for nid in node0]) # N1
    node1 = torch.index_select(node1[~is_new_cell], 1, pred[~is_new_cell]-1).diag()#.detach().cpu() # N1
    node0 = node0[~is_new_cell]
    ## get new id of next stack
    oldid2newid = {}
    for nid in node1.unique():
        nid_place = node1==nid
        scores = edge_pred[~is_new_cell][nid_place].max(1)[0]
        oldid2newid[nid.item()] = (node0[nid_place][scores.argmax()]).item()
    ## fastremap made the id in mask starts from 1, while node id starts from 0 in the graph
    oldid2newid = {k+1:v+1 for k,v in oldid2newid.items()}
    ## current id in mask to original id before fastremap
    orig_remaps = [{v:k for k,v in map.items()} for map in orig_remaps]
    new_remap = {}
    ## remap next stack id to pre slice id 
    for k, v in oldid2newid.items():
        new_remap[orig_remaps[1][k]] = orig_remaps[0][v]

    print(datetime.datetime.now(), 'Complete, there are %d nuclei IDs in the chunk needs to be changed' % len(new_remap))
    return new_remap


def norm_img(img):
    return (img-img.min())/(img.max()-img.min()) 


def match_stitch_pt(pts):
    other_dims = {
        0: [2, 4],
        2: [0, 4],
        4: [0, 2]
    }
    N = pts.shape[0]
    dims = pts[:, ::2] == pts[:, 1::2]
    dims = np.stack(np.where(dims))
    assert dims.shape[1] == N
    next_slice_id = []
    for i in range(N):
        dim = dims[1, i] * 2
        slicen = pts[i, dim]
        f1 = pts[:, dim] == (slicen+1)
        f2 = pts[:, dim+1] == (slicen+1)
        f_next = f1 & f2
        # f1 = pts[:, dim] == (slicen-1)
        # f2 = pts[:, dim+1] == (slicen-1)
        # f_pre = f1 & f2
        for od in other_dims[dim]:
            ff = pts[i, od] == pts[:, od]
            # f_pre = f_pre & ff
            f_next = f_next & ff
        # f = f_next if f_next.any() else f_pre
        f = f_next
        if not f.any(): 
            next_slice_id.append(None)
        else:
            next_slice_id.append(np.where(f)[0])
    return np.array(next_slice_id, dtype=object), dims[1, :] * 2


def get_bbox(masks):
    '''
        input array H x W 

        return [x, y, w, h]
    '''
    # get mask centers
    slices = find_objects(masks)
    bboxes = np.array([[(sr.stop + sr.start)/2, (sc.stop + sc.start)/2, sr.stop - sr.start, sc.stop - sc.start] for sr, sc in slices])
    return bboxes


def build_graph(img1, img2, mask1, mask2, flow1, flow2, builder):
    mask1, remap1 = fastremap.renumber(mask1)
    mask2, remap2 = fastremap.renumber(mask2)
    bbox1 = get_bbox(mask1)
    bbox2 = get_bbox(mask2)
    if len(bbox1.shape) != 2 or len(bbox2.shape) != 2:
        return None, None
    x, edge_index, edge_attr = builder([(img1, mask1.astype(np.int32), bbox1, flow1), (img2, mask2.astype(np.int32), bbox2, flow2)])
    data = Data(
        x=x, 
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    return data, [remap1, remap2]


def get_i_xy(fn):
    return int(fn.split('_')[-1][:-4].replace('resample', ''))


def sort_fs(fs, get_i):
    out = [0 for _ in range(len(fs))]
    for fn in fs:
        i = get_i(fn)
        out[i] = fn
    return out


def load_flow(fdir, zmin, zmax, ymin, ymax, xmin, xmax):
    flist = sort_fs([fn for fn in os.listdir(fdir) if fn.endswith('.npy')], get_i_xy)
    fs = [fn for fn in flist if zmin <= get_i_xy(fn) < zmax]
    with Pool(processes=4) as pool:
        # data = list(tqdm(pool.imap(load_data, [(os.path.join(fdir, fn), ymin, ymax, xmin, xmax) for fn in fs]), total=len(fs), desc="Load flow")) # [3 x Y x X]
        data = list(pool.imap(load_data, [(os.path.join(fdir, fn), ymin, ymax, xmin, xmax) for fn in fs])) # [3 x Y x X]
    return np.stack(data, axis=1)[:3]


def load_data(args):
    path, ymin, ymax, xmin, xmax = args
    return np.load(path)[:, ymin:ymax, xmin:xmax]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__": main()