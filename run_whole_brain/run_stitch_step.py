import h5py, torch, os, json, signal
from tqdm import trange, tqdm
import numpy as np
from torch_geometric.data import Data
import fastremap
from scipy.ndimage import find_objects
import datetime, sys

import utils
from two_slice_stitch import StitchModel

from torch.multiprocessing import Pool
torch.multiprocessing.set_start_method('spawn', force=True)

brain_tag = sys.argv[1]
# brain_tag = 'L73D766P9'
pair_tag = 'pair15'
r = '/lichtman/ziquanw/Lightsheet/results/P4/%s/%s' % (pair_tag, brain_tag)
img_r = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/stitched' % (pair_tag, brain_tag)
brain_result_path = '%s/%s_NIS_results.h5' % (r, brain_tag)
brain_flow_dir = '%s/flow_3d' % r
remap_save_path = '%s/%s_remap.json' % (r, brain_tag)
device = 'cuda:1' # 'cuda:1'

def main():
    print(datetime.datetime.now(), f"Start python program {sys.argv}", flush=True)
    max_nuclei_size = (10, 30, 30) # 
    graph_model = StitchModel(device)
    graph_model.to(device)
    graph_model.load_state_dict(torch.load('downloads/resource/tss_weight.pth', map_location=device))
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
    total = len(dims[(next_id!=None) & (dims==0)])
    stitching_i = 0
    if os.path.exists(remap_save_path):
        with open(remap_save_path, 'r') as jsonf:
            out_json = json.load(jsonf)
        complete_stitch_i = [o['complete_stitch_i'] for o in out_json]
        print(datetime.datetime.now(), f"Already done stitching slices {[[stitch_pts[i][0], next_id[i]] for i in complete_stitch_i]}, skip them")
    else:
        out_json = []
        complete_stitch_i = []
    for i in range(N):
        if next_id[i] is None: continue
        if dims[i] != 0: continue
        stitching_i += 1
        if i in complete_stitch_i: continue
        loader_pool = Pool(processes=2)
        graph_node_pool = Pool(processes=2)
        graph_edge_pool = Pool(processes=2)
        pt = stitch_pts[i]
        dim = dims[i]
        pre_slice_id = pt[dim]
        next_slice_id = [next_id[i]]
        indices = [pre_slice_id] + next_slice_id
        print(datetime.datetime.now(), "[%03d/%d] Start stitch slices %s" % (stitching_i, total, indices), flush=True)
        zs, ze, ys, ye, xs, xe = pt # ~s and ~e are the same in the dim of stitching plane, e.g., zs = ze = 64
        print(datetime.datetime.now(), "Load mask, image, and nuclei probability flow", flush=True)
        assert dim == 0, "Not implemented"
        slice = seg[indices, ys:ye, xs:xe]
        pre_img_fn = flow_fnlist[pre_slice_id].split('_resample')[0]+'.tif'
        next_img_fn = flow_fnlist[next_slice_id[0]].split('_resample')[0]+'.tif'
        image = list(loader_pool.imap(utils.imread, [os.path.join(img_r, pre_img_fn), os.path.join(img_r, next_img_fn)]))
        get_rid_of_zoombie_process(loader_pool)
        image = np.stack([np.asarray(img) for img in image])
        loader_pool = Pool(processes=2)
        flow = load_flow_multiprocess(brain_flow_dir, pre_slice_id, next_slice_id[0], ys, ye, xs, xe, loader_pool)
        get_rid_of_zoombie_process(loader_pool)
        print(datetime.datetime.now(), "Done loading", flush=True)
        dim_vol= dim // 2
        pre_mask, next_stack = np.take(slice, 0, axis=dim_vol), np.take(slice, [ii for ii in range(1, len(indices))], axis=dim_vol)
        pre_slice_image, next_slice_image = np.take(image, 0, axis=dim_vol), np.take(image, 1, axis=dim_vol)
        pre_slice_image, next_slice_image = norm_img(pre_slice_image), norm_img(next_slice_image)
        pre_flow, next_flow = np.take(flow, 0, axis=dim_vol+1), np.take(flow, 1, axis=dim_vol+1)
        remap_dict = stitch_one_gap(graph_model, pre_mask, next_stack, pre_slice_image, next_slice_image, pre_flow, next_flow, dim_vol, graph_node_pool, graph_edge_pool)
        try:
            get_rid_of_zoombie_process(graph_edge_pool)
            get_rid_of_zoombie_process(graph_node_pool)
        except:
            pass
        if remap_dict is None: continue
        out_json.append({
            'stitching_slices': [pre_slice_id, next_slice_id[0]],
            'stitching_dim': dim_vol,
            'plane_range': [zs, ze, ys, ye, xs, xe],
            'remap': remap_dict,
            'complete_stitch_i': i
        })
        with open(remap_save_path,'w') as file:
            file.write(json.dumps(out_json, indent=4, cls=NpEncoder))
        torch.cuda.empty_cache()


def stitch_one_gap(model, pre_mask, next_stack, pre_slice, next_slice, pre_flow, next_flow, axis, graph_node_pool, graph_edge_pool):
    next_mask = np.take(next_stack, 0, axis=axis)
    print(datetime.datetime.now(), 'Build graph of nuclei between two slices', flush=True)
    input, orig_remaps, topn_index = build_graph(pre_slice, next_slice, pre_mask, next_mask, pre_flow, next_flow, model.preprocess, graph_node_pool, graph_edge_pool)
    if input is None: 
        print(datetime.datetime.now(), 'Skip, no nuclei in one slice', flush=True)
        return None
    print(datetime.datetime.now(), 'Run stitching model', flush=True)
    with torch.no_grad():
        try:
            edge_pred = model(input)
            pred = edge_pred.argmax(1).detach().cpu()
        except:
            print(datetime.datetime.now(), 'Graph model out of CUDA mem, using CPU', flush=True)
            model = model.cpu()
            edge_pred = model(input.cpu())
            model = model.to(device)
            pred = edge_pred.argmax(1)
    print(datetime.datetime.now(), 'Decode the output as a ID remap dict', flush=True)
    edge_index = input.edge_index.detach().cpu()
    is_new_cell = pred == 0
    topn_index = topn_index[~is_new_cell]
    pred = pred[~is_new_cell] - 1
    edge_pred = edge_pred[~is_new_cell]
    node0 = topn_index[:, 0]
    node1 = topn_index[:, 1:]
    node1 = list(node1)
    for i in range(len(node1)):
        node1[i] = node1[i][pred[i]]
    node1 = torch.stack(node1)
    # node1 = torch.index_select(node1[~is_new_cell], 1, pred[~is_new_cell]-1).diag()#.detach().cpu() # N1
    # node0 = node0[~is_new_cell]
    ## get new id of next stack
    oldid2newid = {}
    for nid in node1.unique():
        nid_place = node1==nid
        scores = edge_pred[nid_place].max(1)[0]
        oldid2newid[nid.item()] = (node0[nid_place][scores.argmax()]).item()
    ## fastremap made the id in mask starts from 1, while node id starts from 0 in the graph
    oldid2newid = {k+1:v+1 for k,v in oldid2newid.items()}
    ## current id in mask to original id before fastremap
    orig_remaps = [{v:k for k,v in map.items()} for map in orig_remaps]
    new_remap = {}
    ## remap next stack id to pre slice id 
    for k, v in oldid2newid.items():
        new_remap[orig_remaps[1][k]] = orig_remaps[0][v]

    print(datetime.datetime.now(), 'Complete, there are %d nuclei IDs in the chunk needs to be changed' % len(new_remap), flush=True)
    return new_remap


def get_rid_of_zoombie_process(pool):
    processes = pool._pool[:]
    for _curr_process in processes:
        pid = _curr_process.pid
        _curr_process.terminate()
        pool._pool.remove(_curr_process)
        # print(datetime.datetime.now(), f'Kill child process {pid}', flush=True)


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


def build_graph(img1, img2, mask1, mask2, flow1, flow2, builder, graph_node_pool, graph_edge_pool):
    mask1, remap1 = fastremap.renumber(mask1)
    mask2, remap2 = fastremap.renumber(mask2)
    bbox1 = get_bbox(mask1)
    bbox2 = get_bbox(mask2)
    if len(bbox1.shape) != 2 or len(bbox2.shape) != 2:
        return None, None
    x, edge_index, edge_attr, topn_index = builder([(img1, mask1.astype(np.int32), bbox1, flow1), (img2, mask2.astype(np.int32), bbox2, flow2)], 
                                       graph_node_pool, graph_edge_pool)
    data = Data(
        x=x, 
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    return data, [remap1, remap2], topn_index


def get_i_xy(fn):
    return int(fn.split('_')[-1][:-4].replace('resample', ''))


def sort_fs(fs, get_i):
    out = [0 for _ in range(len(fs))]
    for fn in fs:
        i = get_i(fn)
        out[i] = fn
    return out


def load_flow_multiprocess(fdir, zmin, zmax, ymin, ymax, xmin, xmax, pool):
    flist = sort_fs([fn for fn in os.listdir(fdir) if fn.endswith('.npy')], get_i_xy)
    fs = [fn for fn in flist if zmin <= get_i_xy(fn) < zmax+1]
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