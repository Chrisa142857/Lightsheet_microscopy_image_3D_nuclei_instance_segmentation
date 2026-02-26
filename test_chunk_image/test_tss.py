
from tss_utils import build_graph_from_ram
import tifffile as tif
import numpy as np
from tqdm import trange
import imageio, torch
# import fastremap
from two_slice_stitch import StitchModel
# from utils import iou_2mask, mask2mask_list, eval_f1

def main():    # tss_train_output (epoch115.pth)
    output_dir = 'tss_train_output_filter_pyramid'
    best_epoch = 90
    device = 'cuda:1'
    with open('train_list.txt', 'r') as f:        
        path_list = f.read().split('\n')[:-1]
    maskpath_list = path_list
    imgpath_list = [p.replace('/masks', '/images').replace('_masks', '') for p in path_list]
    model = StitchModel(device)    
    model.to(device)
    model.load_state_dict(torch.load('%s/epoch%d.pth'%(output_dir, best_epoch)))
    for i in trange(len(maskpath_list), desc="Initializing data"):
        masks = tif.imread(maskpath_list[i]).astype(np.int32)
        imgs = imageio.v2.mimread(imgpath_list[i])
        pred_mask = test_onestack(masks, imgs, model, device)

def test_onestack(masks, imgs, model, device):
    model.eval()
    thr = 0.5
    # orig_masks = masks.copy()
    inputs, masks, orig_remaps = build_graph_from_ram(imgs, masks, model.preprocess, device=device, istrain=False)
    remaps = [{} for _ in range(len(masks))]
    for si, input in enumerate(inputs):
        edge_pred = model(input)
        pred = edge_pred.argmax(1)#.detach().cpu()
        is_new_cell = pred == 0
        edge_index = input.edge_index
        node0 = edge_index[0].unique()
        edge_index[1] = edge_index[1] - edge_index[0].max() - 1 # recover mask1 node id
        node1 = torch.stack([edge_index[1, edge_index[0]==nid] for nid in node0])
        node1 = torch.index_select(node1[~is_new_cell], 1, pred[~is_new_cell]-1).diag()#.detach().cpu()
        node0 = node0[~is_new_cell]
        if si==0: 
            remaps[si] = {mi.item(): mi.item() for mi in node0}
            mask0 = masks[si]
        assert edge_index[0].max() + 1 == mask0.max()
        mask1 = masks[si+1]
        passed_nid = []
        new_nuclei_num = 1
        for nid in edge_index[1].unique():
            if nid in passed_nid: continue
            assert nid not in remaps[si+1]
            if nid not in node1: # is a new cell
                remaps[si+1][nid.item()] = (edge_index[0].max()+new_nuclei_num).item()
                new_nuclei_num += 1
            else:
                nid_place = node1==nid
                scores = edge_pred[~is_new_cell][nid_place].max(1)[0]
                remaps[si+1][nid.item()] = (node0[nid_place][scores.argmax()]).item()
            passed_nid.append(nid)
        mask0 = mask1
    for i in range(len(remaps)):
        remaps[i] = {k+1:v+1 for k,v in remaps[i].items()}
    pre_remap = remaps[0]
    orig_remaps = [{v:k for k,v in map.items()} for map in orig_remaps]
    new_remaps = [{} for _ in range(len(masks)-1)]
    for i in range(1, len(masks)):
        new_nuclei_id = masks[:i].max() + 1
        new_remap = {}
        remap = remaps[i]
        mask = masks[i]
        new_nuclei_num = 0
        for k, v in remap.items():
            assert (mask == k).any()
            if v in pre_remap: 
                v = pre_remap[v]
                try:
                    new_remaps[i-1][orig_remaps[i][k]] = orig_remaps[i-1][v]
                except:
                    pass
            else: # is a new nuclei
                v = new_nuclei_id
                new_nuclei_id += 1
                assert v not in masks[:i]
            mask[mask == k] = v
            new_remap[k] = v
        pre_remap = new_remap
        masks[i] = mask       
    
    return masks, new_remaps

if __name__ == "__main__":
    main()