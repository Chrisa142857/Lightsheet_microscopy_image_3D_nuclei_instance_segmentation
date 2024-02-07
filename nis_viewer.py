import torch, os, re
from multiprocessing import Pool
from tqdm import trange
import vedo

def main():
    device = 'cuda:1'
    brain_tag = 'L73D766P4'
    pair_tag = 'pair15'
    seg_root = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{pair_tag}/{brain_tag}'
    save_root = f'/cajal/ACMUSERS/ziquanw/Lightsheet/renders/{pair_tag}'
    seg_paths, sortks = listdir_sorted(seg_root, "NIScpp", ftail="seg.zip", sortkid=3)
    seg_paths = [[p] for p in seg_paths]
    stitch_remap_all = torch.load(f"{seg_root}/{brain_tag}_remap.zip").to(device) # old id to new id map
    load_len = 15
    print("Load z range:", sortks[0], sortks[load_len])
    with Pool(min(load_len, 5)) as p:
        nis_stack = list(p.starmap(torch.load, seg_paths[:load_len]))
    depth_list = torch.LongTensor([nis.shape[0] for nis in nis_stack]).cumsum(0)[:-1]
    nis_stack = torch.cat(nis_stack).cpu()
    group_num = 3
    depths = []
    for i in range(len(depth_list)):
        if i//group_num >= len(depths): depths.append([])
        depths[i//group_num].append(depth_list[i])
    for depth in depths:
        strip_ind = []
        for d in depth:
            strip_ind.extend([i for i in range(d-1, d+8)])
        strip_ind = torch.LongTensor(strip_ind)
        nis_strip = nis_stack[strip_ind].to(device)
        max_label = nis_strip.max()
        min_label = nis_strip.min()
        print("nis_strip.shape, max_label, min", nis_strip.shape, max_label.cpu().detach().item(), min_label.cpu().detach().item())
        f1 = stitch_remap_all[0]<=max_label
        f2 = stitch_remap_all[0]>=min_label
        f = f1 & f2
        stitch_remap = stitch_remap_all[:, f]
        for si in trange(stitch_remap.shape[1], desc=f'Stitching depth {depth[0]}~{depth[-1]}'):
            z, y, x = torch.where(nis_strip==stitch_remap[0, si])
            nis_stack[strip_ind[z.cpu()], y.cpu(), x.cpu()] = stitch_remap[1, si].cpu()
            # stitch_nis = torch.zeros_like(nis_stack, dtype=torch.bool)
            # stitch_nis[strip_ind] = (nis_strip==stitch_remap[0, si]).detach().cpu()
            # nis_stack[stitch_nis] = stitch_remap[1, si].cpu()
        
    nis_mesh = vedo.Volume(nis_stack).isosurface()
    nis_mesh.write(f"{save_root}/NIS_mesh_{brain_tag}.ply")



def listdir_sorted(path, tag, ftail='_stitched.tif', sortkid=1):
    fs = os.listdir(path)
    fs = [os.path.join(path, f) for f in fs if tag in f and f.endswith(ftail)]
    ks = []
    for f in fs:
        k = f.split('/')[-1].split('_')[sortkid]
        k = int(re.sub("[^0-9]", "", k))
        ks.append(k)
    orgks = ks.copy()
    ks.sort()
    sorted_fs = []
    for k in ks:
        sorted_fs.append(fs[orgks.index(k)])
        
    return sorted_fs, ks

main()