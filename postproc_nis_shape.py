import argparse
import torch, os, sys
from datetime import datetime
from tqdm import trange, tqdm
import numpy as np

parser = argparse.ArgumentParser(description='None')
parser.add_argument('--gtag', type=str, default='P14')
parser.add_argument('--ptag', type=str, default='male')
parser.add_argument('--btag', type=str, default='L88P2')
parser.add_argument('--device', type=str, default='cuda:2')
args = parser.parse_args()
pair_tag = args.ptag
brain_tag = args.btag
gtag = args.gtag
device = args.device

r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/{gtag}'
pbtag_ls = []
for ptag in os.listdir(r):
    for btag in os.listdir(f'{r}/{ptag}'):
        pbtag_ls.append([ptag, btag])
pbtag_ls = list(reversed(pbtag_ls))

for ptag, btag in pbtag_ls:
    if btag.split('_')[1] == brain_tag: break
assert ptag == pair_tag, f"{ptag} != {pair_tag}"
print(ptag, btag, brain_tag)


if gtag == 'P4':
    overlap_r = 0.2
    root1 = '/cajal/Felix/Lightsheet/P4'
    root2 = '/lichtman/Felix/Lightsheet/P4'
    root = f'{root1}/{ptag}/{btag}' if os.path.exists(f'{root1}/{ptag}/{btag}') else f'{root2}/{ptag}/{btag}'
else:
    overlap_r = 0.1
    root = f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P14/{ptag}/{btag}'
assert os.path.exists(root), root
result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/{gtag}/{ptag}/{btag}'
# result_root = result_path + '/UltraII[%02d x %02d]'
# stack_names = [f for f in os.listdir(result_root % (0, 0)) if f.endswith('instance_center.zip')]


lrange = 1000
stack_names = None
for tile_fn in os.listdir(result_path):
    if not tile_fn.startswith('UltraII'): continue
    stack_names = [f for f in os.listdir(f'{result_path}/{tile_fn}') if f.endswith('instance_center.zip')]
    for stack_name in stack_names:
        labelfn = f"{result_path}/{tile_fn}/{stack_name.replace('instance_center', 'instance_label')}"
        featfn = f"{result_path.replace('cajal', 'scheibel')}/{tile_fn}/{stack_name.replace('instance_center', 'instance_pa')}"
        savefn = f"{result_path.replace('cajal', 'scheibel')}/{tile_fn}/{stack_name.replace('instance_center', 'instance_pa_sorted')}"
        if not os.path.exists(featfn):
            print('None feat', featfn)
            continue
        if os.path.exists(savefn):
            print('Exist save', savefn)
            continue
        print(datetime.now(), tile_fn, stack_name)
        label = torch.load(labelfn, map_location='cpu').long().to(device)
        feat = torch.load(featfn, map_location='cpu')
        feat_label = feat['NIS_label'].long()
        feat = feat['PA']
        assert max(feat_label) <= max(label), f'{max(feat_label)} > {max(label)}'
        assert min(feat_label) >= min(label), f'{min(feat_label)} < {min(label)}'
        
        new_feat = torch.zeros(len(label), feat.shape[1], feat.shape[2])
        for labeli in range(0, len(feat_label), lrange):
            feat_label_batch = feat_label[labeli:labeli+lrange].to(device)
            feat2orig = feat_label_batch[:, None] == label[None, :]
            # assert (feat2orig.sum(1) == 1).all(), torch.where(feat2orig.sum(1) != 1)[0]
            new_feat[torch.where(feat2orig)[1]] = feat[labeli:labeli+lrange]

        torch.save(new_feat, savefn)
        
