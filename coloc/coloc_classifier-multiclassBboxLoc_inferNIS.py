import torch, os
import torch.nn as nn
from torchvision import models
from coloc_classifier_dataset import InferNisPatchColocDataset
from torch.utils.data import DataLoader
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='None')
    parser.add_argument('--weights_path', type=str)
    parser.add_argument('--weights_tag', type=str, default='16brain-layer23')
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--zmin', default=200, type=int)
    parser.add_argument('--zmax', default=400, type=int)
    parser.add_argument('--xyexpand', default=3, type=int)
    parser.add_argument('--gtag', default='P4', type=str)
    parser.add_argument('--ptag', type=str)
    parser.add_argument('--btag', default='L64D804P9', type=str)
    parser.add_argument('--mtag', default='resnet50ChAlign', type=str)
    args = parser.parse_args()
    infer_model(args)


class DualLocResNet(nn.Module):
    def __init__(self, class_names, gtag):
        super().__init__()
        if gtag == 'P14':
            self.model1 = models.resnet50()
            self.model2 = models.resnet50()
        else:
            self.model1 = models.resnet18()
            self.model2 = models.resnet18()
        
        self.xydownr = 500
        self.zdownr = 50
        ### Embedding plus
        self.output_lin1 = nn.Linear(self.model1.fc.in_features, len(class_names)-2)
        self.output_lin2 = nn.Linear(self.model2.fc.in_features, len(class_names)-2)
        self.locx_embed1 = nn.Embedding(1000, self.model1.fc.in_features)
        self.locy_embed1 = nn.Embedding(1000, self.model1.fc.in_features)
        self.locz_embed1 = nn.Embedding(1000, self.model1.fc.in_features)
        self.locx_embed2 = nn.Embedding(1000, self.model2.fc.in_features)
        self.locy_embed2 = nn.Embedding(1000, self.model2.fc.in_features)
        self.locz_embed2 = nn.Embedding(1000, self.model2.fc.in_features)

        ### Embedding cat
        self.cat_lin1 = nn.Sequential(
            nn.Linear(4*self.model1.fc.in_features, self.model1.fc.in_features),
            nn.Dropout(0.1),
        )
        self.cat_lin2 = nn.Sequential(
            nn.Linear(4*self.model2.fc.in_features, self.model2.fc.in_features),
            nn.Dropout(0.1),
        )

        ### Linear
        # self.output_lin1 = nn.Linear(self.model1.fc.in_features*2, len(class_names)-2)
        # self.output_lin2 = nn.Linear(self.model2.fc.in_features*2, len(class_names)-2)
        # self.loc_lin1 = nn.Linear(3, self.model2.fc.in_features)
        # self.loc_lin2 = nn.Linear(3, self.model2.fc.in_features)

        self.model1.fc = nn.Identity()
        self.model2.fc = nn.Identity()


    def forward(self, x, locx, locy, locz):
        y1 = self.model1(x)
        y2 = self.model2(x)
        ### Embedding plus
        # y1 = self.output_lin1(y1 + self.locx_embed1((locx//self.xydownr).long()) + self.locy_embed1((locy//self.xydownr).long()) + self.locz_embed1((locz//self.zdownr).long()))
        # y2 = self.output_lin2(y2 + self.locx_embed2((locx//self.xydownr).long()) + self.locy_embed2((locy//self.xydownr).long()) + self.locz_embed2((locz//self.zdownr).long()))
        ### Embedding cat
        locx = (locx//self.xydownr).long().clip(min=0, max=999)
        locy = (locy//self.xydownr).long().clip(min=0, max=999)
        locz = (locz//self.zdownr).long().clip(min=0, max=999)
        y1 = self.output_lin1(self.cat_lin1(
            torch.cat([y1, self.locx_embed1(locx), self.locy_embed1(locy), self.locz_embed1(locz)], -1)
        ))
        y2 = self.output_lin2(self.cat_lin2(
            torch.cat([y2, self.locx_embed2(locx), self.locy_embed2(locy), self.locz_embed2(locz)], -1)
        ))
        ### Linear
        # loc_y1 = self.loc_lin1(torch.stack([locx, locy, locz], -1).float())
        # loc_y2 = self.loc_lin2(torch.stack([locx, locy, locz], -1).float())
        # y1 = self.output_lin1(torch.cat([y1, loc_y1], -1))
        # y2 = self.output_lin2(torch.cat([y2, loc_y2], -1))
        return torch.stack([y1, y2], 1)


def infer_model(args):
    weights_path = args.weights_path
    if weights_path is None:
        fnlist = [fn for fn in os.listdir('model_weights') if fn.split('coloc')[0] == args.btag and args.weights_tag in fn]
        acclist = [float(fn.split('-')[-1][:-4]) for fn in fnlist]
        fn = fnlist[np.argmax(acclist)]
        weights_path = f'model_weights/{fn}'
        assert os.path.exists(weights_path)
    print('Loading model weights', weights_path)
    batch_size = args.batch_size
    device = args.device
    zrange = [args.zmin, args.zmax]
    mtag = args.mtag
    gtag = args.gtag
    ptag = args.ptag
    btag = args.btag
    if ptag is None:
        ptags = os.listdir('/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4')
        ptag = [ptag for ptag in ptags if np.any([btag in fn.split('_') for fn in os.listdir(f'/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch/P4/{ptag}')])][0]
    
    model = DualLocResNet(['Brn-Ctip-', 'Brn+', 'Ctip+', 'Brn+Ctip+'] if gtag == 'P4' else ['Sox9-Neun-', 'Sox9+', 'Neun+', 'Sox9+Neun+'], gtag)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model = model.to(device)
    dataset = InferNisPatchColocDataset(gtag, ptag, btag, zrange=zrange, device=device, use_bbox=True, bbox_expand=args.xyexpand)
    dloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size//8)
    # class_names = dataset.classes

    # Evaluation
    model.eval()
    # output = {'coordinate_x': np.zeros(len(dataset.ctz)), 'coordinate_y': np.zeros(len(dataset.ctz)), 'coordinate_z': np.zeros(len(dataset.ctz)), 'classification': np.zeros(len(dataset.ctz))}
    tile_names = []
    with torch.no_grad():
        for images, center_indecies in tqdm(dloader, desc='Inferring'):
            images = dataset.collate_fn(images, center_indecies).to(device)
            nis_locx, nis_locy, nis_locz = dloader.dataset.patch_info[center_indecies, 1:4].T.to(device)
            outputs = model(images, nis_locx, nis_locy, nis_locz)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted[..., 0] + predicted[..., 1] * 2
            predicted = predicted.cpu().detach()#.numpy()
            dloader.dataset.patch_info[center_indecies, -1] = predicted.long()
            patch_load_urls = dloader.dataset.patch_load_urls[center_indecies.detach().cpu().numpy(), 0]
            tile_names.extend([url.split('/')[-2] for url in patch_load_urls])

    output = {dloader.dataset.patch_info_header[i]: dloader.dataset.patch_info[dloader.dataset.patch_load_index, i].numpy() for i in range(len(dloader.dataset.patch_info_header))}
    output['Tile_name'] = tile_names
    pd.DataFrame(output).to_csv(f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_classification_addbrain_overfit_layer23/{gtag}/{ptag}_{btag}_{mtag}_results_Z{dataset.zrange[0]:04d}-{dataset.zrange[1]:04d}.csv', index=False)

if __name__ == "__main__":
    main()
