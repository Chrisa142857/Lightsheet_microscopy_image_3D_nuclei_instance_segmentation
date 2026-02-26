import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision
from coloc_classifier_dataset import TrainvalColocMultiClassBboxLocDataset
from torch.utils.data import DataLoader, Subset
import wandb
import copy
import random
import argparse
from datetime import datetime
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchsampler import ImbalancedDatasetSampler

random.seed(142857)

parser = argparse.ArgumentParser(description='None')
parser.add_argument('--proj_name', default='coloc-p4-resnet')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--add_btag', default='', type=str)
parser.add_argument('--gtag', default='P4', type=str)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--eval_ratio', default=0.1, type=float)
parser.add_argument('--nologging', action='store_false')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--weights_path', default='model_weights/coloc-p4-resnet_model_weights_2025-06-13 11-05-40.949827.pth')
parser.add_argument('--notrain', action='store_false')
parser.add_argument('--lr_scheduler', action='store_true')
parser.add_argument('--focalloss', action='store_true')
args = parser.parse_args()
def main():
    train_model(args)
    

# Utility function for image logging
def log_images_to_wandb(images, labels, predictions=None, class_names=None, tag="images"):
    image_grid = []
    for i in range(min(100, len(images))):
        img = images[i].cpu()
        img = torch.cat([img[0], img[1], img[2]], 1)[None]
        label = labels[i].item()
        pred = predictions[i].item() if predictions is not None else None
        # print(label, pred)
        caption = f"Label: {class_names[int(label)]}"
        if pred is not None:
            caption += f"\nPred: {class_names[int(pred)]}"
        image_grid.append(wandb.Image(img, caption=caption))
    wandb.log({tag: image_grid})

class DualLocResNet(nn.Module):
    def __init__(self, class_names):
        super().__init__()
        
        if args.gtag == 'P14':
            self.model1 = models.resnet50()
            self.model2 = models.resnet50()
        else:
            self.model1 = models.resnet18()
            self.model2 = models.resnet18()
        # self.model1 = models.resnet18()
        # self.model2 = models.resnet18()
        # self.model1 = models.resnet50()
        # self.model2 = models.resnet50()
        
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
        y1 = self.output_lin1(self.cat_lin1(
            torch.cat([y1, self.locx_embed1((locx//self.xydownr).long()), self.locy_embed1((locy//self.xydownr).long()), self.locz_embed1((locz//self.zdownr).long())], -1)
        ))
        y2 = self.output_lin2(self.cat_lin2(
            torch.cat([y2, self.locx_embed2((locx//self.xydownr).long()), self.locy_embed2((locy//self.xydownr).long()), self.locz_embed2((locz//self.zdownr).long())], -1)
        ))
        ### Linear
        # loc_y1 = self.loc_lin1(torch.stack([locx, locy, locz], -1).float())
        # loc_y2 = self.loc_lin2(torch.stack([locx, locy, locz], -1).float())
        # y1 = self.output_lin1(torch.cat([y1, loc_y1], -1))
        # y2 = self.output_lin2(torch.cat([y2, loc_y2], -1))
        return torch.stack([y1, y2], 1)


# Example training setup
def train_model(args):
    training = args.notrain
    weights_path = args.weights_path
    proj_name = args.proj_name
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    eval_ratio = args.eval_ratio
    logging = False #args.nologging
    device = args.device
    if logging and training:
        # Initialize Weights & Biases
        wandb.init(dir=f"{proj_name}-wandb", project=proj_name, config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "architecture": "ResNet",
            "dataset": proj_name
        })
        
    # Define transformations and load dataset (CIFAR-10)
    train_transform = transforms.Compose([
        transforms.ColorJitter(0.5),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
    ])

    # ####### P14 ##################################################################
    if args.gtag == 'P14':
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L106P5_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L106P5_coloc_classifications.csv'
        train_dataset1 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, classes=['Sox-Neun-', 'Sox+', 'Neun+', 'Sox+Neun+'], transform=train_transform, device=device)
        eval_dataset1 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, classes=['Sox-Neun-', 'Sox+', 'Neun+', 'Sox+Neun+'], transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L106P3_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L106P3_coloc_classifications.csv'
        train_dataset2 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, classes=['Sox-Neun-', 'Sox+', 'Neun+', 'Sox+Neun+'], transform=train_transform, device=device)
        eval_dataset2 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, classes=['Sox-Neun-', 'Sox+', 'Neun+', 'Sox+Neun+'], transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L102P1_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L102P1_coloc_classifications.csv'
        train_dataset3 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, classes=['Sox-Neun-', 'Sox+', 'Neun+', 'Sox+Neun+'], transform=train_transform, device=device)
        eval_dataset3 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, classes=['Sox-Neun-', 'Sox+', 'Neun+', 'Sox+Neun+'], transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L102P2_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L102P2_coloc_classifications.csv'
        train_dataset4 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, classes=['Sox-Neun-', 'Sox+', 'Neun+', 'Sox+Neun+'], transform=train_transform, device=device)
        eval_dataset4 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, classes=['Sox-Neun-', 'Sox+', 'Neun+', 'Sox+Neun+'], transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L102P1_patches_corpuscallosummask-v1.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L102P1_coloc_corpuscallosummask-v1_classifications.csv'
        train_dataset5 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, classes=['Sox-Neun-', 'Sox+', 'Neun+', 'Sox+Neun+'], transform=train_transform, device=device)
        eval_dataset5 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, classes=['Sox-Neun-', 'Sox+', 'Neun+', 'Sox+Neun+'], transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L102P1_corpuscallosummask_v2_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L102P1_corpuscallosummask_v2_classifications.csv'
        train_dataset6 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, classes=['Sox-Neun-', 'Sox+', 'Neun+', 'Sox+Neun+'], transform=train_transform, device=device)
        eval_dataset6 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, classes=['Sox-Neun-', 'Sox+', 'Neun+', 'Sox+Neun+'], transform=None, device=device)
        class_names = train_dataset2.classes
        train_datasets = [train_dataset1, train_dataset2, train_dataset3, train_dataset4, train_dataset5, train_dataset6]
        eval_datasets = [eval_dataset1, eval_dataset2, eval_dataset3, eval_dataset4, eval_dataset5, eval_dataset6]
        # train_datasets = [train_dataset1, train_dataset2, train_dataset3, train_dataset4, train_dataset5]
        # eval_datasets = [eval_dataset1, eval_dataset2, eval_dataset3, eval_dataset4, eval_dataset5]
        # train_datasets = [train_dataset1, train_dataset2, train_dataset3, train_dataset4]
        # eval_datasets = [eval_dataset1, eval_dataset2, eval_dataset3, eval_dataset4]

    # #############################################################################
    # ##### P4 ##################################################################
    else:
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L66D764P8_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L66D764P8_isocortex_patches_coloc_annotation.csv'
        train_dataset1 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, transform=train_transform, device=device)
        eval_dataset1 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L66D764P5_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L66D764P5_isocortex_patches_coloc_annotation.csv'
        train_dataset2 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, transform=train_transform, device=device)
        eval_dataset2 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L73D766P7_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L73D766P7_isocortex_patches_coloc_annotation.csv'
        train_dataset3 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, transform=train_transform, device=device)
        eval_dataset3 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L64D804P9_isocortex_patches_L64D804P9_z300-600_z800-1000.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L64D804P9_z300-600_z800-1000_classifications.csv'
        patch_info = 'coloc_ann_patch/L64D804P9_isocortex_patch_info_L64D804P9_z300-600_z800-1000.csv'
        train_dataset4 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
        eval_dataset4 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L64D804P3_isocortex_patches_z300-600_z800-1000.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L64D804P3_z300-600_z800-1000_classifications.csv'
        patch_info = 'coloc_ann_patch/L64D804P3_isocortex_patch_info_z300-600_z800-1000.csv'
        train_dataset5 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
        eval_dataset5 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L74D769P4_coloc-mask-v1_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L74D769P4_coloc-mask-v1_classifications.csv'
        patch_info = 'coloc_ann_patch/L74D769P4_coloc-mask-v1_isocortex_patch_info.csv'
        train_dataset6 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
        eval_dataset6 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L74D769P8_coloc-mask-v1_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L74D769P8_coloc-mask-v1_classifications.csv'
        patch_info = 'coloc_ann_patch/L74D769P8_coloc-mask-v1_isocortex_patch_info.csv'
        train_dataset7 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
        eval_dataset7 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L73D766P5_coloc-mask-v1_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L73D766P5_coloc-mask-v1_classifications.csv'
        patch_info = 'coloc_ann_patch/L73D766P5_coloc-mask-v1_isocortex_patch_info.csv'
        train_dataset8 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
        eval_dataset8 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L73D766P4_coloc-mask-v1_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L73D766P4_coloc-mask-v1_classifications.csv'
        patch_info = 'coloc_ann_patch/L73D766P4_coloc-mask-v1_isocortex_patch_info.csv'
        train_dataset9 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
        eval_dataset9 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L73D766P9_coloc-mask-v1_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L73D766P9_coloc-mask-v1_classifications.csv'
        patch_info = 'coloc_ann_patch/L73D766P9_coloc-mask-v1_isocortex_patch_info.csv'
        train_dataset10 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
        eval_dataset10 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L69D764P6_coloc-mask-v1_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L69D764P6_coloc-mask-v1_classifications.csv'
        patch_info = 'coloc_ann_patch/L69D764P6_coloc-mask-v1_isocortex_patch_info.csv'
        train_dataset11 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
        eval_dataset11 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L69D764P9_coloc-mask-v1_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L69D764P9_coloc-mask-v1_classifications.csv'
        patch_info = 'coloc_ann_patch/L69D764P9_coloc-mask-v1_isocortex_patch_info.csv'
        train_dataset12 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
        eval_dataset12 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L77D764P9_coloc-mask-v1_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L77D764P9_coloc-mask-v1_classifications.csv'
        patch_info = 'coloc_ann_patch/L77D764P9_coloc-mask-v1_isocortex_patch_info.csv'
        train_dataset13 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
        eval_dataset13 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L77D764P2_coloc-mask-v1_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L77D764P2_coloc-mask-v1_classifications.csv'
        patch_info = 'coloc_ann_patch/L77D764P2_coloc-mask-v1_isocortex_patch_info.csv'
        train_dataset14 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
        eval_dataset14 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L64D804P3_coloc-mask-v1_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L64D804P3_coloc_classifications.csv'
        patch_info = 'coloc_ann_patch/L64D804P3_coloc-mask-v1_isocortex_patch_info.csv'
        train_dataset15 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
        eval_dataset15 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
        patch_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L66D764P6_coloc-mask-v1_isocortex_patches.tif'
        ann_path = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/coloc_ann_patch/L66D764P6_coloc_classifications.csv'
        patch_info = 'coloc_ann_patch/L66D764P6_coloc-mask-v1_isocortex_patch_info.csv'
        train_dataset16 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
        eval_dataset16 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
        train_datasets = [train_dataset1, train_dataset2, train_dataset3, train_dataset4, train_dataset5, train_dataset6, train_dataset7, train_dataset8, train_dataset9, train_dataset10, train_dataset11, train_dataset12, train_dataset13, train_dataset14, train_dataset15, train_dataset16]
        eval_datasets = [eval_dataset1, eval_dataset2, eval_dataset3, eval_dataset4, eval_dataset5, eval_dataset6, eval_dataset7, eval_dataset8, eval_dataset9, eval_dataset10, eval_dataset11, eval_dataset12, eval_dataset13, eval_dataset14, eval_dataset15, eval_dataset16]
    # train_datasets = [train_dataset1, train_dataset2, train_dataset3, train_dataset4, train_dataset5]
    # eval_datasets = [eval_dataset1, eval_dataset2, eval_dataset3, eval_dataset4, eval_dataset5]
    # train_datasets = [train_dataset6]
    # eval_datasets = [eval_dataset6]
    dsetindex_evalstart = sum([len(d) for d in train_datasets])
    if args.add_btag != '':
        if args.gtag == 'P4':
            patch_path = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{args.gtag}_ann_patches/brainwise/{args.add_btag}_isocortex_patches.tif'
            ann_path = f'/lichtman/Felix/Lightsheet/P4/numorph_src/{args.add_btag}_coloc_classifications.csv'
            patch_info = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{args.gtag}_ann_patches/brainwise/{args.add_btag}_isocortex_patch_info.csv'
            train_dataset17 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
            eval_dataset17 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
            if args.add_btag not in ['L57D855P4', 'L35D719P3']:
                train_datasets += [train_dataset17]
            else:
                dsetindex_evalstart = 0
                train_datasets = [train_dataset17]
            eval_datasets = [eval_dataset17]
            ann_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/train_patches_layer23_bgrm/{args.add_btag}_coloc_layer23_bgrm_classifications.csv'
            assert os.path.exists(ann_path), f"Expected file {ann_path} does not exist"
            if os.path.exists(ann_path):
                patch_path = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{args.gtag}_ann_patches_layer23_bgrm/brainwise/{args.add_btag}_isocortex_patches.tif'
                patch_info = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{args.gtag}_ann_patches_layer23_bgrm/brainwise/{args.add_btag}_isocortex_patch_info.csv'
                train_dataset18 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
                eval_dataset18 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
                train_datasets += [train_dataset18]
                eval_datasets = [eval_dataset17, eval_dataset18]
            
            ann_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/train_patches_layer23_bgrm/{args.add_btag}_coloc_layer23_bgrm_round2_classifications.csv'
            if os.path.exists(ann_path):
                patch_path = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{args.gtag}_ann_patches_layer23_bgrm_round2/brainwise/{args.add_btag}_isocortex_patches.tif'
                patch_info = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{args.gtag}_ann_patches_layer23_bgrm_round2/brainwise/{args.add_btag}_isocortex_patch_info.csv'
                train_dataset19 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
                eval_dataset19 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
                train_datasets += [train_dataset19]
                eval_datasets = [eval_dataset17, eval_dataset18, eval_dataset19]
        else:
            patch_path = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{args.gtag}_ann_patches/brainwise/{args.add_btag}_patches.tif'
            ann_path = f'/lichtman/Ian/Lightsheet/P14/stitched/female/numorph_src/{args.add_btag}_coloc_classifications.csv'
            patch_info = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_nis/{args.gtag}_ann_patches/brainwise/{args.add_btag}_patch_info.csv'
            train_dataset17 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=train_transform, device=device)
            eval_dataset17 = TrainvalColocMultiClassBboxLocDataset(patch_path, ann_path, patch_info=patch_info, transform=None, device=device)
            train_datasets += [train_dataset17]
            eval_datasets = [eval_dataset17]
        

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    eval_dataset = torch.utils.data.ConcatDataset(eval_datasets)
    class_names = np.array(train_dataset1.classes)
    ############################################################################
    print([
        [torch.bincount(torch.from_numpy(train_dataset1.label).long()), len(train_dataset1), len(train_dataset)]
        for train_dataset1 in train_datasets])
    print([
        [torch.bincount(torch.from_numpy(eval_dataset1.label).long()), len(eval_dataset1), len(eval_dataset)]
        for eval_dataset1 in eval_datasets])
    labels = torch.cat([
        torch.from_numpy(train_dataset1.label).long()
        for train_dataset1 in train_datasets])
    # exit()
    dsetindex = torch.arange(len(train_dataset)).tolist()
    random.shuffle(dsetindex)
    dsetindex = torch.LongTensor(dsetindex)
    eval_point = int(len(train_dataset)*eval_ratio)
    train_dataset = Subset(train_dataset, dsetindex[eval_point:])
    if args.add_btag == '':
        test_dataset = Subset(eval_dataset, dsetindex[:eval_point])
    else:
        test_index = dsetindex[:eval_point][dsetindex[:eval_point]>=dsetindex_evalstart] - dsetindex_evalstart
        assert len(test_index)>0, f'{dsetindex[:eval_point]}, {dsetindex_evalstart}'
        test_dataset = Subset(eval_dataset, test_index)

    train_labels = labels[dsetindex[eval_point:]]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=ImbalancedDatasetSampler(train_dataset, labels=train_labels))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    # Set up model, loss, and optimizer
    model = DualLocResNet(class_names)
    # model = models.resnet50()
    # # model.fc = nn.Linear(model.fc.in_features, len(class_names))
    # model.fc = nn.Linear(model.fc.in_features, len(class_names)-2, 2)
    if not training:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    # model = models.VisionTransformer(
    #     image_size=50,
    #     patch_size=10,
    #     num_layers=12,
    #     num_heads=12,
    #     hidden_dim=768,
    #     mlp_dim=3072,
    #     num_classes=len(class_names)
    # )
    model = model.to(device)
    if not training:
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        y_trues = [[], []]
        y_scores = [[], []]
        predicteds = []
        labelall = []
        # bboxz = []
        with torch.no_grad():
            for images, labels, locs in test_loader:
                # bboxz.append(test_loader.dataset.dataset[0].current_bboxz)
                images, labels, locs = images.to(device), labels.to(device), locs.to(device)
                outputs = model(images, locs[:, 0], locs[:, 1], locs[:, 2])
                y_trues[0].append(labels[..., 0].detach().cpu())
                y_trues[1].append(labels[..., 1].detach().cpu())
                
                labels = labels[..., 0] + labels[..., 1] * 2
                
                _, predicted = torch.max(outputs.data, 1)
                y_scores[0].append(outputs.data.softmax(1)[:, 1, 0].detach().cpu())
                y_scores[1].append(outputs.data.softmax(1)[:, 1, 1].detach().cpu())

                predicted = predicted[..., 0] + predicted[..., 1] * 2
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                labelall.append(labels.detach().cpu())
                predicteds.append(predicted.detach().cpu())

        test_accuracy = 100 * correct / total
        print(f"Correctly classified {test_accuracy:.2f}% of cells")
        labelall = torch.cat(labelall)
        predicteds = torch.cat(predicteds)
        for li in labelall.unique():
            idx = labelall==li
            total = labelall[idx].size(0)
            correct = (predicteds[idx] == labelall[idx]).sum().item()
            test_accuracy = 100 * correct / total
            print(f"Correctly classified {test_accuracy:.2f}% of class-{(int(li)+1):02d} cells")
        # print(bboxz)
        chn = 0
        for y_true, y_score in zip(y_trues, y_scores):
            y_true, y_score = torch.cat(y_true), torch.cat(y_score)
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            print(f'Channel {chn}', fpr, tpr, thresholds)
            chn += 1
            plt.figure(figsize=(5,5))
            display = RocCurveDisplay.from_predictions(
                y_true, y_score,
                name=f"Channel {chn}, n={sum(y_true)}",
                # curve_kwargs=dict(color="darkorange"),
                # plot_chance_level=True,
                # despine=True,
            )
            _ = display.ax_.set(
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
            )
            plt.savefig(f'roc_fig/{proj_name}_roc_channel{chn}.png')
            plt.savefig(f'roc_fig/{proj_name}_roc_channel{chn}.svg')
            torch.save({"y_true": y_true, "y_score": y_score}, f'roc_fig/{proj_name}_roc_channel{chn}.pt')
        return 
    
    if not args.focalloss:
        criterion = nn.CrossEntropyLoss()
    else:
        def criterion(cls_logits, cls_targets_idx, C=2):
            cls_targets = torch.nn.functional.one_hot(cls_targets_idx, num_classes=C) \
            .to(torch.float32)
            loss = torchvision.ops.sigmoid_focal_loss(
                inputs=cls_logits, targets=cls_targets,
                alpha=0.25, gamma=2.0, reduction="mean"
            )
            return loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if args.lr_scheduler: lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs//2), gamma=0.1)
    bestacc = 0
    best_train_log_image = train_log_image = None
    best_eval_log_image = eval_log_image = None
    bestepoch = None
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        sample_logged = False

        for images, labels, locs in train_loader:
            images, labels, locs = images.to(device), labels.to(device).long(), locs.to(device)
            optimizer.zero_grad()
            outputs = model(images, locs[:, 0], locs[:, 1], locs[:, 2])
            loss = criterion(outputs[..., 0], labels[..., 0]) + criterion(outputs[..., 1], labels[..., 1])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            labels = labels[..., 0] + labels[..., 1] * 2
            predicted = predicted[..., 0] + predicted[..., 1] * 2
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log a batch of images once per epoch
            if not sample_logged and logging:
                train_log_image = [images, labels, predicted, class_names]
                sample_logged = True

        train_accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        sample_logged = False
        with torch.no_grad():
            for images, labels, locs in test_loader:
                images, labels, locs = images.to(device), labels.to(device), locs.to(device)
                outputs = model(images, locs[:, 0], locs[:, 1], locs[:, 2])
                _, predicted = torch.max(outputs.data, 1)
                labels = labels[..., 0] + labels[..., 1] * 2
                predicted = predicted[..., 0] + predicted[..., 1] * 2
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if not sample_logged and logging:
                    eval_log_image = [images, labels, predicted, class_names]
                    sample_logged = True

        test_accuracy = 100 * correct / total
        if test_accuracy >= bestacc:
            bestacc = test_accuracy
            bestmodel = copy.deepcopy(model)
            best_train_log_image = train_log_image
            best_eval_log_image = eval_log_image
            bestepoch = epoch

        if logging:
            wandb.log({"loss": avg_loss, "train_accuracy": train_accuracy, "eval_accuracy": test_accuracy})

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Eval Accuracy: {test_accuracy:.2f}%")
        if args.lr_scheduler: lr_scheduler.step()

    # Save model
    mweight_fn = f"model_weights/{args.add_btag}{proj_name}_model_weights_{datetime.now()}_bestACC-{bestacc:.2f}.pth".replace(':', '-').replace(' ', '-')
    torch.save(bestmodel.state_dict(), mweight_fn)
    if logging and best_train_log_image is not None:
        images, labels, predicted, class_names = best_train_log_image
        log_images_to_wandb(images, labels, predicted, class_names, tag=f"train_epoch_{bestepoch}")
        images, labels, predicted, class_names = best_eval_log_image
        log_images_to_wandb(images, labels, predicted, class_names, tag=f"eval_epoch_{bestepoch}")
        wandb.save(mweight_fn)

if __name__ == "__main__":
    main()
