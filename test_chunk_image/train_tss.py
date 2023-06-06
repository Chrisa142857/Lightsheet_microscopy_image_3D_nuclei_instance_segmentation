import torch, torchvision, imageio
import numpy as np
import tifffile as tif
from tqdm import tqdm, trange
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tss_utils import build_graph_from_ram

from two_slice_stitch import StitchModel


def main():    # tss_train_output (epoch115.pth)
    output_dir = 'tss_trainv2_output_filter_pyramid'
    lr = 0.001
    epoch_total = 300
    step_size = 30
    lr_gamma = 0.5
    batch_size = 64
    device = 'cuda:1'
    with open('train_list.txt', 'r') as f:        
        trainpath_list = f.read().split('\n')[:-1]
    train_maskpath_list = trainpath_list
    train_imgpath_list = [p.replace('/masks', '/images').replace('_masks', '') for p in trainpath_list]
    with open('val_list.txt', 'r') as f:        
        valpath_list = f.read().split('\n')[:-1]
    val_maskpath_list = valpath_list
    val_imgpath_list = [p.replace('/masks', '/images').replace('_masks', '') for p in valpath_list]
    model = StitchModel(device)    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_gamma)
    # criteria = torchvision.ops.focal_loss.sigmoid_focal_loss
    criteria = torch.nn.CrossEntropyLoss()
    train_dataset = TSSdataset(train_maskpath_list, train_imgpath_list, device, model.preprocess)
    val_dataset = TSSdataset(val_maskpath_list, val_imgpath_list, device, model.preprocess)
    with open('%s/trainval_log.csv' % output_dir, 'w') as f:
        f.write('epoch, train loss, train acc, train acc0, train acc1, val loss, val acc, val acc0, val acc1\n')
    for i in range(epoch_total):
        epoch_info = "Epoch %d lr: %.6f " % (i, scheduler.get_last_lr()[0])
        losses_train, acc_train, acc0_train, acc1_train = trainval_one_epoch(train_dataset, model, optimizer, criteria, device, batch_size, istrain=True)
        scheduler.step()
        torch.save(model.state_dict(), '%s/epoch%d.pth' % (output_dir, i))
        losses_val, acc_val, acc0_val, acc1_val = trainval_one_epoch(val_dataset, model, optimizer, criteria, device, 1, istrain=False)
        epoch_info += "acc_train: %.6f, loss_train: %.6f, acc_val: %.6f, loss_val: %.6f" % (
            np.mean(acc_train), np.mean(losses_train), np.mean(acc_val), np.mean(losses_val)
        )
        print(epoch_info)
        with open('%s/trainval_log.csv' % output_dir, 'a') as f:
            f.write("%d,%f,%f,%f,%f,%f,%f,%f,%f\n" % (
                i, 
                np.mean(losses_train), np.mean(acc_train), np.mean(acc0_train), np.mean(acc1_train), 
                np.mean(losses_val), np.mean(acc_val), np.mean(acc0_val), np.mean(acc1_val)
            ))

import matplotlib.pyplot as plt
def trainval_one_epoch(dataset, model, optimizer, criteria, device, batch_size, istrain=True):
    focalloss_alpha = 0.75
    losses = []
    accs = []
    acc0s = [0]
    acc1s = [0]
    if istrain:
        model.train()
    else:
        model.eval()
    dloader = DataLoader(dataset, batch_size=batch_size, shuffle=istrain)
    iterator = tqdm(dloader) if istrain else dloader
    # all_label = []
    for input in iterator:
        label = input.y
        # all_label.extend(label.tolist())
        # continue
        edge_pred = model(input)
        # loss = criteria(edge_pred, label, reduction='mean', alpha=focalloss_alpha)
        # edge_pred = edge_pred.sigmoid().cpu().detach()
        loss = criteria(edge_pred, label)
        pred = edge_pred.argmax(1)
        acc = sum(pred==label)/len(label)
        accs.append(acc.item())
        # sort_id = torch.argsort(edge_pred, descending=True)
        # removed_edge = []
        # for i in sort_id:
        #     if i in removed_edge: continue
        #     if edge_pred[i] < 0.5: break
        #     pred_ind = input.edge_index[:, i]
        #     pred_score = edge_pred[i]
        #     edge_pred[input.edge_index[0]==pred_ind[0]] = 0
        #     edge_pred[input.edge_index[1]==pred_ind[0]] = 0
        #     removed_edge.extend(torch.where(input.edge_index[0]==pred_ind[0])[0].tolist())
        #     removed_edge.extend(torch.where(input.edge_index[1]==pred_ind[0])[0].tolist())
        #     edge_pred[input.edge_index[0]==pred_ind[1]] = 0
        #     edge_pred[input.edge_index[1]==pred_ind[1]] = 0
        #     removed_edge.extend(torch.where(input.edge_index[0]==pred_ind[1])[0].tolist())
        #     removed_edge.extend(torch.where(input.edge_index[1]==pred_ind[1])[0].tolist())
        #     edge_pred[i] = pred_score

        # pos_data_num = len(edge_pred[label==1])
        # neg_data_num = len(edge_pred[label==0])
        # acc1 = (edge_pred[label==1]>=0.5).sum() / pos_data_num
        # acc0 = (edge_pred[label==0]<0.5).sum() / neg_data_num
        # acc = acc1/2 + acc0/2
        losses.append(loss.item())
        # accs.append(acc.detach().cpu().numpy())
        # acc0s.append(acc0.detach().cpu().numpy())
        # acc1s.append(acc1.detach().cpu().numpy())
        
        if istrain:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # plt.hist(all_label, bins=6)
    # plt.savefig('temp.png')
    # exit()
    return losses, accs, acc0s, acc1s


class TSSdataset(Dataset):

    def __init__(self, maskpath_list, imgpath_list, device, preprocessor) -> None:
        self.labels = []
        self.inputs = []
        self.cur_labels = []
        for i in trange(len(maskpath_list), desc="Initializing data"):
            masks = tif.imread(maskpath_list[i]).astype(np.int32)
            imgs = imageio.v2.mimread(imgpath_list[i])
            data_list, _, _ = build_graph_from_ram(imgs, masks, preprocessor, device=device, istrain=True)
            self.inputs.extend(data_list)

    def __getitem__(self, idx):
        return self.inputs[idx]

    def __len__(self):
        return len(self.inputs)

if __name__ == "__main__":
    main()