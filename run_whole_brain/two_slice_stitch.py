import torch
from torch import nn
import numpy as np
from tss_model import MLP, TSS, get_model_config
import torch.nn.functional as F
from tqdm import trange
from datetime import datetime

class StitchModel(nn.Module):
    '''
        input: two slices, each is [mask, bbox, flow]
        output: edge classification
    '''
    def __init__(self, device) -> None:
        super(StitchModel, self).__init__()
        self.device = device
        self.distance_thr = 20
        tss_config = get_model_config()
        self.nets = TSS(model_params=tss_config)
        self.topn = 2
        self.multi_gpu = 2
        self.classifier = MLP(input_dim=tss_config['classifier_feats_dict']['edge_out_dim']*self.topn, fc_dims=[128, 64, 32, self.topn+1], use_batchnorm=True, dropout_p=0.3)
        # if self.multi_gpu:
        #     self.pool1 = Pool(processes=self.multi_gpu)
        #     self.pool2 = Pool(processes=self.multi_gpu)

    def preprocess(self, data, pool1=None, pool2=None):
        data1, data2 = data
        # H x W, H x W, N x 4, 3 x H x W or 2 x H x W
        img1, mask1, bbox1, flow1 = data1
        img2, mask2, bbox2, flow2 = data2
        img1 = torch.from_numpy(img1.astype(np.float32)).to(self.device)
        img2 = torch.from_numpy(img2.astype(np.float32)).to(self.device)
        bbox1 = torch.from_numpy(bbox1.astype(np.float32)).to(self.device)
        bbox2 = torch.from_numpy(bbox2.astype(np.float32)).to(self.device)
        mask1 = torch.from_numpy(mask1).to(self.device)
        mask2 = torch.from_numpy(mask2).to(self.device)
        flow1 = torch.from_numpy(flow1.astype(np.float32)).to(self.device)
        flow2 = torch.from_numpy(flow2.astype(np.float32)).to(self.device)
        assert (not (bbox1[:, 2] == 0).any()) & (not (bbox1[:, 3] == 0).any())
        assert (not (bbox2[:, 2] == 0).any()) & (not (bbox2[:, 3] == 0).any())
        N1 = len(bbox1)
        if self.multi_gpu == 0:
            print(datetime.now(), "Get node feature of nuclei instance of previous slice")
            node1, zflow1 = run_all_node(img1, mask1, flow1) # N1 x 27
            print(datetime.now(), "Get node feature of nuclei instance of next slice")
            node2, zflow2 = run_all_node(img2, mask2, flow2) # N2 x 27
        else:
            results = pool1.starmap(multi_gpu_node_feat, [(img1.cuda(0), mask1.cuda(0), flow1.cuda(0), 'prev'), (img2.cuda(1), mask2.cuda(1), flow2.cuda(1), 'next')], )
            node1, zflow1, tag1 = results[0]
            node1, zflow1 = node1.to(self.device), zflow1.to(self.device)
            node2, zflow2, tag2 = results[1]
            node2, zflow2 = node2.to(self.device), zflow2.to(self.device)
            assert tag1 == 'prev' and tag2 == 'next'
                
        x = torch.cat([node1, node2]).float().to(self.device) # N1+N2 x node_feat_size
        print(datetime.now(), "Compute distance between two node sets to find N-neighbors")
        # distance = get_distance(bbox1.to('cpu'), bbox2.to('cpu')).to(self.device) # N1 x N2
        distance = get_distance_low_ram(bbox1.to('cpu'), bbox2.to('cpu'))#.to(self.device) # N1 x N2
        print(datetime.now(), "Determine edge index based on deistance")
        if self.multi_gpu == 0:
            out, _ = build_one_graph(distance, self.distance_thr, range(N1), 0, [[], []], [[], []], topn=self.topn)
        else:
            split_p = int(N1/2)
            inputs = [(distance[:split_p].cuda(0), self.distance_thr, range(split_p), 0, [[], []], [[], []], self.topn, True),
                      (distance[split_p:].cuda(1), self.distance_thr, range(N1-split_p), 0, [[], []], [[], []], self.topn, True, split_p)]
            results = pool2.starmap(build_one_graph, inputs)
            out = [[]]
            for ri in range(len(results)):
                out[0] += results[ri][0][0]
        print(datetime.now(), "Compute edge feature")
        edge_index = [[], []]
        edge_attr = []
        for indicies in out[0]:
            assert len(indicies) >= 2
            edge_index[0].extend([indicies[0] for _ in range(len(indicies)-1)])
            edge_index[1].extend([ind+N1 for ind in indicies[1:]])
            edge_attr.extend([
                torch.cat([
                    # hist1[indicies[0]], hist2[ind], 
                    torch.stack([zflow1[indicies[0]], zflow2[ind], bbox1[indicies[0], 0] - bbox2[ind, 0], bbox1[indicies[0], 1] - bbox2[ind, 1], bbox1[indicies[0], 2] / bbox2[ind, 2], bbox1[indicies[0], 3] / bbox2[ind, 3]]),
                ])
            for ind in indicies[1:]])
        # for indicies in out[1]:
        #     edge_index[0].extend([indicies[0]+N1 for _ in range(len(indicies)-1)])
        #     edge_index[1].extend([ind for ind in indicies[1:]])
        #     edge_attr.extend([
        #         torch.cat([
        #             # hist1[ind], hist2[indicies[0]],
        #             torch.stack([zflow1[ind], zflow2[indicies[0]], bbox1[ind, 0] - bbox2[indicies[0], 0], bbox1[ind, 1] - bbox2[indicies[0], 1], bbox1[ind, 2] / bbox2[indicies[0], 2], bbox1[ind, 3] / bbox2[indicies[0], 3]]),
        #         ])
        #     for ind in indicies[1:]])
        edge_index = torch.LongTensor(edge_index).to(self.device)
        edge_attr = torch.stack(edge_attr).to(self.device)
        print(datetime.now(), "Done")
        return x, edge_index, edge_attr

    def forward(self, data):
        # x, edge_index, edge_attr = self.preprocess(data)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.nets(x, edge_index, edge_attr)
        ## accumlate
        # x = torch.stack(x)
        # x = x.sum(0)
        ## or use the last layer
        x = x[-1]
        ## reshape topn node and classify
        node0 = edge_index[0].unique()
        x = torch.stack([x[edge_index[0]==nid].reshape(-1) for nid in node0])
        x = self.classifier(x)
        return x


def multi_gpu_node_feat(img, mask, flow, tag):
    # print(datetime.now(), "Get node feature of nuclei instance of %s slice" % tag)
    node, zflow = run_all_node(img, mask, flow) # N1 x 27
    # print(datetime.now(), "Done %s slice" % tag)
    del img, mask, flow
    return node.detach().cpu(), zflow.detach().cpu(), tag


def build_one_graph(distance, thr, indecies, dim, used, out, topn=5, multi_gpu=False, split_p=None):
    for i in indecies:
        if thr < 0: break
        if i in used[dim]: continue
        new_indecies = torch.argsort(distance.select(dim, i))[:topn]
        used[dim].append(i)
        if split_p is not None:
            out[dim].append([i+split_p] + new_indecies.tolist())
        else:
            out[dim].append([i] + new_indecies.tolist())
    if multi_gpu: del distance
    return out, used


def run_all_node(img, mask, flows, node_feat_size=200):
    '''
        return N x 27 or N x 9, N x 200
    '''
    hists = []
    nodes = []
    zflow = []
    for mi in range(1, mask.max()+1):
        m = mask==mi
        node = F.interpolate(
            img[m].unsqueeze(0).unsqueeze(0), 
            size=node_feat_size, 
            mode='linear'
        ).squeeze()#.to(device)
        zflow.append(flows[-1][m].mean())
        nodes.append(node)
        flow = torch.stack([f[m] for f in flows[:2]])
        hist = hist_flow_vector(flow)
        hists.append(hist)
    # return torch.stack(hists), torch.stack(nodes)
    return torch.cat([torch.stack(nodes), torch.stack(hists)], -1), torch.stack(zflow)

def get_distance(bbox1, bbox2):
    N1 = len(bbox1)
    N2 = len(bbox2)
    c1 = bbox1[:, :2].repeat(N2, 1, 1).permute(1, 2, 0)
    c2 = bbox2[:, :2].T.repeat(N1, 1, 1)
    assert c1.shape == c2.shape == (N1, 2, N2), (c1.shape, c2.shape)
    distance = torch.sqrt(((c1 - c2) ** 2).sum(1)) # N1 x N2
    return distance

def get_distance_low_ram(bbox1, bbox2):
    N1 = len(bbox1)
    N2 = len(bbox2)
    distance = torch.zeros(N1, 2, N2, device=bbox1.device)
    for i in range(N1):
        distance[i] = (bbox1[i, :2] - bbox2[:, :2]).T
    distance = torch.sqrt((distance ** 2).sum(1)) # N1 x N2
    return distance

def hist_flow_vector(flows):
    '''
        input Tensor 3 x N or 2 x N

        return 3*3*3=27 or 3*3=9
    '''
    dims = []
    for flow in flows:
        dims.append([flow < 0, flow == 0, flow > 0])
    hist = []
    if len(dims) == 2:
        for d1 in dims[0]:
            for d2 in dims[1]:
                hist.append((d1 & d2).sum())
    else:
        for d1 in dims[0]:
            for d2 in dims[1]:
                for d3 in dims[2]:
                    hist.append((d1 & d2 & d3).sum())
    hist = torch.stack(hist)
    return hist

