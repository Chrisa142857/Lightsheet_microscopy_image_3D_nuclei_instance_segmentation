import torch
import numpy as np

def eval_f1(pred_mask, gt_mask, iou_thr=0.3, is_3d=True):
    # print("len(pred mask)=%d, len(gt mask)=%d" % (len(pred_mask), len(gt_mask)))
    if len(pred_mask) != 0:
        ious = iou_2mask(pred_mask, gt_mask, is_3d=is_3d)
        ## new version of F1 compute ##############################################
        pred_tp = [0 for _ in range(len(pred_mask))]
        gt_not_hitted = torch.arange(len(gt_mask))
        for i in range(len(pred_mask)):
            ind = ious[i][gt_not_hitted].argmax()
            val = ious[i][gt_not_hitted]
            if len(val.shape) > 0:
                val = val[ind]
            if val >= iou_thr:
                pred_tp[i] = 1
                gt_not_hitted = gt_not_hitted[gt_not_hitted != gt_not_hitted[ind]]
        ## old version of F1 compute ##############################################
        # pred_tp = np.max(ious, axis=1) >= iou_thr
        # gt_not_hitted = np.max(ious, axis=0) < iou_thr
        ###########################################################################
        tp = sum(pred_tp)
        fp = len(pred_tp) - sum(pred_tp)
        # fn = sum(gt_not_hitted)
        fn = len(gt_not_hitted)
    else:
        fn = len(gt_mask)
        tp = 0
        fp = 0
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn)
    f1 = 2*(prec * rec) / (prec + rec) if prec + rec > 0 else 0
    hit_percent = tp / len(gt_mask)
    over_count = fp / len(gt_mask)
    miss_count = fn / len(gt_mask)
    return prec, rec, f1, hit_percent, np.array(pred_tp), gt_not_hitted.numpy()

def mask2mask_list(mask):
    masks = []
    for mid in np.unique(mask):
        if mid == 0: continue
        masks.append(mask==mid)
    masks = np.stack(masks)
    return masks


def extract_bboxes_torch(masks, is_3d=False, is_xywh=True):
    """Compute bounding boxes from masks.
    Input: mask List[height, width, depth]. Mask pixels are either 1 or 0.
    Returns: bbox Tensor [num_instances, (x1, y1, z1, x2, y2, z2)].
    """
    boxes = []
    for i in range(len(masks)):
        m = masks[i]
        if not m.any(): 
            boxes.append(torch.LongTensor([1e5, 1e5, 1e5, -1, -1, -1]).to(m.device) if is_3d else torch.LongTensor([1e5, 1e5, -1, -1]).to(m.device))
            continue
        if is_3d:
            x = torch.where(m.any(1).any(1))[0]
            y = torch.where(m.any(0).any(1))[0]
            z = torch.where(m.any(0).any(0))[0]
        else:
            x = torch.where(m.any(1))[0]
            y = torch.where(m.any(0))[0]
        # if is_3d:
        #     box_is_valid = len(x) > 0 and len(y) > 0 and len(z) > 0
        # else:
        #     box_is_valid = len(x) > 0 and len(y) > 0
        # if box_is_valid:
        x1, x2 = x[[0, -1]]
        y1, y2 = y[[0, -1]]
        x2 += 1
        y2 += 1
        if is_3d:
            z1, z2 = z[[0, -1]]
            z2 += 1
        if is_3d:
            if is_xywh:
                boxes.append(torch.stack([(x1+x2)/2, (y1+y2)/2, (z1+z2)/2, x2-x1, y2-y1, z2-z1]))
            else:
                boxes.append(torch.stack([x1, y1, z1, x2, y2, z2]))
        else:
            if is_xywh:
                boxes.append(torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]))
            else:
                boxes.append(torch.stack([x1, y1, x2, y2]))
    # if len(boxes) > 0:
    boxes = torch.stack(boxes)
    return boxes


def mask_iou(one_mask, masks, axis=(1,2,3)):
    if len(masks.shape) - len(one_mask.shape) == 0:
        if masks.shape[0] > one_mask.shape[0]:
            one_mask = torch.nn.functional.interpolate(one_mask.unsqueeze(0).unsqueeze(0) * 1.0, size=(masks.shape[0], masks.shape[1], masks.shape[2]), mode='nearest-exact').squeeze() > 0
        elif masks.shape[0] < one_mask.shape[0] and len(masks) > 0:
            masks = torch.nn.functional.interpolate(masks.unsqueeze(0).unsqueeze(0) * 1.0, size=(one_mask.shape[0], one_mask.shape[1], one_mask.shape[2]), mode='nearest-exact').squeeze() > 0
        return (one_mask & masks).sum() / (one_mask | masks).sum()
    elif len(masks.shape) - len(one_mask.shape) == 1:
        if masks.shape[1] > one_mask.shape[0]:
            one_mask = torch.nn.functional.interpolate(one_mask.unsqueeze(0).unsqueeze(0) * 1.0, size=(masks.shape[1], masks.shape[2], masks.shape[3]), mode='nearest-exact').squeeze() > 0
        elif masks.shape[1] < one_mask.shape[0] and len(masks) > 0:
            masks = torch.nn.functional.interpolate(masks.unsqueeze(0) * 1.0, size=(one_mask.shape[0], one_mask.shape[1], one_mask.shape[2]), mode='nearest-exact')[0] > 0
        if len(masks) == 0: return torch.zeros(len(masks)).to(masks.device)
        return (one_mask & masks).sum(axis) / (one_mask | masks).sum(axis)
    else:
        print("Error when using mask_iou", masks.shape, one_mask.shape)
        exit()
        
def iou_2mask(masks1, masks2, is_3d=True):    
    device = masks1.device
    ## compute distance between every center of node of graph
    boxes1 = extract_bboxes_torch(masks1, is_3d=is_3d) # N x 4
    boxes2 = extract_bboxes_torch(masks2, is_3d=is_3d) # N x 4
    if is_3d:
        centers1 = boxes1[:, :3]# N x 3
        centers2 = boxes2[:, :3]# N x 3
        axis = (1,2,3)
    else:
        centers1 = boxes1[:, :2]# N x 2
        centers2 = boxes2[:, :2]# N x 2
        axis = (1,2)
    N1, N2 = len(centers1), len(centers2)
    c1 = centers1.repeat(N2, 1, 1).permute(1, 2, 0) # N x 3 x (N) (repeat in last axis)
    c2 = centers2.T.repeat(N1, 1, 1) # (N) x 3 x N (repeat in first axis)
    distance = torch.sqrt(((c1 - c2) ** 2).sum(1)) # N1 x N2
    valid_ids = distance <= 30
    # pbar = tqdm(total=last_len, desc='NMS postprocessing:')
    all_iou = []
    for i in range(N1):
        valid_id = valid_ids[i]
        ious = torch.zeros(N2).float().to(device)
        ious[valid_id] = mask_iou(masks1[i], masks2[valid_id], axis=axis)
        all_iou.append(ious)
    return torch.stack(all_iou).detach().cpu().numpy()