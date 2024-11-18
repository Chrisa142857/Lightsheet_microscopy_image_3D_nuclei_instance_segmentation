import numpy as np
from scipy.ndimage import median_filter, maximum_filter
import torch

def simulate_gradz(nimg, YX_flow, inside):
    return fun1(nimg, YX_flow, inside)

def fun(nimg, YX_flow, inside):
    filter_size = 12
    # filter = maximum_filter
    filter = median_filter
    grad_Z = np.zeros_like(YX_flow[0], dtype=np.float32)
    for i in range(1, nimg+1):
        if i == 1: grad_Z[i-1] = (YX_flow[:, i]**2).sum(axis=0)**0.5 - np.zeros((YX_flow.shape[-2], YX_flow.shape[-1]))
        elif i == nimg: grad_Z[i-1] = np.zeros((YX_flow.shape[-2], YX_flow.shape[-1])) - (YX_flow[:, i-1]**2).sum(axis=0)**0.5 
        else: 
            z1 = ((YX_flow[:, i]**2).sum(axis=0)**0.5 - (YX_flow[:, i-2]**2).sum(axis=0)**0.5)# / (cosine_sim+1)
            z2 = (YX_flow[:, i]**2).sum(axis=0)**0.5 - filter((YX_flow[:, i-2]**2).sum(axis=0)**0.5, size=filter_size)
            z3 = filter((YX_flow[:, i]**2).sum(axis=0)**0.5, size=filter_size) - (YX_flow[:, i-2]**2).sum(axis=0)**0.5
            grad_Z[i-1] = (z1 + z2 + z3) / 3
        grad_Z[i-1] *= inside[i-1]
    return grad_Z



def fun1(nimg, YX_flow, inside):
    stagen = 7
    filter_size = 3
    filter = median_filter
    grad_Z = np.zeros_like(YX_flow[0], dtype=np.float32)
    for i in range(1, nimg+1):
        if i == 1: grad_Z[i-1] = (YX_flow[:, i]**2).sum(axis=0)**0.5 - np.zeros((YX_flow.shape[-2], YX_flow.shape[-1]))
        elif i == nimg: grad_Z[i-1] = np.zeros((YX_flow.shape[-2], YX_flow.shape[-1])) - (YX_flow[:, i-1]**2).sum(axis=0)**0.5 
        else: 
            pre = (YX_flow[:, i-2]**2).sum(axis=0)**0.5
            next = (YX_flow[:, i]**2).sum(axis=0)**0.5
            grad_Z[i-1] += (next - pre)
            for level in range(stagen):
                pre_nextstage = filter(pre, size=filter_size)
                next_nextstage = filter(next, size=filter_size)
                grad_Z[i-1] += (next - pre_nextstage)
                grad_Z[i-1] += (next_nextstage - pre)
                pre = pre_nextstage
                next = next_nextstage

            grad_Z[i-1] /= (1 + stagen*2)
        grad_Z[i-1] *= inside[i-1]
    return grad_Z


