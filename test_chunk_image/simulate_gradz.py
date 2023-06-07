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
        if i == 1: grad_Z[i-1] = (YX_flow[:, i]**2).sum(axis=0)**0.5 - np.zeros((128, 128))
        elif i == nimg: grad_Z[i-1] = np.zeros((128, 128)) - (YX_flow[:, i-1]**2).sum(axis=0)**0.5 
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
    # grad_Z = np.zeros_like(YX_flow[0], dtype=np.float32)
    grad_Z = []
    for i in range(1, nimg+1):
        if i == 1: g = (YX_flow[:, i]**2).sum(axis=0)**0.5
        elif i == nimg: g = -(YX_flow[:, i-1]**2).sum(axis=0)**0.5 
        else: 
            s,e = max(0, i-2), min(i+2, nimg)
            chunk = YX_flow[:, s:e]
            inside_chunks = inside[s:e]
            # print(chunk.shape)
            chunks = torch.nn.functional.interpolate(torch.from_numpy(chunk).unsqueeze(0), scale_factor=(4/2.5, 1, 1), mode='nearest-exact').squeeze().numpy()
            inside_chunks = torch.nn.functional.interpolate(torch.from_numpy(inside_chunks).unsqueeze(0).unsqueeze(0) * 1.0, scale_factor=(4/2.5, 1, 1), mode='nearest-exact').squeeze().numpy() > 0
            # print(chunk.shape)
            for j in range(1, (chunks.shape[1]-2)):
                chunk = chunks[:, j:j+3]
                pre = (chunk[:, 0]**2).sum(axis=0)**0.5
                next = (chunk[:, 2]**2).sum(axis=0)**0.5
                # grad_Z[i-1] += (next - pre)
                g = next - pre
                for level in range(stagen):
                    pre_nextstage = filter(pre, size=filter_size)
                    next_nextstage = filter(next, size=filter_size)
                    g += (next - pre_nextstage)
                    g += (next_nextstage - pre)
                    pre = pre_nextstage
                    next = next_nextstage

                g /= (1 + stagen*2)
                g *= inside_chunks[j+1]
                grad_Z.append(np.stack((g, chunk[0, 1], chunk[1, 1], inside_chunks[j+1])))
    grad_Z = np.stack(grad_Z, axis=1)
    print(grad_Z.shape)
    return grad_Z

