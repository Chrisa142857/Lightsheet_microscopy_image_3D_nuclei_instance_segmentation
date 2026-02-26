import torch, tifffile, os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io
from tqdm import tqdm, trange
import scipy.ndimage
from skimage.registration import phase_cross_correlation
from skimage import img_as_ubyte
from datetime import datetime
import pandas as pd
import nibabel as nib
# from multiprocessing import Pool

class TrainvalColocDataset(Dataset):

    def __init__(self, patch_path, ann_path, classes = ['Brn-Ctip-', 'Brn+', 'Ctip+', 'Brn+Ctip+'], win_size=30, transform=None, device='cpu'):
        
        patches = tifffile.imread(patch_path)
        self.label = np.loadtxt(ann_path, delimiter=',', skiprows=1)
        patches = patches[self.label>0]
        self.label = self.label[self.label>0]
        patches = torch.from_numpy(patches).float().to(device)
        if patches.shape[1] == 3:
            patches = patches.permute(0, 2, 3, 1)
        self.classes = classes
        self.patches = patches
        self.win_size = win_size
        self.transform = transform
        
        # self.norm = transforms.Normalize((0.5,), (0.5,))
        self.norm = transforms.RandomAutocontrast(1).to(device)

    def __getitem__(self, index):
        x1 = int(self.patches.shape[1] / 2 - self.win_size / 2)
        x2 = x1 + self.win_size
        y1 = int(self.patches.shape[2] / 2 - self.win_size / 2)
        y2 = y1 + self.win_size
        patch = self.patches[index].float()
        patch = patch.permute(2, 0, 1)
        patch = [self.norm(channel[None])[0] for channel in patch]
        if self.transform is not None:
            patch = self.transform(torch.stack(patch)[None])[0][:, x1:x2, y1:y2]
        else:
            patch = torch.stack(patch)[:, x1:x2, y1:y2]
        label = int(self.label[index])-1
        # multiclass_label = torch.zeros(len(self.classes)-2)
        # if label == 1 or label == 3: multiclass_label[0] = 1 # 10 or 11
        # if label == 2 or label == 3: multiclass_label[1] = 1 # 01 or 11
        # return patch, multiclass_label
        return patch, label

    def __len__(self):
        return len(self.patches)

class TrainvalColocMultiClassDataset(Dataset):

    def __init__(self, patch_path, ann_path, classes = ['Brn-Ctip-', 'Brn+', 'Ctip+', 'Brn+Ctip+'], win_size=30, transform=None, device='cpu'):
        
        patches = tifffile.imread(patch_path)
        self.label = np.loadtxt(ann_path, delimiter=',', skiprows=1)
        patches = patches[self.label>0]
        self.label = self.label[self.label>0]
        patches = torch.from_numpy(patches).float().to(device)
        if patches.shape[1] == 3:
            patches = patches.permute(0, 2, 3, 1)
        self.classes = classes
        self.patches = patches
        self.win_size = win_size
        self.transform = transform
        
        # self.norm = transforms.Normalize((0.5,), (0.5,))
        self.norm = transforms.RandomAutocontrast(1).to(device)

    def __getitem__(self, index):
        x1 = int(self.patches.shape[1] / 2 - self.win_size / 2)
        x2 = x1 + self.win_size
        y1 = int(self.patches.shape[2] / 2 - self.win_size / 2)
        y2 = y1 + self.win_size
        patch = self.patches[index].float()
        patch = patch.permute(2, 0, 1)
        patch = [self.norm(channel[None])[0] for channel in patch]
        if self.transform is not None:
            patch = self.transform(torch.stack(patch)[None])[0][:, x1:x2, y1:y2]
        else:
            patch = torch.stack(patch)[:, x1:x2, y1:y2]
        label = int(self.label[index])-1
        multiclass_label = torch.zeros(len(self.classes)-2)
        if label == 1 or label == 3: multiclass_label[0] = 1 # 10 or 11
        if label == 2 or label == 3: multiclass_label[1] = 1 # 01 or 11
        return patch, multiclass_label
        # return patch, label

    def __len__(self):
        return len(self.patches)

class TrainvalColocBboxDataset(Dataset):

    def __init__(self, patch_path, ann_path, classes = ['Brn-Ctip-', 'Brn+', 'Ctip+', 'Brn+Ctip+'], win_size=30, transform=None, device='cpu', expand=3):
        
        patches = tifffile.imread(patch_path)
        # patch_info = patch_path.replace('patches.tif', 'patch_info.csv')
        patch_info = patch_path.replace('patches', 'patch_info').replace('.tif', '.csv')
        self.bbox_shape = np.array(pd.read_csv(patch_info))[:, [-3,-2]]
        self.expand = expand
        self.label = np.loadtxt(ann_path, delimiter=',', skiprows=1)
        patches = patches[self.label>0]
        self.label = self.label[self.label>0]
        patches = torch.from_numpy(patches).float().to(device)
        if patches.shape[1] == 3:
            patches = patches.permute(0, 2, 3, 1)
        self.classes = classes
        self.patches = patches
        self.win_size = win_size
        self.transform = transform
        
        # self.norm = transforms.Normalize((0.5,), (0.5,))
        self.norm = transforms.RandomAutocontrast(1).to(device)
        self.resize = transforms.Resize([win_size, win_size])

    def __getitem__(self, index):
        patch = self.patches[index].float()
        bboxw, bboxh = self.bbox_shape[index]
        bboxw, bboxh = bboxw + self.expand, bboxh + self.expand
        x1 = int(self.patches.shape[1] / 2 - bboxw / 2)
        x2 = x1 + bboxw
        y1 = int(self.patches.shape[2] / 2 - bboxh / 2)
        y2 = y1 + bboxh
        patch = patch.permute(2, 0, 1)
        patch = [self.norm(channel[None])[0] for channel in patch]
        if self.transform is not None:
            patch = self.transform(torch.stack(patch)[None])[0][:, x1:x2, y1:y2]
        else:
            patch = torch.stack(patch)[:, x1:x2, y1:y2]
        label = int(self.label[index])-1
        patch = self.resize(patch[None])[0]
        # multiclass_label = torch.zeros(len(self.classes)-2)
        # if label == 1 or label == 3: multiclass_label[0] = 1 # 10 or 11
        # if label == 2 or label == 3: multiclass_label[1] = 1 # 01 or 11
        # return patch, multiclass_label
        return patch, label

    def __len__(self):
        return len(self.patches)

class TrainvalColocMultiClassBboxDataset(Dataset):

    def __init__(self, patch_path, ann_path, patch_info=None, classes = ['Brn-Ctip-', 'Brn+', 'Ctip+', 'Brn+Ctip+'], win_size=30, transform=None, device='cpu', expand=3):
        
        patches = tifffile.imread(patch_path)
        # patch_info = patch_path.replace('patches.tif', 'patch_info.csv') if patch_info is None else patch_info
        patch_info = patch_path.replace('patches', 'patch_info').replace('.tif', '.csv') if patch_info is None else patch_info
        self.bbox_shape = np.array(pd.read_csv(patch_info))[:, [-3,-2]]
        self.bbox_z = np.array(pd.read_csv(patch_info))[:, [-8]]
        self.expand = expand
        self.label = np.loadtxt(ann_path, delimiter=',', skiprows=1)
        patches = patches[self.label>0]
        self.bbox_shape = self.bbox_shape[self.label>0]
        self.bbox_z = self.bbox_z[self.label>0]
        self.label = self.label[self.label>0]
        patches = torch.from_numpy(patches).float().to(device)
        if patches.shape[1] == 3:
            patches = patches.permute(0, 2, 3, 1)
        self.classes = classes
        self.patches = patches
        self.win_size = win_size
        self.transform = transform
        
        # self.norm = transforms.Normalize((0.5,), (0.5,))
        self.norm = transforms.RandomAutocontrast(1).to(device)
        self.resize = transforms.Resize([win_size, win_size])
        self.current_bboxz = self.bbox_z[0]

    def __getitem__(self, index):
        patch = self.patches[index].float()
        bboxz = self.bbox_z[index]
        self.current_bboxz = bboxz
        bboxw, bboxh = self.bbox_shape[index]
        bboxw, bboxh = bboxw + self.expand, bboxh + self.expand
        x1 = int(self.patches.shape[1] / 2 - bboxw / 2)
        x2 = x1 + bboxw
        y1 = int(self.patches.shape[2] / 2 - bboxh / 2)
        y2 = y1 + bboxh
        patch = patch.permute(2, 0, 1)
        patch = [self.norm(channel[None])[0] for channel in patch]
        if self.transform is not None:
            patch = self.transform(torch.stack(patch)[None])[0][:, x1:x2, y1:y2]
        else:
            patch = torch.stack(patch)[:, x1:x2, y1:y2]
        label = int(self.label[index])-1
        patch = self.resize(patch[None])[0]
        multiclass_label = torch.zeros(len(self.classes)-2)
        if label == 1 or label == 3: multiclass_label[0] = 1 # 10 or 11
        if label == 2 or label == 3: multiclass_label[1] = 1 # 01 or 11
        return patch, multiclass_label

    def __len__(self):
        return len(self.patches)

class TrainvalColocBboxLocDataset(Dataset):

    def __init__(self, patch_path, ann_path, classes = ['Brn-Ctip-', 'Brn+', 'Ctip+', 'Brn+Ctip+'], win_size=30, transform=None, device='cpu', expand=3):
        
        patches = tifffile.imread(patch_path)
        patch_info = patch_path.replace('patches', 'patch_info').replace('.tif', '.csv')
        self.bbox_shape = np.array(pd.read_csv(patch_info))[:, [-3,-2, 1, 2, 3]]
        self.expand = expand
        self.label = np.loadtxt(ann_path, delimiter=',', skiprows=1)
        patches = patches[self.label>0]
        self.label = self.label[self.label>0]
        patches = torch.from_numpy(patches).float().to(device)
        if patches.shape[1] == 3:
            patches = patches.permute(0, 2, 3, 1)
        self.classes = classes
        self.patches = patches
        self.win_size = win_size
        self.transform = transform
        
        # self.norm = transforms.Normalize((0.5,), (0.5,))
        self.norm = transforms.RandomAutocontrast(1).to(device)
        self.resize = transforms.Resize([win_size, win_size])
        
        self.maxx, self.maxy, self.maxz = self.bbox_shape[:, -3:].max(0)

    def __getitem__(self, index):
        patch = self.patches[index].float()
        bboxw, bboxh, bboxx, bboxy, bboxz = self.bbox_shape[index]
        bboxw, bboxh = bboxw + self.expand, bboxh + self.expand
        x1 = int(self.patches.shape[1] / 2 - bboxw / 2)
        x2 = x1 + bboxw
        y1 = int(self.patches.shape[2] / 2 - bboxh / 2)
        y2 = y1 + bboxh
        patch = patch.permute(2, 0, 1)
        patch = [self.norm(channel[None])[0] for channel in patch]
        if self.transform is not None:
            patch = self.transform(torch.stack(patch)[None])[0][:, x1:x2, y1:y2]
        else:
            patch = torch.stack(patch)[:, x1:x2, y1:y2]
        label = int(self.label[index])-1
        patch = self.resize(patch[None])[0]
        # multiclass_label = torch.zeros(len(self.classes)-2)
        # if label == 1 or label == 3: multiclass_label[0] = 1 # 10 or 11
        # if label == 2 or label == 3: multiclass_label[1] = 1 # 01 or 11
        # return patch, multiclass_label
        # return patch, label, torch.FloatTensor([bboxx/self.maxx, bboxy/self.maxy, bboxz.item()/self.maxz]) #torch.LongTensor([bboxx, bboxy, bboxz])
        return patch, label, torch.LongTensor([bboxx, bboxy, bboxz])

    def __len__(self):
        return len(self.patches)

class TrainvalColocMultiClassBboxLocDataset(Dataset):

    def __init__(self, patch_path, ann_path, patch_info=None, classes = ['Brn-Ctip-', 'Brn+', 'Ctip+', 'Brn+Ctip+'], win_size=30, transform=None, device='cpu', expand=3):
        
        patches = tifffile.imread(patch_path)
        patch_info = patch_path.replace('patches', 'patch_info').replace('.tif', '.csv') if patch_info is None else patch_info
        self.bbox_shape = np.array(pd.read_csv(patch_info))[:, [-3,-2, 1, 2]]
        self.bbox_z = np.array(pd.read_csv(patch_info))[:, [-8]]
        self.expand = expand
        self.label = np.loadtxt(ann_path, delimiter=',', skiprows=1)
        patches = patches[self.label>0]
        self.bbox_shape = self.bbox_shape[self.label>0]
        self.bbox_z = self.bbox_z[self.label>0]
        self.label = self.label[self.label>0]
        patches = torch.from_numpy(patches).float().to(device)
        if patches.shape[1] == 3:
            patches = patches.permute(0, 2, 3, 1)
        self.classes = classes
        self.patches = patches
        self.win_size = win_size
        self.transform = transform
        
        # self.norm = transforms.Normalize((0.5,), (0.5,))
        self.norm = transforms.RandomAutocontrast(1).to(device)
        self.resize = transforms.Resize([win_size, win_size])
        self.current_bboxz = self.bbox_z[0]

        self.maxx, self.maxy = self.bbox_shape[:, -2:].max(0)
        self.maxz = self.bbox_z.max()

    def __getitem__(self, index):
        patch = self.patches[index].float()
        bboxz = self.bbox_z[index]
        self.current_bboxz = bboxz
        bboxw, bboxh, bboxx, bboxy = self.bbox_shape[index]
        bboxw, bboxh = bboxw + self.expand, bboxh + self.expand
        x1 = int(self.patches.shape[1] / 2 - bboxw / 2)
        x2 = x1 + bboxw
        y1 = int(self.patches.shape[2] / 2 - bboxh / 2)
        y2 = y1 + bboxh
        patch = patch.permute(2, 0, 1)
        patch = [self.norm(channel[None])[0] for channel in patch]
        if self.transform is not None:
            patch = self.transform(torch.stack(patch)[None])[0][:, x1:x2, y1:y2]
        else:
            patch = torch.stack(patch)[:, x1:x2, y1:y2]
        label = int(self.label[index])-1
        patch = self.resize(patch[None])[0]
        multiclass_label = torch.zeros(len(self.classes)-2)
        if label == 1 or label == 3: multiclass_label[0] = 1 # 10 or 11
        if label == 2 or label == 3: multiclass_label[1] = 1 # 01 or 11
        # return patch, multiclass_label, torch.FloatTensor([bboxx/self.maxx, bboxy/self.maxy, bboxz.item()/self.maxz]) #torch.LongTensor([bboxx, bboxy, bboxz.item()])
        return patch, multiclass_label, torch.LongTensor([bboxx, bboxy, bboxz.item()])

    def __len__(self):
        return len(self.patches)

class InferColocDataset(Dataset):

    def __init__(self, ls_root, centeroid_path, zrange=[0, -1], win_size=30, norm_size=100, transform=None, channel_sort = ['topro', 'brn2', 'ctip2']):
        fn_split = '_'
        z_spliti = 1
        ch_spliti = 3
        self.channel_sort = channel_sort
        ct = scipy.io.loadmat(centeroid_path)['coordinates']
        print(ct.max(0), ct.min(0))
        # exit()
        ctx, cty, ctz = ct[:, 0], ct[:, 1], ct[:, 2]
        zrange = [zr if zr >= 0 else ctz.max()+zr+1 for zr in zrange]
        self.zrange = zrange
        ct_idx = np.where(np.logical_and(ctz >= zrange[0], ctz <= zrange[1]))[0]
        print(f'center z range: [{ctz.min()}, {ctz.max()}]')
        assert len(ct_idx) > 0, f'CHANGE --zmin or --zmax: zrange {zrange} leads to 0 ct, where ctz range: [{ctz.min()}, {ctz.max()}]'

        ls_paths = os.listdir(ls_root)
        ls_paths = np.array([f'{ls_root}/{fn}' for fn in ls_paths if fn.endswith('.tif')])
        ls_fn_z = np.array([int(ls_path.split('/')[-1].split(fn_split)[z_spliti])-1 for ls_path in ls_paths])
        # tmp_img = tifffile.imread(ls_paths[0])
        self.ls_paths = ls_paths
        self.ls_fn_z = ls_fn_z
        self.fn_split = fn_split
        self.z_spliti = z_spliti
        self.ch_spliti = ch_spliti
        self.load_sort = ct_idx[np.argsort(ctz[ct_idx])]
        self.ctx = ctx
        self.cty = cty
        self.ctz = ctz
        self.cached_z = self.ctz[self.load_sort[0]]
        self.load_image(self.cached_z)
        self.classes = ['Brn-Ctip-', 'Brn+', 'Ctip+', 'Brn+Ctip+']
        
        self.norm_size = norm_size
        self.win_size = win_size
        self.transform = transform
        # self.norm = transforms.Normalize((0.5,), (0.5,))
        self.norm = transforms.RandomAutocontrast(1)

    def load_image(self, z):
        assert len(np.where(self.ls_fn_z==z)[0]) == 3, f'len({self.ls_paths[self.ls_fn_z==z]})!=3'
        ls_load_path = {p.split('/')[-1].split(self.fn_split)[self.ch_spliti]: p for p in self.ls_paths[self.ls_fn_z==z]}
        img = [tifffile.imread(ls_load_path[ch])
                                for ch in self.channel_sort]
        ref_img = img[0]
        for channeli in range(1, len(img)):
            down_time = 3
            mov_img = img[channeli]
            shift, error, diffphase = phase_cross_correlation(ref_img[::down_time,::down_time], mov_img[::down_time,::down_time])
            shift = shift * down_time
            img[channeli] = scipy.ndimage.shift(mov_img, shift)

        self.cached_ls_image = torch.from_numpy(np.stack(img, 0)).float()

    def __getitem__(self, index):
        center_index = self.load_sort[index]
        z = self.ctz[center_index]
        if z != self.cached_z:
            self.cached_z = z
            self.load_image(self.cached_z)
        
        x = self.ctx[center_index]
        y = self.cty[center_index]
        half_norm_size = self.norm_size // 2
        x1 = x - half_norm_size if x >= half_norm_size else 0
        x2 = x1 + self.norm_size
        y1 = y - half_norm_size if y >= half_norm_size else 0
        y2 = y1 + self.norm_size
        patch = self.cached_ls_image[:, x1:x2, y1:y2]
        patch = torch.stack([self.norm(patch[i][None])[0] for i in range(len(self.channel_sort))])
        
        x1 = int(patch.shape[1] / 2 - self.win_size / 2)
        x2 = x1 + self.win_size
        y1 = int(patch.shape[2] / 2 - self.win_size / 2)
        y2 = y1 + self.win_size
        patch = patch[:, x1:x2, y1:y2]

        return patch, center_index, x, y, z

    def __len__(self):
        return len(self.load_sort)

from convert_p4_tileNIS2numorph import parse_stitch_tform

# class InferNisLocPatchColocDataset(InferNisPatchColocDataset):


class InferNisPatchColocDataset(Dataset):

    def __init__(self, 
            gtag = 'P4',
            ptag = 'pair11',
            btag = 'L66D764P8', 
            device = 'cuda:0',
            use_bbox = False,
            bbox_expand = 3,
            zrange=[0, -1], patch_sz=101, win_size=30):
        self.use_bbox = use_bbox
        self.bbox_expand = bbox_expand
        self.win_size = win_size
        if use_bbox:
            self.resize = transforms.Resize([win_size+bbox_expand-3, win_size+bbox_expand-3])
        self.device = device
        self.patch_sz = patch_sz
        self.zrange = zrange
        stitched_info, _, _ = parse_stitch_tform(ptag, gtag, btag)
        # doubled_label = torch.load(f'/export_home/ziquanw/tmp/doubled_NIS_label/{btag}_doubled_label_byNapari.zip', map_location=device)
        # doubled_dict_ktag = 'UltraII[%02d x %02d]'
        channel_sort = ['C01', 'C00', 'C02']
        pair_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/{gtag}/{ptag}'
        for dir in os.listdir(pair_path):
            if len(dir.split('_')) < 2 or '.' in dir: continue
            if btag == dir.split('_')[1]: 
                result_path = f'{pair_path}/{dir}'
                break
        tile_tag = 'UltraII[%02d x %02d]'
        result_root = result_path + '/' + tile_tag
        tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path) if 'UltraII' in fn])
        ncol, nrow = tile_loc.max(0)+1
        raw_img_root = result_root.replace('results', 'image_before_stitch_all_channel')
        stack_names = None
        for i in range(ncol):
            for j in range(nrow):
                if stack_names is None:
                    stack_names = [f for f in os.listdir(result_root % (i, j)) if f.endswith('instance_bbox.zip')]
                
                pstack_names = [f for f in os.listdir(result_root % (i, j)) if f.endswith('instance_bbox.zip')]
                if len(pstack_names) > len(stack_names): stack_names = pstack_names
        stack_names = sort_stackname(stack_names)
        meta_name = stack_names[0].replace('instance_bbox', 'seg_meta')
        tile_shape = torch.load(f'{result_root % (0, 0)}/{meta_name}')
        tile_shape = [s.item() for s in tile_shape]
        # lrange = 1000
        patch_load_urls = []
        patch_load_x1x2y1y2 = []
        patch_info = []
        patch_load_index = []
        self.patch_info_header = ['NIS_ID', 'coordinate_x', 'coordinate_y', 'coordinate_z', 'nuclei_w', 'nuclei_h', 'nuclei_d', 'NIS_inside_tile', 'classification']
        center_num = 0
        # pbar = tqdm(total=ncol*nrow, desc=f'{datetime.now()} Patch info preparing')
        zmin, zmax = zrange
        for i in range(ncol):
            for j in range(nrow):
                # pbar.update(1)
                topleft_loc = {
                    stitched_info['z'][stitchi]: [stitched_info['x1'][stitchi], stitched_info['y1'][stitchi]] 
                    for stitchi in range(len(stitched_info['z'])) 
                    if stitched_info['ijkey'][stitchi] == tile_tag % (i, j)
                }
                centers = []
                nis_labels = []
                for stack_name in stack_names:
                    ct_path = f"{result_root % (i, j)}/{stack_name}"
                    if not os.path.exists(ct_path): continue
                    tile_shape = torch.load(f"{result_root % (i, j)}/{stack_name.replace('instance_bbox', 'seg_meta')}")
                    stack_z = int(stack_name.split('zmin')[1].split('_')[0])
                    ct = torch.load(ct_path, map_location=device).float()#.to(device)
                    ct[:, ::3] = (ct[:, ::3] + stack_z) * (2.5/4)
                    if zrange[1] == -1:
                        zmax = max(ct[:, 3].max() + 1, zmax)
                    centers.append(ct)
                    nis_label = torch.load(f"{result_root % (i, j)}/{stack_name.replace('instance_bbox', f'instance_label')}", map_location=device)
                    nis_labels.append(nis_label)
                if len(centers) == 0: continue
                centers = torch.cat(centers).round().long()
                nis_labels = torch.cat(nis_labels)
                ## Z stitch #############################
                if os.path.exists(f"{result_root % (i, j)}/{btag}_remap.zip"):
                    zstitch_remap = torch.load(f"{result_root % (i, j)}/{btag}_remap.zip", map_location=device)
                    (centers,), nis_labels = do_zstitch(zstitch_remap, [centers], nis_labels, device=device)
                ## Stitch undouble ######################
                # keep_ind = []
                # for labeli in range(0, len(nis_labels), lrange):
                #     label_batch = nis_labels[labeli:labeli+lrange]#.to(device)
                #     label2rm = label_batch[:, None] == doubled_label[doubled_dict_ktag % (i, j)][None, :]
                #     do_rm = label2rm.any(1)
                #     keep_ind.append(torch.arange(labeli, labeli+len(label_batch), device=device)[torch.logical_not(do_rm)])
                # keep_ind = torch.cat(keep_ind)
                # if len(nis_labels) > 0:
                #     centers = centers[keep_ind]
                #     nis_labels = nis_labels[keep_ind]
                ##########################################
                imgfns = os.listdir(raw_img_root % (i, j))
                imgfns = np.array(imgfns)
                # imgfns = np.array([fn for fn in imgfns if btag in fn])
                imgz = np.array([int(fn.split(' Z')[-1][:4]) for fn in imgfns])
                imgchs = np.array([fn.split('_')[-2] for fn in imgfns])
                info = torch.zeros([len(centers), len(channel_sort) * 2 + 2], dtype=centers.dtype, device=device)#.to(device)

                middlemask = torch.logical_and(
                    torch.logical_and(centers[:, 1] >= patch_sz//2, centers[:, 2] >= patch_sz//2),
                    torch.logical_and(centers[:, 4] <= tile_shape[1] - patch_sz//2, centers[:, 5] <= tile_shape[2] - patch_sz//2)
                ).to(info.dtype)

                ct_allz = (centers[:, 0] + centers[:, 3]) // 2
                print(datetime.now(), f'Patch info preparing: tile {i} {j}', 'imgz.min(), imgz.max()', imgz.min(), imgz.max(), ct_allz.min().item(), ct_allz.max().item())
                ct_zsort_idx = []
                patch_urls = pd.DataFrame({ch: np.full((len(centers), ), np.nan) for ch in channel_sort})
                patch_x1x2y1y2 = torch.zeros([len(centers), 4], dtype=centers.dtype, device=device)
                for ctz in ct_allz.unique():
                    if ctz < zmin or ctz >= zmax: continue
                    # ctzmask = torch.where(torch.logical_and(ct_allz==ctz, middlemask))[0].tolist()
                    ctzmask = torch.where(ct_allz==ctz)[0].tolist()
                    info[ctzmask, 6] = middlemask[ctzmask]
                    ct_zsort_idx.extend(ctzmask)

                    bz1, bx1, by1, bz2, bx2, by2 = centers[ctzmask].T.long()
                    x = (bx1 + bx2) // 2
                    y = (by1 + by2) // 2
                    ### Global NIS center
                    if int(ctz) in topleft_loc:
                        tx, ty = topleft_loc[int(ctz)]
                    else: 
                        tx, ty = 0, 0
                    info[ctzmask, 0] = x + tx
                    info[ctzmask, 1] = y + ty
                    info[ctzmask, 2] = ctz
                    ###########
                    depth = bz2 - bz1
                    w = bx2 - bx1
                    h = by2 - by1
                    info[ctzmask, 3] = w
                    info[ctzmask, 4] = h
                    info[ctzmask, 5] = depth
                    imgfn = imgfns[imgz == ctz.cpu().item()]
                    imgch = imgchs[imgz == ctz.cpu().item()]
                    for ch in channel_sort:
                        fn = imgfn[imgch==ch]
                        assert len(fn) == 1, f"{imgfns}, {ch}, {imgz.min()}, {imgz.max()}, {ctz.cpu().item()}"
                        patch_urls[ch].iloc[ctzmask] = f'{raw_img_root % (i, j)}/{fn[0]}'
                    patch_x1x2y1y2[ctzmask, 0] = x - patch_sz // 2
                    patch_x1x2y1y2[ctzmask, 1] = x - patch_sz // 2 + patch_sz
                    patch_x1x2y1y2[ctzmask, 2] = y - patch_sz // 2
                    patch_x1x2y1y2[ctzmask, 3] = y - patch_sz // 2 + patch_sz

                info = torch.cat([nis_labels[:, None], info], -1)
                patch_info.append(info.cpu())
                patch_load_urls.append(patch_urls)
                patch_load_x1x2y1y2.append(patch_x1x2y1y2.cpu())
                patch_load_index.append(torch.LongTensor(ct_zsort_idx)+center_num)
                center_num += len(centers)

        self.patch_load_index = torch.cat(patch_load_index)
        self.patch_load_urls = np.array(pd.concat(patch_load_urls))
        self.patch_load_x1x2y1y2 = torch.cat(patch_load_x1x2y1y2)
        self.patch_info = torch.cat(patch_info)
        ## Within atlas #############################################################################
        ctx, cty, ctz = self.patch_info[:, 1:4].T
        annotations = get_annotations(ptag, btag, torch.stack([ctz, ctx, cty], -1))
        filtered_nis = torch.where(annotations>0)[0]
        self.patch_load_index = self.patch_load_index[torch.isin(self.patch_load_index, filtered_nis)]
        #############################################################################################
        _, uni_load_url_index = np.unique(self.patch_load_urls[self.patch_load_index, 0], return_index=True)
        uni_load_url = self.patch_load_urls[self.patch_load_index][uni_load_url_index]
        image_list = []
        for url in tqdm(uni_load_url, desc=f'{datetime.now()} Preload tile images'):
            image_list.append(self.load_image(url))

        self.cached_image = {}

        channel_align_path = f'/scheibel/ACMUSERS/ziquanw/Lightsheet/coloc_channel_align/{gtag}/{ptag}_{btag}_{zrange[0]:04d}-{zrange[1]:04d}_channel_shift.csv'
        # if os.path.exists(channel_align_path):
        if False:
            channel_align_df = pd.read_csv(channel_align_path)
            channel_align_df.index = channel_align_df['index']
            channel_align_df = channel_align_df.drop('index', axis=1)
        else:
            channel_align_df = {'index': [uni_load_url[i][0] for i in range(len(uni_load_url))]}
            for ch in channel_sort[1:]:
                channel_align_df[f'{ch}_shift_x'] = [None for i in range(len(uni_load_url))]
                channel_align_df[f'{ch}_shift_y'] = [None for i in range(len(uni_load_url))]
            channel_align_df = pd.DataFrame(channel_align_df)
            channel_align_df.index = channel_align_df['index']
            channel_align_df = channel_align_df.drop('index', axis=1)
        for i in trange(len(image_list), desc=f'{datetime.now()} Preproc tile images'):
            img_id = uni_load_url[i][0]
            if os.path.exists(channel_align_path):
                try:
                    shifts = [channel_align_df.loc[img_id][[f'{ch}_shift_x', f'{ch}_shift_y']].tolist() for ch in channel_sort[1:]]
                except:
                    shifts = [None for ch in channel_sort[1:]]
            else:
                shifts = [None for ch in channel_sort[1:]]
            image, shifts = self.preprocess_image(image_list[i], shifts=shifts)
            self.cached_image[img_id] = image
            if not os.path.exists(channel_align_path):
                for chi, ch in enumerate(channel_sort[1:]):
                    channel_align_df.loc[img_id][[f'{ch}_shift_x', f'{ch}_shift_y']] = shifts[chi]

        # if not os.path.exists(channel_align_path):
        #     channel_align_df.to_csv(channel_align_path)

        self.classes = ['Brn-Ctip-', 'Brn+', 'Ctip+', 'Brn+Ctip+'] if gtag == 'P4' else ['Sox9-Neun-', 'Sox9+', 'Neun+', 'Sox9+Neun+']
        self.norm = transforms.RandomAutocontrast(1)

    def load_image(self, fnlist):
        return [tifffile.imread(fn) for fn in fnlist]

    def preprocess_image(self, img, shifts=[None, None]):
        down_time = 2
        # img = [tifffile.imread(fn) for fn in fnlist]
        ref_img = img[0]
        all_shift = []
        for channeli in range(1, len(img)):
            mov_img = img[channeli]
            # if shifts[channeli-1] is None:
            #     shift, error, diffphase = phase_cross_correlation(ref_img[::down_time,::down_time], mov_img[::down_time,::down_time])
            #     shift = (shift * down_time).astype(int)
            #     all_shift.append(shift)
            # else:
            #     shift = shifts[channeli-1]
            shift, error, diffphase = phase_cross_correlation(ref_img[::down_time,::down_time], mov_img[::down_time,::down_time])
            shift = (shift * down_time).astype(int)
            all_shift.append(shift)
                
            # img[channeli] = scipy.ndimage.shift(mov_img, shift)
            if shift[0] != 0 or shift[1] != 0:
                img[channeli] = shift_image(mov_img, shift[0], shift[1])
        img = np.stack(img, -1)
        img = np.pad(img, ((self.patch_sz, self.patch_sz), (self.patch_sz, self.patch_sz), (0, 0)))
        # self.cached_ls_image = img # torch.from_numpy(img).float()
        return img, all_shift


    def adjust_intensity(self, patch):
        """
        Adjust intensity for each patch in the list `patches`, analogous to MATLAB imadjustn.
        """
        patch = patch.astype(np.float64)
        # compute percentiles
        p_low = np.percentile(patch, 0.02) / 65535.0
        p_high = np.percentile(patch, 99.98) / 65535.0
        
        # normalize between 0–1
        low, high = p_low, p_high
        patch_norm = np.clip((patch/65535.0 - low) / (high - low), 0, 1)
        
        # convert to uint8
        patch_uint8 = img_as_ubyte(patch_norm)
        return patch_uint8
    
    def __getitem__(self, index):
        center_index = self.patch_load_index[index]
        patch_load_url = self.patch_load_urls[center_index]
        ls_image = self.cached_image[patch_load_url[0]]
        x1, x2, y1, y2 = self.patch_load_x1x2y1y2[center_index]
        patch = ls_image[x1+self.patch_sz:x2+self.patch_sz, y1+self.patch_sz:y2+self.patch_sz, :].copy()
        patch[..., 0] = self.adjust_intensity(patch[..., 0])
        patch[..., 1] = self.adjust_intensity(patch[..., 1])
        patch[..., 2] = self.adjust_intensity(patch[..., 2])  
        patch = torch.from_numpy(patch.astype(np.float32)).float()
        return patch, center_index

    # def collate_fn(self,  patches, center_indexes):
    #     output = []
    #     for patch, center_index in zip(patches, center_indexes):
    def collate_fn(self, patches, center_indexes):
        output = []
        bsz, ph, pw, chnum = patches.shape
        patches = patches.to(self.device) # B x H x W x 3
        patches = self.norm(patches.permute(0, 3, 1, 2).reshape(bsz*chnum, ph, pw)[:, None])[:, 0].reshape(bsz, chnum, ph, pw)
        # patch = torch.stack([self.norm(patch[..., i][:, None])[0] for i in range(3)]) 
        for patch, center_index in zip(patches, center_indexes):
            if not self.use_bbox:
                inputw, inputh = self.win_size, self.win_size
            else:
                inputw, inputh = self.patch_info[center_index, 4:6].T
                inputw, inputh = inputw + self.bbox_expand, inputh + self.bbox_expand
            x1 = max(0, int(patch.shape[1] / 2 - inputw / 2))
            x2 = x1 + inputw
            y1 = max(0, int(patch.shape[2] / 2 - inputh / 2))
            y2 = y1 + inputh
            patch = patch[..., x1:x2, y1:y2]
            if self.use_bbox:
                patch = self.resize(patch[None])[0]
            output.append(patch)

        return torch.stack(output)
    
    def __len__(self):
        return len(self.patch_load_index)

def shift_image(image, dx, dy):
    """
    Shift a 2D image by dx pixels horizontally and dy pixels vertically.
    
    Parameters:
    image: 2D numpy array representing the image
    dx: horizontal shift (positive = right, negative = left)
    dy: vertical shift (positive = down, negative = up)
    
    Returns:
    shifted_image: 2D numpy array of the shifted image with zero padding
    """
    # Use np.roll for fast shifting
    shifted = np.roll(image, dy, axis=0)  # shift vertically
    shifted = np.roll(shifted, dx, axis=1)  # shift horizontally
    
    # Zero out the wrapped regions
    if dy > 0:
        shifted[:dy, :] = 0
    elif dy < 0:
        shifted[dy:, :] = 0
        
    if dx > 0:
        shifted[:, :dx] = 0
    elif dx < 0:
        shifted[:, dx:] = 0
    
    return shifted

def get_annotations(ptag, btag, center):
    ann_fn = f'/lichtman/Felix/Lightsheet/P4/{ptag}/output_{btag}/registered/{btag}_MASK_topro_25_all.nii'
    if not os.path.exists(ann_fn): return torch.ones(len(center))
    brain_mask_downr = torch.FloatTensor([4/25, 0.75/25, 0.75/25])#.to(device)
    brain_mask1 = interested_brain_mask = torch.from_numpy(np.array(nib.load(ann_fn).get_fdata()).transpose(2,0,1).copy())
    down_center = center * brain_mask_downr[None]
    down_center = down_center.clip(max=torch.FloatTensor(list(brain_mask1.shape))[None]-1, min=torch.zeros(1,3)).long()
    annotations = interested_brain_mask[down_center[:, 0], down_center[:, 1], down_center[:, 2]]
    return annotations

def sort_stackname(stack_names):
    stack_z = []
    for stack_name in stack_names:
        stack_z.append(int(stack_name.split('zmin')[1].split('_')[0]))

    argsort = np.argsort(stack_z)
    return [stack_names[i] for i in argsort]

def do_zstitch(zstitch_remap, items, label, zstitch_fn=['avg'], device='cuda:0'):
    lrange = 1000
    ## loc: gnn stitch source (current tile) nis index, stitch_remap_loc: index of pairs in the stitch remap list
    loc, stitch_remap_loc = [], []
    for lrangei in range(0, len(label), lrange):
        lo, stitch_remap_lo = torch.where(label[lrangei:lrangei+lrange, None] == zstitch_remap[0, None, :])
        loc.append(lo+lrangei)
        stitch_remap_loc.append(stitch_remap_lo)
    if len(loc) == 0: return items, label
    loc, stitch_remap_loc = torch.cat(loc), torch.cat(stitch_remap_loc)

    ## pre_loc: gnn stitch target (previous tile) nis index, tloc: index of remaining Z stitch pairs after nis being removed by X-Y stitching
    pre_loc, tloc = [], []
    for lrangei in range(0, len(label), lrange):
        pre_lo, tlo = torch.where(label[lrangei:lrangei+lrange, None] == zstitch_remap[1, None, stitch_remap_loc])
        pre_loc.append(pre_lo+lrangei)
        tloc.append(tlo)
    pre_loc, tloc = torch.cat(pre_loc), torch.cat(tloc)
    ## source nis is removed from keeping mask
    keep_mask = torch.ones(len(items[0]), device=device).bool()
    keep_mask[loc] = False
#     keep_masks[stack_name][f'{i}-{j}'] = torch.logical_and(keep_masks[stack_name][f'{i}-{j}'], keep_mask)

    # merge stitched source nis to target nis
    loc = loc[tloc]
    for i, item in enumerate(items):
        if zstitch_fn[i] == 'avg':
            item[pre_loc] = (item[loc] + item[pre_loc]) // 2
        elif zstitch_fn[i] == 'sum':
            item[pre_loc] = item[loc] + item[pre_loc]
        else:
            raise NotImplementedError(zstitch_fn[i])
        
        items[i] = item[keep_mask]
        
    label = label[keep_mask]
    
    # if feat is not None:
    #     # feat = feat[loc] + feat[pre_loc]
    #     feat = feat[keep_mask]
    return items, label

if __name__ == '__main__':
    # dset = TrainvalColocDataset('/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/data/pair3/output_L35D719P4/classifier/L35D719P4_patches.tif', '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/data/pair3/output_L35D719P4/classifier/L35D719P4_classifications.csv')    
    dset = InferColocDataset('/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/data/pair3/output_L35D719P4/stitched', '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/data/pair3/output_L35D719P4/variables/centroids.mat') 
    i = 0
    for data in tqdm(dset):
        data, label = data
        print(data.shape, data.max(), data.min(), label)
        i += 1
        if i >= 10: break