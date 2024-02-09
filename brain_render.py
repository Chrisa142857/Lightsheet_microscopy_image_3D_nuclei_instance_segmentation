import vedo
from vedo import Plotter, utils  # <- this will be used to render an embedded scene 
# from itkwidgets import view
# from imio import load
# from vedo import Volume, Points, VolumeSlice, Text2D, probe_points, Axes
from vedo.applications import RayCastPlotter, IsosurfaceBrowser
from vedo.pyplot import histogram
import os, re
import pandas
import torch
import random
import nibabel as nib
import numpy as np
import scipy
import h5py
from tqdm import tqdm, trange
from datetime import datetime


def load_atlas(mask_fn):
    orig_roi_mask = torch.from_numpy(np.transpose(nib.load(mask_fn).get_fdata(), (2, 0, 1))[:, :, ::-1].copy())
    roi_mask = orig_roi_mask.clone()
    assert (roi_mask==orig_roi_mask).all()
    roi_remap = {}
    for newi, i in enumerate(roi_mask.unique()):
        roi_remap[newi] = i
        roi_mask[roi_mask == i] = newi
    roi_mask = roi_mask / roi_mask.max()
    return roi_mask.numpy()

# vol = np.transpose(nib.load('P1_average_to_mri_warped_annotation_split_hemisphere.nii.gz').get_fdata(), (2, 0, 1))[:, :, ::-1].copy()
# vol = vedo.Volume(vol)
# plt = vedo.applications.IsosurfaceBrowser(vol, use_gpu=True, c='green')
# plt.show(axes=7, bg2='lb').close()
# exit()

# def main(pair_tag, brain_tag):
#     device='cuda:1'
#     r = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/registered' % (pair_tag, brain_tag)
#     mask_fn = os.path.join(r, '%s_MASK_topro_25_all.nii' % brain_tag)
#     # mask_fn = '%s_MASK_topro_25_all.nii' % brain_tag
#     mydata = load_atlas(mask_fn)
#     # zoom_f = 2
#     # mydata = scipy.ndimage.zoom(mydata, zoom_f)
#     # vol = Volume(mydata)#.print()
#     # mesh = isosurface(vol)
#     # mesh.alpha = 0.1
#     # center = '/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/center_pts_%s_%s_RoI16001.csv' % (pair_tag, brain_tag)
#     lightsheet_r = '/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4'
#     fn = '%s/NIS_%s_%s_RoI16001.csv' % (lightsheet_r, pair_tag, brain_tag)
#     center = torch.from_numpy(pandas.read_csv(fn).to_numpy())[:, :3].to(device)
#     vol = torch.from_numpy(pandas.read_csv(fn).to_numpy())[:, 3].to(device)
#     # vol = vol * 2.25
#     center = center[vol<1000]
#     vol = vol[vol<1000]
#     # cid = torch.arange(center.shape[0]).tolist()
#     # random.shuffle(cid)
#     # center = center[cid[:1000000]]
#     mshape = mydata.shape
#     center[:,0] = center[:,0] * mshape[0]
#     center[:,1] = center[:,1] * mshape[1]
#     center[:,2] = center[:,2] * mshape[2]
#     z = center[:, 0].clip(min=0, max=mshape[0]-0.1)
#     y = center[:, 1].clip(min=0, max=mshape[1]-0.1)
#     x = center[:, 2].clip(min=0, max=mshape[2]-0.1)
#     print(center.shape)
#     loc = torch.arange(mshape[0]*mshape[1]*mshape[2]).view(mshape[0], mshape[1], mshape[2]).to(device)
#     loc = loc[(z.long(), y.long(), x.long())]
#     loc_count = loc.bincount()
#     loc_count = loc_count[loc_count!=0]
#     atlas_loc = loc.unique().to(device)
#     ## volume avg
#     vol_avg = torch.zeros(mshape[0]*mshape[1]*mshape[2], dtype=torch.float64).to(device)
#     for loci in tqdm(atlas_loc):
#         vol_avg[loci] = vol[loci == loc].mean()
#     vol_avg = vol_avg.view(mshape[0], mshape[1], mshape[2]).cpu()
#     ## density map
#     density = torch.zeros(mshape[0]*mshape[1]*mshape[2], dtype=torch.float64).to(device)
#     density[atlas_loc] = loc_count.double() / center.shape[0]
#     density = density.view(mshape[0], mshape[1], mshape[2]).cpu()
#     # density[(x[atlas_loc].long(), y[atlas_loc].long(), z[atlas_loc].long())] = loc_count
#     # density = density.transpose(2, 0)
#     # from brainrender.scene import Scene
#     # from brainrender.actors import Points


#     # # initialise brainrender scene
#     # scene = Scene()
#     # scene.jupyter = True
#     # # create points actor
#     # cells = Points(center.numpy(), radius=45, colors="palegoldenrod", alpha=0.8)

#     # # visualise injection site (retrosplenial cortex)
#     # scene.add_brain_region(["RSPd"], color="mediumseagreen", alpha=0.6)
#     # scene.add_brain_region(["RSPv"], color="purple", alpha=0.6)
#     # scene.add_brain_region(["RSPagl"], color="mediumseagreen", alpha=0.6)

#     # # Add cells
#     # scene.add(cells)

#     # scene.render()

#     #  to actually display the scene we use `vedo`'s `show` method to show the scene's actors
#     # plt = Plotter()
#     # plt.show(*scene.renderables)  # same as vedo.show(*scene.renderables)
#     # cells = Points(torch.stack([x, y, z]).numpy(), r=9)#.addPointArray(scals, name='scals')
#     # densecloud = cells.densify(0.1, closest=10, niter=1) # return a new pointcloud.Points
#     # print(cells.N(), densecloud.N())
#     # plt.show(mesh, cells)
#     # plt.show(mesh, lego)
#     # vol = Volume(density)
#     # plt = SlicerPlotter( vol,
#     #                      bg='white', bg2='lightblue',
#     #                      cmaps=("gist_ncar_r","jet","Spectral_r","hot_r","bone_r"),
#     #                      useSlider3D=False,
#     #                    )
#     # plt.show()
#     # plt = IsosurfaceBrowser(vol) 
#     # plt.show(axes=7, bg2='lb')
#     new_header = nib.load(mask_fn).header
#     new_header['quatern_b'] = 0.5
#     new_header['quatern_c'] = -0.5
#     new_header['quatern_d'] = 0.5
#     new_header['qoffset_x'] = -0.0
#     new_header['qoffset_y'] = -0.0
#     new_header['qoffset_z'] = 0.0
#     affine_m = np.eye(4, 4)
#     affine_m[:3, :3] = 0
#     affine_m[0, 1] = -1
#     affine_m[1, 2] = -1
#     affine_m[2, 0] = 1
#     # np.transpose(density.numpy().astype(np.float32), (1, 2, 0))[::-1, ::-1]
#     nib.save(nib.Nifti1Image(density.numpy().astype(np.float32), affine_m, header=new_header), f'{lightsheet_r}/NIS_density_{pair_tag}_{brain_tag}.nii')
#     nib.save(nib.Nifti1Image(vol_avg.numpy().astype(np.float32), affine_m, header=new_header), f'{lightsheet_r}/NIS_volavg_{pair_tag}_{brain_tag}.nii')
#     # exit()
#     ## density map
#     # plt = RayCastPlotter(Volume(density.numpy()).mode(1).c('jet'), bg='black', bg2='blackboard', axes=7)  # Plotter instance
#     ## volume avg
#     # plt = RayCastPlotter(Volume(vol_avg.numpy()).mode(1).c('jet'), bg='black', bg2='blackboard', axes=7)  # Plotter instance

#     # plt.add(mesh)
#     # plt.show(viewup="z")#.close()
#     # exit()
#     # density = density*1000000
#     # vol = Volume(density.numpy()).print_histogram(bins=10, logscale=True, horizontal=True)
#     # vol.crop(back=0.30) # crop 50% from neg. y
#     # lego = vol.legosurface(vmin=density[density>0].min().item(), vmax=None, boundary=True)
#     # lego.cmap('jet', vmin=0, vmax=1.25).add_scalarbar()
#     # plt.show(lego, __doc__, axes=1, viewup='z').close()
#     # plt.show(vol)

def main(pair_tag, brain_tag, img_tags=[]):
    device = 'cuda:0'
    downsample_res = [25, 25, 25]
    seg_res = [2.5, 0.75, 0.75]
    # data_root = f"/cajal/Felix/Lightsheet/P4/{pair_tag}/output_{brain_tag}/registered/{brain_tag}_MOV_atlas_25.nii"
    seg_root = f"/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{pair_tag}/{brain_tag}"
    stat_root = f"/cajal/ACMUSERS/ziquanw/Lightsheet/statistics/{pair_tag}"
    save_root = f"/cajal/ACMUSERS/ziquanw/Lightsheet/renders/{pair_tag}"
    os.makedirs(save_root, exist_ok=True)
    new_header, affine_m = init_nib_header()
    seg_paths, sortks = listdir_sorted(seg_root, "NIScpp", ftail="seg.zip", sortkid=3)
    ## For visualization purpose
    dratio = [s/d for s, d in zip(seg_res, downsample_res)]
    dratio = [r*2 for r in dratio]
    sshape = torch.load(seg_paths[-1]).shape
    sshape = list(sshape)
    sshape[0] = sshape[0] + sortks[-1]
    dshape = [0,0,0]
    dshape[0] = int(sshape[0] * dratio[0])
    dshape[1] = int(sshape[1] * dratio[1])
    dshape[2] = int(sshape[2] * dratio[2])
    print(datetime.now(), f"Downsample space shape: {dshape}, ratio: {dratio}")
    total_center = torch.load(f"{stat_root}/{brain_tag}_nis_center.zip", map_location='cpu')
    total_vol = torch.load(f"{stat_root}/{brain_tag}_nis_volume.zip", map_location='cpu')
    print(datetime.now(), f"Loaded {total_vol.shape} NIS")
    density_total, vol_avg = downsample(total_center, total_vol, dratio, dshape, device, skip_vol=True)
    print(datetime.now(), f"Saving total density {pair_tag} {brain_tag}, Max density {density_total.max()}")
    nib.save(nib.Nifti1Image(density_total.numpy().astype(np.float64), affine_m, header=new_header), f'{save_root}/NIS_density_dr{dratio[0]}_topro_{pair_tag}_{brain_tag}.nii')
    # nib.save(nib.Nifti1Image(vol_avg.numpy().astype(np.float64), affine_m, header=new_header), f'{save_root}/NIS_volavg_{pair_tag}_{brain_tag}.nii')
    intensity_label_dict = torch.load(f"{stat_root}/{brain_tag}_NIS_colocContrastCalib_label.zip", map_location='cpu')
    # label_des = {'pn_mask':1, 'np_mask': 2, 'pp_mask': 3, 'nn_mask': 4}    
    for k in intensity_label_dict:
        density_k, vol_avg = downsample(total_center[intensity_label_dict[k]], total_vol[intensity_label_dict[k]], dratio, dshape, device, skip_vol=True)
        nib.save(nib.Nifti1Image(density_k.numpy().astype(np.float64), affine_m, header=new_header), f'{save_root}/NIS_density_dr{dratio[0]}_{k}_{pair_tag}_{brain_tag}.nii')
    
    
    # print(len(intensity_label_dict['pp_mask']), total_center.shape)
    # total_center = total_center[:len(intensity_label_dict['pp_mask'])]
    # total_vol = total_vol[:len(intensity_label_dict['pp_mask'])]
    # intensity_label = torch.zeros(total_center.shape[0]).long()
    # for k in label_des:
    #     intensity_label[intensity_label_dict[k][:len(intensity_label)]] = label_des[k] 
    # print(datetime.now(), f"intensity_label.bincount: {intensity_label.bincount()}")
    # density_c1, _ = downsample(total_center[intensity_label==2], total_vol[intensity_label==2], dratio, dshape, device, skip_vol=True)
    # density_c2, _ = downsample(total_center[intensity_label==3], total_vol[intensity_label==3], dratio, dshape, device, skip_vol=True)
    # # for k in label_des:
    # #     density_c1, _ = downsample(total_center[intensity_label==label_des[k]], total_vol[intensity_label==label_des[k]], dratio, dshape, device, skip_vol=True)
    # #     print(datetime.now(), f"Saving C{label_des[k]}({k}) density {pair_tag} {brain_tag}, Max density {density_c1.max()}")
    # #     nib.save(nib.Nifti1Image(density_c1.numpy().astype(np.float64), affine_m, header=new_header), f'{save_root}/NIS_density_C{label_des[k]}({k})_{pair_tag}_{brain_tag}.nii')
    # density = torch.zeros_like(density_c1)
    # density[density_c1>density_c2] = 1
    # density[density_c1<density_c2] = 2
    # nib.save(nib.Nifti1Image(density.numpy().astype(np.float32), affine_m, header=new_header), f'{save_root}/NIS_density_C2vsC3_{pair_tag}_{brain_tag}.nii')
    # coloc = plot_local_2channel(intensity_label, total_center, total_vol, 150, 2267, 2267, 500, device)
    # nib.save(nib.Nifti1Image(coloc.numpy().astype(np.float32), affine_m, header=new_header), f'{save_root}/NIS_density_C2C3loc_{pair_tag}_{brain_tag}.nii')

def plot_local_2channel(clabel, center, vol, min0, min1, min2, size, device):
    zr = 0.2
    center = center.to(device).long()
    vol = vol.to(device)
    clabel = clabel.to(device)
    f0 = center[:, 0] >= min0
    f1 = center[:, 1] >= min1
    f2 = center[:, 2] >= min2
    f3 = center[:, 0] < min0 + int(size/zr)
    f4 = center[:, 1] < min1 + size
    f5 = center[:, 2] < min2 + size
    f = f0&f1&f2&f3&f4&f5
    vol = vol[f]
    rads = ((vol*3/4)/torch.pi)**(1/3)
    rads = rads.long()
    center = center[f]
    center[:, 0] = center[:, 0] - min0
    center[:, 1] = center[:, 1] - min1
    center[:, 2] = center[:, 2] - min2
    clabel = clabel[f]
    dot_size = 3
    # dot_c=torch.arange(dot_size)
    # dot_x, dot_y, dot_z = torch.meshgrid([dot_c, dot_c, dot_c])
    # dot_x, dot_y, dot_z = dot_x.view(-1), dot_y.view(-1), dot_z.view(-1)
    out = torch.zeros((int(size/zr), size, size), device=device)
    for i in trange(len(center)):
        r = rads[i]
        tag = clabel[i]
        if tag == 1 or tag == 4: continue
        # for x,y,z in zip(dot_x, dot_y, dot_z):
        out[center[i, 0]-r-dot_size:center[i, 0]+r+dot_size, center[i, 1]-r-dot_size:center[i, 1]+r+dot_size, center[i, 2]-r-dot_size:center[i, 2]+r+dot_size] = tag
    # print("c1==0,c2==1", f.sum())
    # out[center[f, 0], center[f, 1], center[f, 2]] = 2
    # print("c1==1,c2==0", f.sum())
    # out[center[f, 0], center[f, 1], center[f, 2]] = 3
    return out.cpu()

def downsample(center, vol, ratio, dshape, device, skip_vol=False):
    center = center.to(device)
    vol = vol.float().to(device)
    center[:,0] = center[:,0] * ratio[0]
    center[:,1] = center[:,1] * ratio[1]
    center[:,2] = center[:,2] * ratio[2]
    z = center[:, 0].clip(min=0, max=dshape[0]-0.9)
    y = center[:, 1].clip(min=0, max=dshape[1]-0.9)
    x = center[:, 2].clip(min=0, max=dshape[2]-0.9)
    # print(center.shape)
    loc = torch.arange(dshape[0]*dshape[1]*dshape[2]).view(dshape[0], dshape[1], dshape[2]).to(device) 
    loc = loc[(z.long(), y.long(), x.long())] # all nis location in the downsample space
    loc_count = loc.bincount() 
    loc_count = loc_count[loc_count!=0] 
    atlas_loc = loc.unique().to(device) # unique location in the downsample space
    ## volume avg & local intensity
    vol_avg = None
    if not skip_vol:
        vol_avg = torch.zeros(dshape[0]*dshape[1]*dshape[2], dtype=torch.float64).to(device)
        for loci in tqdm(atlas_loc, desc="Collect NIS property in local cube"): 
            where_loc = torch.where(loc==loci)[0]
            vol_avg[loci] = vol[where_loc].mean()
        vol_avg = vol_avg.view(dshape[0], dshape[1], dshape[2]).cpu()
    ## density map
    density = torch.zeros(dshape[0]*dshape[1]*dshape[2], dtype=torch.float64).to(device)
    density[atlas_loc] = loc_count.double() #/ center.shape[0]
    density = density.view(dshape[0], dshape[1], dshape[2]).cpu()
    return density, vol_avg

def init_nib_header():
    mask_fn = "/lichtman/Felix/Lightsheet/P4/pair15/output_L73D766P4/registered/L73D766P4_MASK_topro_25_all.nii"
    new_header = nib.load(mask_fn).header
    new_header['quatern_b'] = 0.5
    new_header['quatern_c'] = -0.5
    new_header['quatern_d'] = 0.5
    new_header['qoffset_x'] = -0.0
    new_header['qoffset_y'] = -0.0
    new_header['qoffset_z'] = 0.0
    affine_m = np.eye(4, 4)
    affine_m[:3, :3] = 0
    affine_m[0, 1] = -1
    affine_m[1, 2] = -1
    affine_m[2, 0] = 1
    return new_header, affine_m


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


if __name__ == '__main__':
    # _r = '/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4'

    # data_list = []
    # for r, d, fs in os.walk(_r):
    #     data_list.extend([os.path.join(r, f) for f in fs if f.endswith('_remap.json') and 'pair' in r])
    # # print(data_list)
    # # exit()
    # for remap_fn in tqdm(data_list[1:]):
    #     # if 'L73D766P5' not in remap_fn: continue
    #     if 'L79D769P8' not in remap_fn: continue
    #     # try:
    #     brain_tag = os.path.dirname(remap_fn).split('/')[-1]
    #     pair_tag = os.path.dirname(os.path.dirname(remap_fn)).split('/')[-1]
    #     main(pair_tag, brain_tag)
    #     # except:
    #     #     continue


    img_tags = ['C1_', 'C2_','C3_']

    # pair_tag = 'pair6'
    # brain_tag = 'L57D855P6'
    # main(pair_tag, brain_tag, img_tags)
    # # TODO: NIS used data in lichtman, but should to use cajal
    # pair_tag = 'pair6'
    # brain_tag = 'L57D855P2'
    # main(pair_tag, brain_tag, img_tags)


    # pair_tag = 'pair5'
    # brain_tag = 'L57D855P4'
    # main(pair_tag, brain_tag, img_tags)
    # pair_tag = 'pair5'
    # brain_tag = 'L57D855P5'
    # main(pair_tag, brain_tag, img_tags)


    # pair_tag = 'pair19'
    # brain_tag = 'L79D769P8'
    # main(pair_tag, brain_tag, img_tags)
    # pair_tag = 'pair19'
    # brain_tag = 'L79D769P5'
    # main(pair_tag, brain_tag, img_tags)


    # pair_tag = 'pair17'
    # brain_tag = 'L77D764P2'
    # main(pair_tag, brain_tag, img_tags)
    
    # pair_tag = 'pair17'
    # brain_tag = 'L77D764P9'
    # main(pair_tag, brain_tag, img_tags)


    
    # pair_tag = 'pair13'
    # brain_tag = 'L69D764P6'
    # main(pair_tag, brain_tag, img_tags)
    
    # pair_tag = 'pair13'
    # brain_tag = 'L69D764P9'
    # main(pair_tag, brain_tag, img_tags)

    
    # pair_tag = 'pair15'
    # brain_tag = 'L73D766P4'
    # main(pair_tag, brain_tag, img_tags)
    
    pair_tag = 'pair15'
    brain_tag = 'L73D766P9'
    main(pair_tag, brain_tag, img_tags)


    
    # pair_tag = 'pair10'
    # brain_tag = 'L64D804P3'
    # main(pair_tag, brain_tag, img_tags)
    # # 
    # pair_tag = 'pair10'
    # brain_tag = 'L64D804P9'
    # main(pair_tag, brain_tag, img_tags)
    
    # pair_tag = 'pair11'
    # brain_tag = 'L66D764P3'
    # main(pair_tag, brain_tag, img_tags)
    # pair_tag = 'pair11'
    # brain_tag = 'L66D764P8'
    # main(pair_tag, brain_tag, img_tags)

    # pair_tag = 'pair12'
    # brain_tag = 'L66D764P5'
    # main(pair_tag, brain_tag, img_tags)
    # pair_tag = 'pair12'
    # brain_tag = 'L66D764P6'
    # main(pair_tag, brain_tag, img_tags)

    # pair_tag = 'pair14'
    # brain_tag = 'L73D766P5'
    # main(pair_tag, brain_tag, img_tags)
    # pair_tag = 'pair14'
    # brain_tag = 'L73D766P7'
    # main(pair_tag, brain_tag, img_tags)

    # pair_tag = 'pair16'
    # brain_tag = 'L74D769P4'
    # main(pair_tag, brain_tag, img_tags)
    # pair_tag = 'pair16'
    # brain_tag = 'L74D769P8'
    # main(pair_tag, brain_tag, img_tags)

    # pair_tag = 'pair18'
    # brain_tag = 'L77D764P4'
    # main(pair_tag, brain_tag, img_tags)

    # pair_tag = 'pair20'
    # brain_tag = 'L79D769P7'
    # main(pair_tag, brain_tag, img_tags)
    # pair_tag = 'pair20'
    # brain_tag = 'L79D769P9'
    # main(pair_tag, brain_tag, img_tags)

    # pair_tag = 'pair21'
    # brain_tag = 'L91D814P2'
    # main(pair_tag, brain_tag, img_tags)
    # pair_tag = 'pair21'
    # brain_tag = 'L91D814P6'
    # main(pair_tag, brain_tag, img_tags)

    # pair_tag = 'pair22'
    # brain_tag = 'L91D814P3'
    # main(pair_tag, brain_tag, img_tags)
    # pair_tag = 'pair22'
    # brain_tag = 'L91D814P4'
    # main(pair_tag, brain_tag, img_tags)

    # pair_tag = 'pair3'
    # brain_tag = 'L35D719P1'
    # main(pair_tag, brain_tag, img_tags)
    # pair_tag = 'pair3'
    # brain_tag = 'L35D719P4'
    # main(pair_tag, brain_tag, img_tags)


    # pair_tag = 'pair8'
    # brain_tag = 'L59D878P2'
    # main(pair_tag, brain_tag, img_tags)
    # pair_tag = 'pair8'
    # brain_tag = 'L59D878P5'
    # main(pair_tag, brain_tag, img_tags)


    # pair_tag = 'pair9'
    # brain_tag = 'L64D804P4'
    # main(pair_tag, brain_tag, img_tags)
    # pair_tag = 'pair9'
    # brain_tag = 'L64D804P6'
    # main(pair_tag, brain_tag, img_tags)