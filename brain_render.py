import vedo
from vedo import Plotter, utils  # <- this will be used to render an embedded scene 
# from itkwidgets import view
# from imio import load
# from vedo import Volume, Points, VolumeSlice, Text2D, probe_points, Axes
from vedo.applications import RayCastPlotter, IsosurfaceBrowser
from vedo.pyplot import histogram
import os
import pandas
import torch
import random
import nibabel as nib
import numpy as np
import scipy
import h5py
from tqdm import tqdm
# try:
#     import vedo.vtkclasses as vtk
# except ImportError:
#     import vtkmodules.all as vtk
# def isosurface(self, value=None, flying_edges=True):
#     """
#     Return an `Mesh` isosurface extracted from the `Volume` object.

#     Set `value` as single float or list of values to draw the isosurface(s).
#     Use flying_edges for faster results (but sometimes can interfere with `smooth()`).

#     Examples:
#         - [isosurfaces.py](https://github.com/marcomusy/vedo/tree/master/examples/volumetric/isosurfaces.py)

#             ![](https://vedo.embl.es/images/volumetric/isosurfaces.png)
#     """
#     scrange = self._data.GetScalarRange()

#     if flying_edges:
#         cf = vtk.vtkFlyingEdges3D()
#         cf.InterpolateAttributesOn()
#     else:
#         cf = vtk.vtkContourFilter()
#         cf.UseScalarTreeOn()

#     cf.SetInputData(self._data)
#     cf.ComputeNormalsOn()

#     if utils.is_sequence(value):
#         cf.SetNumberOfContours(len(value))
#         for i, t in enumerate(value):
#             cf.SetValue(i, t)
#     else:
#         if value is None:
#             value = (2 * scrange[0] + scrange[1]) / 3.0
#             # print("automatic isosurface value =", value)
#         cf.SetValue(0, value)

#     cf.Update()
#     poly = cf.GetOutput()

#     out = vedo.mesh.Mesh(poly, c=None, alpha=0.1).phong()
#     out.mapper().SetScalarRange(scrange[0], scrange[1])

#     out.pipeline = utils.OperationNode(
#         "isosurface",
#         parents=[self],
#         comment=f"#pts {out.inputdata().GetNumberOfPoints()}",
#         c="#4cc9f0:#e9c46a",
#     )
#     return out

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

def main(pair_tag, brain_tag):
    device='cuda:1'
    r = '/lichtman/Felix/Lightsheet/P4/%s/output_%s/registered' % (pair_tag, brain_tag)
    mask_fn = os.path.join(r, '%s_MASK_topro_25_all.nii' % brain_tag)
    # mask_fn = '%s_MASK_topro_25_all.nii' % brain_tag
    mydata = load_atlas(mask_fn)
    # zoom_f = 2
    # mydata = scipy.ndimage.zoom(mydata, zoom_f)
    # vol = Volume(mydata)#.print()
    # mesh = isosurface(vol)
    # mesh.alpha = 0.1
    # center = '/lichtman/ziquanw/Lightsheet/results/P4/center_pts_%s_%s_RoI16001.csv' % (pair_tag, brain_tag)
    lightsheet_r = '/lichtman/ziquanw/Lightsheet/results/P4'
    fn = '%s/NIS_%s_%s_RoI16001.csv' % (lightsheet_r, pair_tag, brain_tag)
    center = torch.from_numpy(pandas.read_csv(fn).to_numpy())[:, :3].to(device)
    vol = torch.from_numpy(pandas.read_csv(fn).to_numpy())[:, 3].to(device)
    # vol = vol * 2.25
    center = center[vol<1000]
    vol = vol[vol<1000]
    # cid = torch.arange(center.shape[0]).tolist()
    # random.shuffle(cid)
    # center = center[cid[:1000000]]
    mshape = mydata.shape
    center[:,0] = center[:,0] * mshape[0]
    center[:,1] = center[:,1] * mshape[1]
    center[:,2] = center[:,2] * mshape[2]
    z = center[:, 0].clip(min=0, max=mshape[0]-0.1)
    y = center[:, 1].clip(min=0, max=mshape[1]-0.1)
    x = center[:, 2].clip(min=0, max=mshape[2]-0.1)
    print(center.shape)
    loc = torch.arange(mshape[0]*mshape[1]*mshape[2]).view(mshape[0], mshape[1], mshape[2]).to(device)
    loc = loc[(z.long(), y.long(), x.long())]
    loc_count = loc.bincount()
    loc_count = loc_count[loc_count!=0]
    atlas_loc = loc.unique().to(device)
    ## volume avg
    vol_avg = torch.zeros(mshape[0]*mshape[1]*mshape[2], dtype=torch.float64).to(device)
    for loci in tqdm(atlas_loc):
        vol_avg[loci] = vol[loci == loc].mean()
    vol_avg = vol_avg.view(mshape[0], mshape[1], mshape[2]).cpu()
    ## density map
    density = torch.zeros(mshape[0]*mshape[1]*mshape[2], dtype=torch.float64).to(device)
    density[atlas_loc] = loc_count.double() / center.shape[0]
    density = density.view(mshape[0], mshape[1], mshape[2]).cpu()
    # density[(x[atlas_loc].long(), y[atlas_loc].long(), z[atlas_loc].long())] = loc_count
    # density = density.transpose(2, 0)
    # from brainrender.scene import Scene
    # from brainrender.actors import Points


    # # initialise brainrender scene
    # scene = Scene()
    # scene.jupyter = True
    # # create points actor
    # cells = Points(center.numpy(), radius=45, colors="palegoldenrod", alpha=0.8)

    # # visualise injection site (retrosplenial cortex)
    # scene.add_brain_region(["RSPd"], color="mediumseagreen", alpha=0.6)
    # scene.add_brain_region(["RSPv"], color="purple", alpha=0.6)
    # scene.add_brain_region(["RSPagl"], color="mediumseagreen", alpha=0.6)

    # # Add cells
    # scene.add(cells)

    # scene.render()

    #  to actually display the scene we use `vedo`'s `show` method to show the scene's actors
    # plt = Plotter()
    # plt.show(*scene.renderables)  # same as vedo.show(*scene.renderables)
    # cells = Points(torch.stack([x, y, z]).numpy(), r=9)#.addPointArray(scals, name='scals')
    # densecloud = cells.densify(0.1, closest=10, niter=1) # return a new pointcloud.Points
    # print(cells.N(), densecloud.N())
    # plt.show(mesh, cells)
    # plt.show(mesh, lego)
    # vol = Volume(density)
    # plt = SlicerPlotter( vol,
    #                      bg='white', bg2='lightblue',
    #                      cmaps=("gist_ncar_r","jet","Spectral_r","hot_r","bone_r"),
    #                      useSlider3D=False,
    #                    )
    # plt.show()
    # plt = IsosurfaceBrowser(vol) 
    # plt.show(axes=7, bg2='lb')
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
    # np.transpose(density.numpy().astype(np.float32), (1, 2, 0))[::-1, ::-1]
    nib.save(nib.Nifti1Image(density.numpy().astype(np.float32), affine_m, header=new_header), f'{lightsheet_r}/NIS_density_{pair_tag}_{brain_tag}.nii')
    nib.save(nib.Nifti1Image(vol_avg.numpy().astype(np.float32), affine_m, header=new_header), f'{lightsheet_r}/NIS_volavg_{pair_tag}_{brain_tag}.nii')
    # exit()
    ## density map
    # plt = RayCastPlotter(Volume(density.numpy()).mode(1).c('jet'), bg='black', bg2='blackboard', axes=7)  # Plotter instance
    ## volume avg
    # plt = RayCastPlotter(Volume(vol_avg.numpy()).mode(1).c('jet'), bg='black', bg2='blackboard', axes=7)  # Plotter instance

    # plt.add(mesh)
    # plt.show(viewup="z")#.close()
    # exit()
    # density = density*1000000
    # vol = Volume(density.numpy()).print_histogram(bins=10, logscale=True, horizontal=True)
    # vol.crop(back=0.30) # crop 50% from neg. y
    # lego = vol.legosurface(vmin=density[density>0].min().item(), vmax=None, boundary=True)
    # lego.cmap('jet', vmin=0, vmax=1.25).add_scalarbar()
    # plt.show(lego, __doc__, axes=1, viewup='z').close()
    # plt.show(vol)

if __name__ == '__main__':
    _r = '/lichtman/ziquanw/Lightsheet/results/P4'

    data_list = []
    for r, d, fs in os.walk(_r):
        data_list.extend([os.path.join(r, f) for f in fs if f.endswith('_remap.json') and 'pair' in r])
    # print(data_list)
    # exit()
    for remap_fn in tqdm(data_list[1:]):
        if 'pair3' not in remap_fn: continue
        brain_tag = os.path.dirname(remap_fn).split('/')[-1]
        pair_tag = os.path.dirname(os.path.dirname(remap_fn)).split('/')[-1]
        main(pair_tag, brain_tag)