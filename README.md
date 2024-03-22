# Lightsheet microscopy whole brain 3D nuclei instance segmentation
[![](https://img.shields.io/badge/github-_project_-blue?style=social&logo=github)](https://github.com/Chrisa142857/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation) &ensp;

2D Unet-based

### Motivations:
 - None has developed a nuclei instance segmentation for whole brain.
 - Lightsheet microscopy 3D image has anisotropic resolution.
 - Resolution is isotropic in X-Y plane, where human annotating nuclei, then tracking Z stack.

### Performance and time cost:
 - Recall > 90% 
 - Precision > 90%
 - Time cost ~= **12**hr/brain

### Data:
Multiple whole brains of mouse in different grown stage. Each brain has ~1500x9000x9000 voxels, and about 30,000,000 to 50,000,000 cells.

### Example to test a whole-brain:

#### 1 Obtain NIS results of one brain
```
P_tag=Name the test (e.g., P4)
pair_tag=Name the brain group
brain_tag=Name the brain
dataroot=/path/to/directory/2D_slice_image
saveroot=/path/to/directory/saving/results/${P_tag}/${pair_tag}/${brain_tag}/
mkdir -p ${saveroot}
nohup cpp/build/test ${pair_tag} ${brain_tag} ${device} ${dataroot} ${saveroot} > cpp_logs/${brain_tag}_${P_tag}.log
```

#### 2 Statistic NIS results and collect all NIS center, coordinates, intensity
Change paths in `run_whole_brain/statistic_cpp.py`
```
device='cuda:X'
seg_root = 'XXX'
save_root = 'XXX'
P_tag = 'XXX'
brain_tag = 'XXX'
pair_tag = 'XXX'
data_root = 'XXX'
```
Adjust `img_tags` to get intensity of different image channels.

Then `python run_whole_brain/statistic_cpp.py`. All NIS will be saved to `save_root`.

#### 2.5 (Optional) Cell type labeling using intensity of different channels
See `nis_coloc.py`

#### 3 Visualize downsampled whole-brain NIS result
Similarly, change paths in `brain_render.py`
```
downsample_res = X.X
seg_res = X.X
seg_root = 'XXX'
stat_root = 'XXX'
save_root = 'XXX'
```

Then `python brain_render.py`

#### 4 (Optional) Statistic downsampled NIS result in region-level with registered atlas
See `stats/statistic_nii.py`, `stats/statistic_csv.py`.


### TODO: Interactive visualization of whole brain nuclei segmentation results
 - [ ] Github [page](http://lightsheet-nis.ziquanw.com/).

### Codes availability
 - [x] Train 2D Unet
 - [x] Source code of the executible to test a whole brain

### Data availability
 - [ ] Train-val data
 - [ ] Test whole brain

### Usage
 - Follow `README` under directories.