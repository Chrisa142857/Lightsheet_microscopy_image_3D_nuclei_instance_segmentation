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
 - Time cost ~= **15**hr/brain

### Data:
Multiple whole brains of mouse in different grown stage. Each brain has ~1500x9000x9000 voxels, and about 30,000,000 to 50,000,000 cells.

### Example to test one partial brain:

#### 1 Obtain NIS results of brain

Refer to `cpp/README.md`

#### 2 Postprocess NIS cpp output

`python coord_to_bbox.py [options]` will generate bounding box of NIS in the output folders for stitching.

```
usage: coord_to_bbox.py [-h] [--device DEVICE]
                        [--nis_output_root NIS_OUTPUT_ROOT]

Generate NIS bounding box

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE
  --nis_output_root NIS_OUTPUT_ROOT
                        Output dir of NIS cpp
```

#### 3 Image stitch

Change paths in `image_stitch/stitch_main.py`, then 
```
python image_stitch/stitch_main.py
```

#### 3.5 (Optional) Refine stitch by point registration

Note that this can lead to a worse stitching result.

Change paths in `image_stitch/ptreg_stitch_p14.py`, then 
```
python image_stitch/ptreg_stitch_p14.py
```

#### 4 Generate brain maps

Go to this [repo](https://github.com/Chrisa142857/napari-whole-brain-map). Do `Non-GUI usage` section.

#### 5 (Optional) Colocalization with additional channels

In working progress.

### TODO: Interactive visualization of whole brain nuclei segmentation results
 - [ ] Github [page](http://lightsheet-nis.ziquanw.com/).

### Codes availability
 - [x] Train 2D Unet
 - [x] Source code of the executible to test a whole brain

### Data availability
 - [ ] Train-val data
 - [ ] Test whole brain
