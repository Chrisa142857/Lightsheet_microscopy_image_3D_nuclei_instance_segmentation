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