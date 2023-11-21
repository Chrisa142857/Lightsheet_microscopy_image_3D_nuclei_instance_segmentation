
fix=/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/register_atlas/L74D769P4/L35D719P4_C1_topro_25_300x300x300_1.nrrd
mov=/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/register_atlas/L74D769P4/NIS_density_pair16_L74D769P4.nii
trans=/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/register_atlas/L74D769P4/L74D769P4_ManualTransform.mat
out=/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/register_atlas/L74D769P4/NIS_density_pair16_L74D769P4_linear_transform.nii
BRAINSFit --numberOfIterations 0 --initialTransform $trans --outputVolume $out --fixedVolume $fix --movingVolume $mov --useRigid

while read -r trans; do
    
done < downloads/register_manualtrans.txt

