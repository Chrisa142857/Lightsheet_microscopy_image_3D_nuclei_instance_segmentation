import supervision as sv
import nibabel as nib
import numpy as np

def write_vtk(numpy_array, fn):
    f = open(fn,'w') # change your vtk file name
    f.write('X,Y,Z\n')
    # f.write('# vtk DataFile Version 2.0\n')
    # f.write('test\n')
    # f.write('ASCII\n')
    # f.write('DATASET STRUCTURED_POINTS\n')
    # f.write('DIMENSIONS 3 3 4\n') # change your dimension
    # f.write('SPACING 0.100000 0.100000 0.100000\n')
    # f.write('ORIGIN 0 0 0\n')
    # f.write('POINT_DATA 36\n') # change the number of point data
    # f.write('SCALARS volume_scalars float 1\n')
    # f.write('LOOKUP_TABLE default\n')
    loc = ''
    for x, y, z in numpy_array:
        loc += f'{x},{y},{z}\n'
    f.write(loc) # change your numpy array here but need some processing
    f.close()

vol_fn = '/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/pair16/L74D769P4/whole_brain_map/NIS_density_after_rm_pair16_L74D769P4.nii.gz'
brain_mask_fn = '/lichtman/Felix/Lightsheet/P4/pair16/output_L74D769P4/registered/L74D769P4_MASK_topro_25_all.nii'

mask = nib.load(brain_mask_fn).get_fdata()
polygons = []
for mi in np.unique(mask):
    if mi == 0: continue
    m3d = (mask == mi).astype(np.uint8)
    poly = []
    for z, m in enumerate(m3d):
        p = sv.mask_to_polygons(m)
        if len(p) == 0: continue
        p = np.concatenate(p)
        p = np.concatenate([np.zeros([p.shape[0], 1])+z, p], 1)
        poly.append(p)
    poly = np.concatenate(poly)
    print(poly)
    write_vtk(poly, './temp.csv')
    exit()
    polygons.append(poly)
