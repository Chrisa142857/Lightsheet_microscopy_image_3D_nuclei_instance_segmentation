import os, json
import xml.etree.ElementTree as ET 
import numpy as np
import torch

# tz tx ty
Adjust = {'L57D855P5':
    {
        # 'Z0750-Z0800': [62.5,0,0],
        # 'Z0800-Z0850': [62.5,0,0],
        # 'Z0850-Z0900': [62.5,0,0],
        # 'Z0900-Z0950': [62.5,0,0],
        'Z0500-Z0750': [-62.5,0,0],
    },
    'L57D855P4': 
    {},
}
def main():
    overlap_r_in_scripts = 0.2
    overlap_r = overlap_r_in_scripts
    ptag = 'pair5'
    btag = 'L57D855P4'
    result_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}'
    result_path = [f'{result_r}/{fn}' for fn in os.listdir(result_r) if btag in fn and not fn.startswith('L')][0]
    root = result_path + '/UltraII[%02d x %02d]'

    # r = f'/cajal/Felix/Lightsheet/stitching/Manual_aligned_{btag}'
    r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/imaris_stitch/Manual_aligned_{btag}'
    save_r = f'/cajal/ACMUSERS/ziquanw/Lightsheet/stitch_by_ptreg/{ptag}/{btag}/NIS_tranform'

    tile_loc = np.array([[int(fn[8:10]), int(fn[-3:-1])] for fn in os.listdir(result_path) if 'Ultra' in fn])
    print(result_path, tile_loc)
    ncol, nrow = tile_loc.max(0)+1
    assert len(tile_loc) == nrow*ncol, f'tile of raw data is not complete, tile location: {tile_loc}'
    stack_names = [f for f in os.listdir(root % (0, 0)) if f.endswith('instance_center.zip')]
    for stack_name in stack_names:
        meta_name = stack_name.replace('instance_center', 'seg_meta')
        seg_shape = torch.load(f'{root % (0, 0)}/{meta_name}').tolist()
        break
    print(seg_shape)
    tile_lt_loc = {
        f'{i}-{j}': get_tile_lt_loc(seg_shape[1]*(1-overlap_r),seg_shape[2]*(1-overlap_r),i,j) for i in range(ncol) for j in range(nrow)
    }
    tile_lt_loc_down = {
        f'{i}-{j}': get_tile_lt_loc(seg_shape[1]*(1-overlap_r),seg_shape[2]*(1-overlap_r),ncol-1-i,j) for i in range(ncol) for j in range(nrow)
    }

    tform_manual = {}
    zshift = {k: 0 for k in tile_lt_loc}
    fns = list(os.listdir(r))
    chunkzs = []
    for fn in fns:
        chunkz = int(fn.split('_')[1])
        chunkzs.append(-1*chunkz)
    
    all_zshift = {}
    for fni in np.argsort(chunkzs):
        fn = fns[fni]
        all_zshift[fn] = get_zshift(f'{r}/{fn}', ncol)
    avg_zshift = {k: [] for k in all_zshift[fn]}
    for fn in all_zshift:
        for k in avg_zshift:
            avg_zshift[k].append(all_zshift[fn][k])
    for k in avg_zshift:
        print(k, avg_zshift[k])
        avg_zshift[k] = np.mean(avg_zshift[k])
    print(avg_zshift)
    for fni in np.argsort(chunkzs):
        fn = fns[fni]
        chunkz = int(fn.split('_')[1])
        tform, zshift = parseXML(f'{r}/{fn}', tile_lt_loc, zshift, [seg_shape[1],seg_shape[2]], ncol, btag, avg_zshift)
        for k in tform:
            assert k not in tform_manual
            tform_manual[k] = tform[k]
    print(tform_manual[0])
    zs = list(tform_manual.keys())
    assert min(zs) == 0, min(zs)
    assert len(zs) == max(zs)+1, f'{len(zs)} != {max(zs)+1}'
    tform = []
    for z in range(max(zs)+1):
        tform.append(tform_manual[z])

    with open(f'{save_r}/{btag}_tform_manual.json', 'w', encoding='utf-8') as f:
        json.dump(tform, f, ensure_ascii=False, indent=4)

def get_zshift(xmlfile, nx): 
    tree = ET.parse(xmlfile) 
    if tree.getroot().attrib['Direction'] == 'RightUp':
        def retrieve_tile_xy(fn):
            tilex, tiley = fn.split('_')[-3].split(' x ')
            tilex = int(tilex[8:10])
            tiley = int(tiley[:-1])
            return tilex, tiley
    elif tree.getroot().attrib['Direction'] == 'RightDown':
        def retrieve_tile_xy(fn):
            tilex, tiley = fn.split('_')[-3].split(' x ')
            tilex = int(tilex[8:10])
            tiley = int(tiley[:-1])
            return nx-1-tilex, tiley
    z_shift = {}
    for i in range(len(tree.getroot()[0])):
        if tree.getroot()[0][i].tag == 'Image':
            item = tree.getroot()[0][i].attrib
            zsplit, fn = item['Filename'].split('\\')[-2:]
            tilex, tiley = retrieve_tile_xy(fn)
            k = f'{tilex}-{tiley}'
            z_shift[k] = float(item['MinZ'])# - minz_shift + prev_z_shift[k] + extra_tz

    # print("Max z shift", np.array(list(z_shift.values())).max())
    print("Avg z shift", np.array(list(z_shift.values())).mean())
    return z_shift
            

def get_tile_lt_loc(w, h, i, j):
    return [i*w, j*h]

# def get_minxyz(tree, item, xshape=0, yshape=0):
#     if tree.getroot().attrib['Direction'] == 'RightUp':
#         return float(item['MinZ']), float(item['MinY']), float(item['MinX'])
#     if tree.getroot().attrib['Direction'] == 'RightDown':
#         return float(item['MinZ']), xshape-float(item['MinY']), float(item['MinX'])
#     if tree.getroot().attrib['Direction'] == 'LeftUp':
#         return float(item['MinZ']), float(item['MinY']), yshape-float(item['MinX'])
#     if tree.getroot().attrib['Direction'] == 'LeftDown':
#         return float(item['MinZ']), xshape-float(item['MinY']), yshape-float(item['MinX'])
    
def parseXML(xmlfile, tile_lt_loc, prev_z_shift, seg_shape, nx, btag, z_shift): 
    tree = ET.parse(xmlfile) 
    if tree.getroot().attrib['Direction'] == 'RightUp':
        if_down = 1
        def retrieve_manual_tform(items):
            ty, tx = float(items['MinX']), float(items['MinY'])
            return tx, ty
        def retrieve_tile_xy(fn):
            tilex, tiley = fn.split('_')[-3].split(' x ')
            tilex = int(tilex[8:10])
            tiley = int(tiley[:-1])
            return tilex, tiley
    elif tree.getroot().attrib['Direction'] == 'RightDown':
        if_down = -1
        def retrieve_manual_tform(items):
            th, tw = float(items['MaxX']) - float(items['MinX']), float(items['MaxY']) - float(items['MinY'])
            wrat, hrat = 1,1 #tw / seg_shape[0], th / seg_shape[1]
            ty, tx = float(items['MinX']), float(items['MinY'])
            return tx/wrat, ty/hrat
        def retrieve_tile_xy(fn):
            tilex, tiley = fn.split('_')[-3].split(' x ')
            tilex = int(tilex[8:10])
            tiley = int(tiley[:-1])
            return nx-1-tilex, tiley
    out = {}
    # z_shift = zshift
    # z_shifts = []
    # for i in range(len(tree.getroot()[0])):
    #     if tree.getroot()[0][i].tag == 'Image':
    #         item = tree.getroot()[0][i].attrib
    #         z_shifts.append(float(item['MinZ']))
    # minz_shift = min(z_shifts)
    for i in range(len(tree.getroot()[0])):
        if tree.getroot()[0][i].tag == 'Image':
            item = tree.getroot()[0][i].attrib
            zsplit, fn = item['Filename'].split('\\')[-2:]
            if zsplit in Adjust[btag]:
                extra_tz, extra_tx, extra_ty = Adjust[btag][zsplit]
            else:
                extra_tz, extra_tx, extra_ty = 0,0,0
            zmin = int(zsplit.split('-')[0][1:])
            zmax = int(zsplit.split('-')[1][1:])
            tilex, tiley = retrieve_tile_xy(fn)
            k = f'{tilex}-{tiley}'
            # get_minxyz(tree, item, xshape=)
            tx, ty = retrieve_manual_tform(item)
            if tilex == 0 and tiley == 0 and tx != 0 and ty != 0: 
                print(tree.getroot().attrib['Direction'], k, tile_lt_loc[k], tx, ty, float(item['MinZ']), f"{xmlfile.split('/')[-1]} anchor moved !")
            else:
                print(tree.getroot().attrib['Direction'], k, tile_lt_loc[k], tx, ty, float(item['MinZ']))
            tx = if_down*(tx - tile_lt_loc[k][0]) + extra_tx
            ty = (ty - tile_lt_loc[k][1]) + extra_ty
            # z_shift[k] = float(item['MinZ'])# - minz_shift + prev_z_shift[k] + extra_tz
            for z in range(zmin, zmax):
                if z not in out: out[z] = {}
                out[z][k] = [z_shift[k], tx*0.75, ty*0.75]
    # print("Max z shift", np.array(list(z_shift.values())).max())
    print("Avg z shift", np.array(list(z_shift.values())).mean())
    return out, z_shift
            

if __name__ == '__main__': main()