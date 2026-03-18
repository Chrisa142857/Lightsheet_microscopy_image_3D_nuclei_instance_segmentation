# ptags = ['male', 'male', 'female', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'female', 'extrabrains', 'extrabrains', 'female', 'female']
# btags = ['L82D711P2', 'L82D711P3', 'L86P4', 'L87D868P2', 'L87P1', 'L88P1', 'L88P2', 'L88P3', 'L88P4', 'L90P3', 'L92P2', 'L92P3', 'L92P4', 'L94P1', 'L94P3sox9toproneun', 'L94P4', 'L95P2', 'L95P3', 'L95P4', 'L96P3', 'L97P1', 'L97P2', 'L106P3', 'L106P5']

# skipb = ['L87P1', 'L88P1', 'L88P3', 'L88P4', 'L90P3', 'L88P2', 'L82D711P3', 'L92P2', 'L97P1', 'L97P2']
# btags = [b for b in btags if b not in skipb]
import glob
flist = glob.glob('/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/pair*/2*')
flist = [f for f in flist if '.' not in f and f.split('/')[-2] not in ['pair18', 'pair19']]
ptags = [f.split('/')[-2] for f in flist]
btags = [f.split('/')[-1].split('_')[1] for f in flist]
btag2b = {
    f.split('/')[-1].split('_')[1]: f.split('/')[-1]
    for f in flist
}
import os
import numpy as np
cmds = []
basecmd = '/ram/USERS/ziquanw/softwares/miniconda3/envs/wholeBrain/bin/python /ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/analysis_nis_shape.py --gtag P4 --ptag %s --btag %s --ttag %d-%d'
for ptag, btag in zip(ptags, btags):
    b = btag2b[btag]
    result_path = f'/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/{ptag}/{b}'
    result_root = result_path + '/UltraII[%02d x %02d]'
    for i in range(4):
        for j in range(5):
            if not os.path.exists(result_root % (i, j)): continue
            stack_names = os.listdir(result_root % (i, j))
            stack_names = [n for n in stack_names if 'instance_center' in n]
            tgt_names = [f"{result_root % (i, j)}/{n.replace('instance_center', 'instance_pa')}".replace('cajal', 'scheibel') for n in stack_names]
            ifexist = [os.path.exists(n) for n in tgt_names]
            if not np.all(ifexist):
                # print(np.array(stack_names)[~np.array(ifexist)])
                # print(np.array(tgt_names)[~np.array(ifexist)])
                cmds.append(f"{basecmd % (ptag, btag, i, j)}")
            
print('\n'.join(cmds))
