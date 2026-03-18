## ssh yukon

screen -S ${btag}-${tiletag}
conda activate wholeBrain
cd /ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation
python analysis_nis_shape.py --gtag P14 --ptag ${ptag} --btag ${btag} --ttag ${tiletag}

python analysis_nis_shape.py --gtag P14 --ptag male --btag L82D711P2 --device cuda:0
python analysis_nis_shape.py --gtag P14 --ptag male --btag L82D711P3 --device cuda:1
python analysis_nis_shape.py --gtag P14 --ptag female --btag L86P4 --device cuda:2
python analysis_nis_shape.py --gtag P14 --ptag male --btag L87D868P2 --device cuda:3
python analysis_nis_shape.py --gtag P14 --ptag male --btag L87P1 --device cuda:4
python analysis_nis_shape.py --gtag P14 --ptag male --btag L88P1 --device cuda:5
python analysis_nis_shape.py --gtag P14 --ptag male --btag L88P2 --device cuda:6
python analysis_nis_shape.py --gtag P14 --ptag female --btag L88P3 --device cuda:7
python analysis_nis_shape.py --gtag P14 --ptag female --btag L88P4 --device cuda:0
python analysis_nis_shape.py --gtag P14 --ptag female --btag L90P3 --device cuda:1
python analysis_nis_shape.py --gtag P14 --ptag male --btag L92P2 --device cuda:2
python analysis_nis_shape.py --gtag P14 --ptag female --btag L92P3 --device cuda:3
python analysis_nis_shape.py --gtag P14 --ptag female --btag L92P4 --device cuda:4
python analysis_nis_shape.py --gtag P14 --ptag male --btag L94P1 --device cuda:5
python analysis_nis_shape.py --gtag P14 --ptag female --btag L94P3sox9toproneun --device cuda:6 &
python analysis_nis_shape.py --gtag P14 --ptag female --btag L94P4 --device cuda:7
python analysis_nis_shape.py --gtag P14 --ptag male --btag L95P2 --device cuda:0 &
python analysis_nis_shape.py --gtag P14 --ptag female --btag L95P3 --device cuda:1 &
python analysis_nis_shape.py --gtag P14 --ptag female --btag L95P4 --device cuda:2 &
python analysis_nis_shape.py --gtag P14 --ptag female --btag L96P3 --device cuda:3 &
python analysis_nis_shape.py --gtag P14 --ptag extrabrains --btag L97P1 --device cuda:4 &
python analysis_nis_shape.py --gtag P14 --ptag extrabrains --btag L97P2 --device cuda:5 &
python analysis_nis_shape.py --gtag P14 --ptag female --btag L106P3 --device cuda:6 &
python analysis_nis_shape.py --gtag P14 --ptag female --btag L106P5 --device cuda:7 


## require torch >= 2.7 for torch.where(..., return_indices=True)
screen -S postproc_pa1
conda activate flashattn3 
cd /ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation
###
python postproc_nis_shape.py --gtag P14 --ptag male --btag L87P1 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag male --btag L88P1 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag female --btag L88P3 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag female --btag L88P4 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag female --btag L90P3 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag male --btag L88P2 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag male --btag L82D711P2 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag male --btag L82D711P3 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag female --btag L86P4 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag male --btag L87D868P2 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag male --btag L92P2 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag female --btag L92P3 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag female --btag L92P4 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag male --btag L94P1 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag female --btag L94P3sox9toproneun --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag female --btag L94P4 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag male --btag L95P2 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag female --btag L95P3 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag female --btag L95P4 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag female --btag L96P3 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag extrabrains --btag L97P1 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag extrabrains --btag L97P2 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag female --btag L106P3 --device cuda:5
python postproc_nis_shape.py --gtag P14 --ptag female --btag L106P5 --device cuda:5