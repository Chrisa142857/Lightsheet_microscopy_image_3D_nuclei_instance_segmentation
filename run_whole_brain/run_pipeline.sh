## Run first two steps: 2D Unet + 2D-to-3D
## python run_2steps.py brain start_slicei end_slicei gpu_id

NOW="$(date +"%m-%d-%Y-%T")"
nohup python run_whole_brain/run_2steps.py L73D766P4 0 -1 1 >> run_2dunet_log_$NOW.out 

# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_2d_to_3d.py L73D766P4 0 >> run_2d_to_3d_log_$NOW.out 

# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_2dunet.py L73D766P9 0 -1 0 >> run_2dunet_log_$NOW.out 

# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_2d_to_3d.py L73D766P9 0 >> run_2d_to_3d_log_$NOW.out 
