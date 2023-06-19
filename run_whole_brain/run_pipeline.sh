NOW="$(date +"%m-%d-%Y-%T")"

## Run first two steps: 2D Unet + 2D-to-3D
## python run_2steps.py brain start_slicei end_slicei gpu_id

# python run_whole_brain/run_2steps.py L73D766P4 0 -1 1 >> log_$NOW.out 

# ## Run second CPU steps: 3D to mask
# ## python run_cpu_step.py brain   

# python run_whole_brain/run_cpu_step.py L73D766P4 0 -1 1 >> log_$NOW.out 


## Run third steps: stitch gaps
## python run_stitch_step.py brain   

python run_whole_brain/run_stitch_step.py L73D766P4 >> log_$NOW.out 
python run_whole_brain/run_stitch_step.py L73D766P9 >> log_$NOW.out 

