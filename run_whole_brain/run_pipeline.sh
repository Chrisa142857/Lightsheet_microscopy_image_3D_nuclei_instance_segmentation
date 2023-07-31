conda activate wholeBrain
NOW="$(date +"%m-%d-%Y-%T")"

## Run first two steps: 2D Unet + 2D-to-3D
## python run_2steps.py pairID brainID gpu_id

# python run_whole_brain/run_2steps.py L73D766P4 pair15 0 >> log_$NOW.out 
# python run_whole_brain/run_2steps.py L73D766P9 pair15 1 >> log_$NOW.out 
# nohup python run_whole_brain/run_2steps.py L64D804P4 pair9 0 >> log_$NOW.out
# nohup python run_whole_brain/run_2steps.py L64D804P6 pair9 1 >> log_$NOW.out 
# nohup python run_whole_brain/run_2steps.py L59D878P2 pair8 0 >> log_$NOW.out
# nohup python run_whole_brain/run_2steps.py L59D878P5 pair8 1 >> log_$NOW.out 
# nohup python run_whole_brain/run_2steps.py L64D804P3 pair10 0 >> log_$NOW.out
# nohup python run_whole_brain/run_2steps.py L64D804P9 pair10 1 >> log_$NOW.out 
# nohup python run_whole_brain/run_2steps.py L66D764P5 pair12 0 >> log_$NOW.out
# nohup python run_whole_brain/run_2steps.py L66D764P6 pair12 1 >> log_$NOW.out
# nohup python run_whole_brain/run_2steps.py L73D766P5 pair14 0 >> log_$NOW.out
# nohup python run_whole_brain/run_2steps.py L73D766P7 pair14 1 >> log_$NOW.out 
# nohup python run_whole_brain/run_2steps.py L66D764P3 pair11 0 >> log_$NOW.out
# nohup python run_whole_brain/run_2steps.py L66D764P8 pair11 1 >> log_$NOW.out 
# nohup python run_whole_brain/run_2steps.py L74D769P4 pair16 0 >> log_$NOW.out
# nohup python run_whole_brain/run_2steps.py L74D769P8 pair16 1 >> log_$NOW.out 

# ## Run second CPU steps: 3D to mask
# ## python run_cpu_step.py brain   

# python run_whole_brain/run_cpu_step.py L73D766P4 0 -1 1 >> log_$NOW.out 
# nohup python run_whole_brain/run_cpu_step.py L64D804P4 pair9 >> log_$NOW.out
# nohup python run_whole_brain/run_cpu_step.py L64D804P6 pair9 >> log_$NOW.out
# nohup python run_whole_brain/run_cpu_step.py L59D878P2 pair8 >> log_$NOW.out
# nohup python run_whole_brain/run_cpu_step.py L59D878P5 pair8 >> log_$NOW.out 
# nohup python run_whole_brain/run_cpu_step.py L64D804P3 pair10 >> log_$NOW.out
# nohup python run_whole_brain/run_cpu_step.py L64D804P9 pair10 >> log_$NOW.out 
# nohup python run_whole_brain/run_cpu_step.py L66D764P5 pair12 >> log_$NOW.out
# nohup python run_whole_brain/run_cpu_step.py L66D764P6 pair12 >> log_$NOW.out
# nohup python run_whole_brain/run_cpu_step.py L73D766P5 pair14 >> log_$NOW.out
# nohup python run_whole_brain/run_cpu_step.py L73D766P7 pair14 >> log_$NOW.out 
# nohup python run_whole_brain/run_cpu_step.py L66D764P3 pair11 >> log_$NOW.out
# nohup python run_whole_brain/run_cpu_step.py L66D764P8 pair11 >> log_$NOW.out 
# nohup python run_whole_brain/run_cpu_step.py L74D769P4 pair16 >> log_$NOW.out
# nohup python run_whole_brain/run_cpu_step.py L74D769P8 pair16 >> log_$NOW.out 

## Run third steps: stitch gaps
## python run_stitch_step.py brain   

# python run_whole_brain/run_stitch_step.py L73D766P4 >> log_$NOW.out 
# python run_whole_brain/run_stitch_step.py L73D766P9 >> log_$NOW.out 
# nohup python run_whole_brain/run_stitch_step.py L64D804P4 pair9 >> log_$NOW.out
# nohup python run_whole_brain/run_stitch_step.py L64D804P6 pair9 >> log_$NOW.out
# nohup python run_whole_brain/run_stitch_step.py L59D878P2 pair8 >> log_$NOW.out
# nohup python run_whole_brain/run_stitch_step.py L59D878P5 pair8 >> log_$NOW.out 
# nohup python run_whole_brain/run_stitch_step.py L64D804P3 pair10 >> log_$NOW.out
# nohup python run_whole_brain/run_stitch_step.py L64D804P9 pair10 >> log_$NOW.out 
# nohup python run_whole_brain/run_stitch_step.py L66D764P5 pair12 >> log_$NOW.out
# nohup python run_whole_brain/run_stitch_step.py L66D764P6 pair12 >> log_$NOW.out
# nohup python run_whole_brain/run_stitch_step.py L73D766P5 pair14 >> log_$NOW.out
# nohup python run_whole_brain/run_stitch_step.py L73D766P7 pair14 >> log_$NOW.out 
# nohup python run_whole_brain/run_stitch_step.py L66D764P3 pair11 >> log_$NOW.out
# nohup python run_whole_brain/run_stitch_step.py L66D764P8 pair11 >> log_$NOW.out 
# nohup python run_whole_brain/run_stitch_step.py L74D769P4 pair16 >> log_$NOW.out
# nohup python run_whole_brain/run_stitch_step.py L74D769P8 pair16 >> log_$NOW.out 


