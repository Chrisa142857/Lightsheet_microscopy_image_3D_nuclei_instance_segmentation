# conda activate wholeBrain
NOW="$(date +"%m-%d-%Y-%T")"

## Run third steps: stitch gaps
## python run_stitch_step.py brain   

# nohup python run_whole_brain/run_stitch_step.py L73D766P4 pair15 0 >> log_$NOW.out  & 
# sleep 1
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L73D766P9 pair15 0 >> log_$NOW.out 
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L64D804P4 pair9 0 >> log_$NOW.out & 
# sleep 1
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L64D804P6 pair9 1 >> log_$NOW.out
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L59D878P2 pair8 0 >> log_$NOW.out & 
# sleep 1
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L59D878P5 pair8 1 >> log_$NOW.out 
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L64D804P3 pair10 0 >> log_$NOW.out & 
# sleep 1
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L64D804P9 pair10 1 >> log_$NOW.out
# NOW="$(date +"%m-%d-%Y-%T")" 
# nohup python run_whole_brain/run_stitch_step.py L66D764P5 pair12 0 >> log_$NOW.out & 
# sleep 1
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L66D764P6 pair12 1 >> log_$NOW.out
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L73D766P5 pair14 0 >> log_$NOW.out & 
# sleep 1
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L73D766P7 pair14 1 >> log_$NOW.out 
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L66D764P3 pair11 0 >> log_$NOW.out & 
# sleep 1
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L66D764P8 pair11 0 >> log_$NOW.out &
# sleep 1
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L74D769P4 pair16 0 >> log_$NOW.out & 
# sleep 1
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L74D769P8 pair16 1 >> log_$NOW.out
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L91D814P2 pair21 0 >> log_$NOW.out & 
# sleep 1
# nohup python run_whole_brain/run_stitch_step.py L91D814P6 pair21 >> log_$NOW.out 
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L91D814P3 pair22 1 >> log_$NOW.out
# sleep 1
# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_stitch_step.py L91D814P4 pair22 0 >> log_$NOW.out 


NOW="$(date +"%m-%d-%Y-%T")"
nohup python run_whole_brain/run_stitch_step.py L79D769P9 pair20 0 >> log_$NOW.out 
NOW="$(date +"%m-%d-%Y-%T")"
nohup python run_whole_brain/run_stitch_step.py L79D769P7 pair20 0 >> log_$NOW.out 
NOW="$(date +"%m-%d-%Y-%T")"
nohup python run_whole_brain/run_stitch_step.py L35D719P4 pair3 0 >> log_$NOW.out 
NOW="$(date +"%m-%d-%Y-%T")"
nohup python run_whole_brain/run_stitch_step.py L35D719P1 pair3 0 >> log_$NOW.out 

