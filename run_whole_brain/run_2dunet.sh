NOW="$(date +"%m-%d-%Y-%T")"
## python .py start_slicei end_slicei gpu_id
nohup python run_whole_brain/run_2dunet.py 0 -1 1 > run_2dunet_log_$NOW.out &

# NOW="$(date +"%m-%d-%Y-%T")"
# nohup python run_whole_brain/run_2dunet.py 500 -1 1 > run_2dunet_log_$NOW.out &