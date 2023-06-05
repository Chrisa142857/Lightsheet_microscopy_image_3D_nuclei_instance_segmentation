NOW="$(date +"%m-%d-%Y-%T")"
python -m cellpose --use_gpu --gpu_device 1 --train --dir /ram/USERS/ziquanw/cellpose_exp/data_P4_P15/train --test_dir /ram/USERS/ziquanw/cellpose_exp/data_P4_P15/val --save_each --n_epochs 100 --save_every 10 --verbose >> train_log_$NOW.txt

