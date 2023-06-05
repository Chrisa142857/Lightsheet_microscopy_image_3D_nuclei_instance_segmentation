NOW="$(date +"%m-%d-%Y-%T")"
# python -m cellpose --use_gpu --gpu_device 0 --train --dir /ram/USERS/ziquanw/cellpose_exp/data_P4_P15_rescaled-as-P15/train --test_dir /ram/USERS/ziquanw/cellpose_exp/data_P4_P15_rescaled-as-P15/val --save_each --n_epochs 100 --save_every 10 --verbose >> train_log_$NOW.txt

python -m cellpose --use_gpu --gpu_device 0 --train --pretrained_model nuclei --dir /ram/USERS/ziquanw/cellpose_exp/data_P4_P15_rescaled-as-P15/train --test_dir /ram/USERS/ziquanw/cellpose_exp/data_P4_P15_rescaled-as-P15/val --save_each --n_epochs 100 --save_every 10 --verbose >> train_log_$NOW.txt

