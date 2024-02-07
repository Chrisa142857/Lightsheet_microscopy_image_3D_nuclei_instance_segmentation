cd cellpose
NOW="$(date +"%m-%d-%Y-%T")"
python -m cellpose --use_gpu --gpu_device 0 --train --pretrained_model nuclei --dir ../../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n4 --test_dir ../../downloads/train_data/data_P4_P15_rescaled-as-P15/val --save_each --n_epochs 100 --save_every 10 --verbose >> ../train_n4_log_$NOW.txt
NOW="$(date +"%m-%d-%Y-%T")"
python -m cellpose --use_gpu --gpu_device 0 --train --pretrained_model nuclei --dir ../../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n8 --test_dir ../../downloads/train_data/data_P4_P15_rescaled-as-P15/val --save_each --n_epochs 100 --save_every 10 --verbose >> ../train_n8_log_$NOW.txt
NOW="$(date +"%m-%d-%Y-%T")"
python -m cellpose --use_gpu --gpu_device 0 --train --pretrained_model nuclei --dir ../../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n12 --test_dir ../../downloads/train_data/data_P4_P15_rescaled-as-P15/val --save_each --n_epochs 100 --save_every 10 --verbose >> ../train_n12_log_$NOW.txt

