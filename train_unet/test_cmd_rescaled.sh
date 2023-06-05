NOW="$(date +"%m-%d-%Y-%T")"

IMG_PATH="../data/Felix_P4/Felix_01x00_C00_Z0800_sub5_2.tif"
python -m cellpose/cellpose --use_gpu --gpu_device 1 --image_path $IMG_PATH --pretrained_model ../downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_15_12_23_32.710910_epoch_81 --do_3D --save_tif --no_npy --verbose --savedir ../downloads/train_data/test_out_rescaled >> test_log_$NOW.txt


IMG_PATH="../data/Carolyn_org_Sept/images/Side0[00x02]sub4.tif"
python -m cellpose/cellpose --use_gpu --gpu_device 1 --image_path $IMG_PATH --pretrained_model ../downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_15_12_23_32.710910_epoch_81 --do_3D --save_tif --no_npy --verbose --savedir ../downloads/train_data/test_out_rescaled >> test_log_$NOW.txt


IMG_PATH="../data/Carolyn_org_Sept/images/Side0[01x00]sub1.tif"
python -m cellpose/cellpose --use_gpu --gpu_device 1 --image_path $IMG_PATH --pretrained_model ../downloads/train_data/data_P4_P15_rescaled-as-P15/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_05_15_12_23_32.710910_epoch_81 --do_3D --save_tif --no_npy --verbose --savedir ../downloads/train_data/test_out_rescaled >> test_log_$NOW.txt

