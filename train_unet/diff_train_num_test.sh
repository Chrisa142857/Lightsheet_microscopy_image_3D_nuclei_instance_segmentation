cd cellpose 
# NOW="$(date +"%m-%d-%Y-%T")"
# IMG_PATH="/ram/USERS/ziquanw/data/Felix_P4/Felix_01x00_C00_Z0800_sub5_2.tif"
# python -m cellpose --use_gpu --gpu_device 0 --image_path $IMG_PATH --pretrained_model ../../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n4/models/cellpose_residual_on_style_on_concatenation_off_train_n4_2024_01_22_17_57_52.823846_epoch_31 --do_3D --save_tif --no_npy --verbose --savedir ../../downloads/train_data/test_out_rescaled_n4
# IMG_PATH="/ram/USERS/ziquanw/data/Carolyn_org_Sept/images/Side0[00x02]sub4.tif"
# python -m cellpose --use_gpu --gpu_device 0 --image_path $IMG_PATH --pretrained_model ../../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n4/models/cellpose_residual_on_style_on_concatenation_off_train_n4_2024_01_22_17_57_52.823846_epoch_31 --do_3D --save_tif --no_npy --verbose --savedir ../../downloads/train_data/test_out_rescaled_n4
# IMG_PATH="/ram/USERS/ziquanw/data/Carolyn_org_Sept/images/Side0[01x00]sub1.tif"
# python -m cellpose --use_gpu --gpu_device 0 --image_path $IMG_PATH --pretrained_model ../../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n4/models/cellpose_residual_on_style_on_concatenation_off_train_n4_2024_01_22_17_57_52.823846_epoch_31 --do_3D --save_tif --no_npy --verbose --savedir ../../downloads/train_data/test_out_rescaled_n4

NOW="$(date +"%m-%d-%Y-%T")"
IMG_PATH="/ram/USERS/ziquanw/data/Felix_P4/Felix_01x00_C00_Z0800_sub5_2.tif"
python -m cellpose --use_gpu --gpu_device 0 --image_path $IMG_PATH --pretrained_model ../../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n8/models/cellpose_residual_on_style_on_concatenation_off_train_n8_2024_01_22_18_05_20.815697_epoch_21 --do_3D --save_tif --no_npy --verbose --savedir ../../downloads/train_data/test_out_rescaled_n8
IMG_PATH="/ram/USERS/ziquanw/data/Carolyn_org_Sept/images/Side0[00x02]sub4.tif"
python -m cellpose --use_gpu --gpu_device 0 --image_path $IMG_PATH --pretrained_model ../../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n8/models/cellpose_residual_on_style_on_concatenation_off_train_n8_2024_01_22_18_05_20.815697_epoch_21 --do_3D --save_tif --no_npy --verbose --savedir ../../downloads/train_data/test_out_rescaled_n8
IMG_PATH="/ram/USERS/ziquanw/data/Carolyn_org_Sept/images/Side0[01x00]sub1.tif"
python -m cellpose --use_gpu --gpu_device 0 --image_path $IMG_PATH --pretrained_model ../../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n8/models/cellpose_residual_on_style_on_concatenation_off_train_n8_2024_01_22_18_05_20.815697_epoch_21 --do_3D --save_tif --no_npy --verbose --savedir ../../downloads/train_data/test_out_rescaled_n8


# NOW="$(date +"%m-%d-%Y-%T")"
# IMG_PATH="/ram/USERS/ziquanw/data/Felix_P4/Felix_01x00_C00_Z0800_sub5_2.tif"
# python -m cellpose --use_gpu --gpu_device 0 --image_path $IMG_PATH --pretrained_model ../../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n12/models/cellpose_residual_on_style_on_concatenation_off_train_n12_2024_01_22_18_19_24.791368_epoch_81 --do_3D --save_tif --no_npy --verbose --savedir ../../downloads/train_data/test_out_rescaled_n12
# IMG_PATH="/ram/USERS/ziquanw/data/Carolyn_org_Sept/images/Side0[00x02]sub4.tif"
# python -m cellpose --use_gpu --gpu_device 0 --image_path $IMG_PATH --pretrained_model ../../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n12/models/cellpose_residual_on_style_on_concatenation_off_train_n12_2024_01_22_18_19_24.791368_epoch_81 --do_3D --save_tif --no_npy --verbose --savedir ../../downloads/train_data/test_out_rescaled_n12
# IMG_PATH="/ram/USERS/ziquanw/data/Carolyn_org_Sept/images/Side0[01x00]sub1.tif"
# python -m cellpose --use_gpu --gpu_device 0 --image_path $IMG_PATH --pretrained_model ../../downloads/train_data/data_P4_P15_rescaled-as-P15/train_n12/models/cellpose_residual_on_style_on_concatenation_off_train_n12_2024_01_22_18_19_24.791368_epoch_81 --do_3D --save_tif --no_npy --verbose --savedir ../../downloads/train_data/test_out_rescaled_n12

