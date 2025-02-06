## Executible to infer a whole brain

This code has only been validated on Linux.

### To build

 - Download the Libtorch via
```
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```
 - Install OpenCV: [Linux](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html).
 - Change `TORCH_LIBRARIES` in `CMakeLists.txt`. Then
```
sh build_main.sh
```

### To use

 - Download models from G-drive [URL](https://drive.google.com/drive/folders/12YGRtoW4DHftVyhaGoZMl-xdc02Mj9SB?usp=sharing). Only `cuda:0` and `cuda:1` is supported.
 - NIS command line is `build_main/main --help`
```
Usage: main [options] 

Optional arguments:
-h --help                               shows help message and exits [default: false]
-v --version                            prints version information and exits [default: false]
--device                                GPU ID [default: "cuda:0"]
--batch_size                            batch_size [default: "400"]
--chunk_depth                           chunk_depth [default: "600"]
--cellprob_threshold                    cellprob_threshold [default: "0.1"]
-mroot --model_root                     path to .pt models [default: ""]
-ptag --pair_tag                        pair_tag [default: "pair4"]
-btag --brain_tag                       brain_tag [default: "100"]
-in --data_root                         specify the data_root. [default: "./outputs/"]
-out --save_root                        specify the save_root [default: ""]
--lt_corner_image_path                  left_top_corner_image_path for foreground detection [default: "None"]
--rt_corner_image_path                  right_top_corner_image_path for foreground detection [default: "None"]
--lb_corner_image_path                  left_bottom_corner_image_path for foreground detection [default: "None"]
--rb_corner_image_path                  right_bottom_corner_image_path for foreground detection [default: "None"]
-no_fg_det --no_foreground_detection    skip foreground_detection [default: false]
```

### Example
 - Download an example data (please email me to gain access) from G-drive , which contains a 2x2 tiles (100x2000x2000 each tile) of a part of one P4 mouse brain.
 - Put data under `../downloads/data/test_pair/test_brain`, then the `data` folder should looks like
```
.
└── test_pair
    └── test_brain
        ├── UltraII[00 x 00]
        ├── UltraII[00 x 01]
        ├── UltraII[01 x 00]
        └── UltraII[01 x 01]
            ...
            └── L ... Z0099.ome.tif
```
 - Put models under `../downloads/resource`
 - Run the following. The peak GPU memory usage will be `~16GB`, the peak CPU memory usage will be `~38GB`. Reduce `--chunk_depth` or `--batch_size` if it is still out of CPU or GPU memory, respectively.
```
mkdir -p "../downloads/cpp_output/test_pair/test_brain/UltraII[00 x 00]"

build_main/main --device cuda:0 --chunk_depth 35 -mroot ../downloads/resource -ptag test_pair -btag test_brain -in "../downloads/data/test_pair/test_brain/UltraII[00 x 00]" -out "../downloads/cpp_output/test_pair/test_brain/UltraII[00 x 00]" -no_fg_det

mkdir -p "../downloads/cpp_output/test_pair/test_brain/UltraII[00 x 01]"

build_main/main --device cuda:0 --chunk_depth 35 -mroot ../downloads/resource -ptag test_pair -btag test_brain -in "../downloads/data/test_pair/test_brain/UltraII[00 x 01]" -out "../downloads/cpp_output/test_pair/test_brain/UltraII[00 x 01]" -no_fg_det

mkdir -p "../downloads/cpp_output/test_pair/test_brain/UltraII[01 x 00]"

build_main/main --device cuda:0 --chunk_depth 35 -mroot ../downloads/resource -ptag test_pair -btag test_brain -in "../downloads/data/test_pair/test_brain/UltraII[01 x 00]" -out "../downloads/cpp_output/test_pair/test_brain/UltraII[01 x 00]" -no_fg_det

mkdir -p "../downloads/cpp_output/test_pair/test_brain/UltraII[01 x 01]"

build_main/main --device cuda:0 --chunk_depth 35 -mroot ../downloads/resource -ptag test_pair -btag test_brain -in "../downloads/data/test_pair/test_brain/UltraII[01 x 01]" -out "../downloads/cpp_output/test_pair/test_brain/UltraII[01 x 01]" -no_fg_det
```
