TORCH_LIBRARIES=/ram/USERS/ziquanw/libs/pytorch-install-cu128/libtorch
export CUDA_HOME=/ram/USERS/ziquanw/softwares/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

rm -rf build_main_hummer/
mkdir build_main_hummer && cd build_main_hummer


# cmake version
# cmake version 3.22.1
# CMake suite maintained and supported by Kitware (kitware.com/cmake).

/usr/bin/cmake -DCMAKE_PREFIX_PATH=/ram/USERS/ziquanw/libs/pytorch-install-cu128 \
      -DCUDA_TOOLKIT_ROOT_DIR=/ram/USERS/ziquanw/softwares/cuda-12.8 \
      -DCUDAToolkit_ROOT=/ram/USERS/ziquanw/softwares/cuda-12.8 \
      -DCUDNN_ROOT=/usr \
      -DCUDNN_INCLUDE_DIR=/usr/include \
      -DCUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so.9 \
      -DUSE_CUDNN=ON \
      -DCMAKE_PREFIX_PATH="${TORCH_LIBRARIES}" ..
/usr/bin/cmake --build . --config Release