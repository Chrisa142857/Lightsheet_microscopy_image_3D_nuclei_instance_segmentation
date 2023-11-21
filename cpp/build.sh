TORCH_LIBRARIES=/ram/USERS/ziquanw/libs/pytorch-install

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=${TORCH_LIBRARIES} ..
cmake --build . --config Release