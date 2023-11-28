TORCH_LIBRARIES=/ram/USERS/ziquanw/libs/pytorch-install
HIGHFIVE_INSTALL_PREFIX=/ram/USERS/ziquanw/libs/HighFive_build

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="${TORCH_LIBRARIES};${HIGHFIVE_INSTALL_PREFIX}" ..
cmake --build . --config Release