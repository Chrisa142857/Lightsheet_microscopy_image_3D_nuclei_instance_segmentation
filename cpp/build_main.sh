TORCH_LIBRARIES=/ram/USERS/ziquanw/libs/pytorch-install
HIGHFIVE_INSTALL_PREFIX=/ram/USERS/ziquanw/libs/HighFive_build

mkdir build_main
cd build_main

# cmake version
# cmake version 3.22.1
# CMake suite maintained and supported by Kitware (kitware.com/cmake).

/usr/bin/cmake -DCMAKE_PREFIX_PATH="${TORCH_LIBRARIES}" ..
/usr/bin/cmake --build . --config Release