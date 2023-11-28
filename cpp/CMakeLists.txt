cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)
set(TORCH_LIBRARIES /ram/USERS/ziquanw/libs/pytorch-install)

find_package(HDF5 REQUIRED)
find_package(Torch REQUIRED)
find_package( OpenCV REQUIRED )
find_package(HighFive REQUIRED)

file(GLOB SOURCES
    ./*.h
    ./*.cpp
)

add_executable(test ${SOURCES})

target_link_libraries(test HighFive)
target_link_libraries(test "${TORCH_LIBRARIES}")
target_link_libraries(test ${OpenCV_LIBS})
target_link_libraries(test HDF5::HDF5)

set_property(TARGET test PROPERTY CXX_STANDARD 17)