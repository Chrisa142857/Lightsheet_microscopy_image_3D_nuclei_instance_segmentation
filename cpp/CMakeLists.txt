cmake_minimum_required(VERSION 3.18)
project(main)

# LibTorch and OpenCV are located via CMAKE_PREFIX_PATH (host build: /opt/libtorch;
# conda build: $CONDA_PREFIX). find_package(Torch) sets TORCH_LIBRARIES itself.
find_package(Torch REQUIRED)
find_package( OpenCV REQUIRED )

file(GLOB SOURCES
    ./*.h
    ./*.hpp
    ./*.cpp
)
list(REMOVE_ITEM SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp"
)

add_executable(main ${SOURCES})

target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(main ${OpenCV_LIBS})

set_property(TARGET main PROPERTY CXX_STANDARD 17)