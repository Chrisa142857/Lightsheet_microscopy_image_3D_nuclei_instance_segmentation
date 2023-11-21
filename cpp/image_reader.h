#include <torch/script.h> // One-stop header.

#include <opencv2/opencv.hpp>
#include <cstdlib> // Include the necessary header for exit()
// #include <opencv2/core.hpp>
// #include <opencv2/highgui.hpp>
// #include <ants.hh>
#include <opencv2/core.hpp>

torch::Tensor load_tif_as_tensor(std::string path);
// torch::Tensor
// void load_nii_as_tensor(std::string path);
// std::vector<cv::Mat> opencv_read_nii(std::string path);