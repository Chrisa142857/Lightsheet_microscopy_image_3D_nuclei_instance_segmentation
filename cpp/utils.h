#include <torch/script.h> // One-stop header.

#include <chrono>
#include <iostream>
#include <filesystem>
#include <set>


std::vector<torch::Tensor> pad_image(torch::Tensor img0);
std::set<std::string> listdir_sorted(std::string path);
void print_size(torch::Tensor tensor);
void print_with_time(std::string msg);
