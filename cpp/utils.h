#include <torch/script.h> // One-stop header.
#include <torch/nn/functional.h>

#include <chrono>
#include <iostream>
#include <filesystem>
#include <set>
#include <sstream>
#include <iomanip>

std::string replaceWithFormattedNumbers(const std::string& x, int z1, int z2, std::string extra_str);
std::string getFilename(const std::string& path);
int split_then_int(const std::string& x, int loc);
std::vector<torch::Tensor> pad_image(torch::Tensor img0);
std::set<std::string> listdir_sorted(std::string path);
void print_size(torch::Tensor tensor);
void print_with_time(std::string msg);
torch::Tensor normalize_image(torch::Tensor img);
torch::Tensor preproc_image(torch::Tensor img);
