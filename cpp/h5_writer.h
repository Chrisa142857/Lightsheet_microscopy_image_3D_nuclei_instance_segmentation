#include <highfive/highfive.hpp>
#include <torch/script.h> // One-stop header.
#include "utils.h"

using namespace HighFive;

DataSet create_extensible_dataset(File file, std::string DATASET_NAME, hsize_t ncol, bool dtype_is_long);

void init_h5data(std::string FILE_NAME, std::vector<hsize_t> whole_brain_shape);

std::vector<torch::Tensor> save_h5data(std::string FILE_NAME, std::vector<torch::Tensor> datalist, hsize_t old_instance_n, hsize_t old_contour_n, hsize_t zmin, hsize_t zmax, std::vector<hsize_t> whole_brain_shape);
