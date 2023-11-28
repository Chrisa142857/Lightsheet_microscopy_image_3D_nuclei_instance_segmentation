#include <highfive/highfive.hpp>
#include <torch/script.h> // One-stop header.

using namespace HighFive;

DataSet create_extensible_dataset(File file, std::string DATASET_NAME, hsize_t ncol, bool dtype_is_long);

std::vector<DataSet> init_h5data(std::string FILE_NAME, std::vector<hsize_t> whole_brain_shape);

torch::Tensor save_h5data(std::vector<DataSet> dsetlist, std::vector<torch::Tensor> datalist, hsize_t old_instance_n, hsize_t old_contour_n, hsize_t zmin, hsize_t zmax, std::vector<hsize_t> whole_brain_shape);
