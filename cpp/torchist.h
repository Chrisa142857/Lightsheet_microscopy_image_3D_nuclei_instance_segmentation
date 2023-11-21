#include <torch/torch.h>
#include <limits>
#include "utils.h"

torch::Tensor histogramdd(
    torch::Tensor x,
    std::vector<torch::Tensor> edges
);
torch::Tensor ravel_multi_index(torch::Tensor coords, torch::Tensor shape);
torch::Tensor out_of_bounds(torch::Tensor x, torch::Tensor low, torch::Tensor upp);