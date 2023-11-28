#include <torch/script.h> // One-stop header.
#include <torch/nn/functional.h>
#include "utils.h"

torch::Tensor gnn_stitch_gap(
    torch::jit::script::Module gnn_message_passing, 
    torch::jit::script::Module gnn_classifier,
    torch::Tensor img1, 
    torch::Tensor mask1, 
    torch::Tensor flow1,
    torch::Tensor img2, 
    torch::Tensor mask2, 
    torch::Tensor flow2,
    std::string device
);

std::vector<torch::Tensor> build_graph(
    torch::Tensor img1, 
    torch::Tensor mask1, 
    torch::Tensor flow1,
    torch::Tensor img2, 
    torch::Tensor mask2, 
    torch::Tensor flow2,
    std::string device
);

std::vector<torch::Tensor> get_edge(
    torch::Tensor bbox1, //[N x 4]
    torch::Tensor bbox2, // [N x 4]
    torch::Tensor zflow1, //[N]
    torch::Tensor zflow2, // [N]
    std::string device
);

std::vector<torch::Tensor> get_feat(
    torch::Tensor img, 
    torch::Tensor mask, 
    torch::Tensor flow
);

torch::Tensor get_bbox(torch::Tensor mask);

torch::Tensor hist_flow_vector(torch::Tensor flow);

torch::Tensor norm_tensor(torch::Tensor tensor);
