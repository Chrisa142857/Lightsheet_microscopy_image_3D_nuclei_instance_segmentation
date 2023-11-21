#include <torch/script.h> // One-stop header.
#include "torchist.h"
#include <future>


torch::Tensor flow_2Dto3D(
    torch::Tensor flow_2d, 
    torch::Tensor pre_last_second, 
    torch::jit::script::Module sim_grad_z,
    std::string device,
    bool skip_first
    );

torch::Tensor index_flow(
    // torch::jit::script::Module meshgrider, 
    torch::Tensor dP, 
    int64_t Lz, int64_t Ly, int64_t Lx, int64_t niter
    );

std::vector<torch::Tensor> flow_3DtoNIS(
    // torch::jit::script::Module meshgrider, 
    torch::jit::script::Module flow_3DtoSeed,
    torch::Tensor p, 
    torch::Tensor iscell, 
    int64_t rpad
    );
