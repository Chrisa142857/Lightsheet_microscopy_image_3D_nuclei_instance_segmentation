#include <torch/script.h> // One-stop header.
#include <torch/nn/functional.h>
#include <future>
#include "utils.h"
#include "image_reader.h"

std::vector<torch::Tensor> loop_unet(
  std::vector<std::string> img_fns, 
  // torch::jit::script::Module* get_tile_param,
  // torch::jit::script::Module* preproc,
  torch::jit::script::Module* nis_unet,
  bool do_fg_filter,
  std::string device,
  std::string lefttop_fn = "",
  std::string righttop_fn = "",
  std::string leftbottom_fn = "",
  std::string rightbottom_fn = ""
);

torch::Tensor average_tiles(torch::Tensor y, torch::Tensor ysub, torch::Tensor xsub, int64_t Ly, int64_t Lx);

torch::Tensor _taper_mask(torch::Device device, int64_t bsize = 224, float sig = 7.5);

std::vector<torch::Tensor> tile_image(torch::Tensor image, int64_t bsize, torch::Tensor overlap);