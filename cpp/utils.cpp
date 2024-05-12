#include "utils.h"

namespace fs = std::filesystem;
namespace F = torch::nn::functional;

std::set<std::string> listdir_sorted(std::string path){
    std::set<std::string> sorted_by_name;
    for (auto &entry : fs::directory_iterator(path)){
        sorted_by_name.insert(entry.path());
    }
    return sorted_by_name;
}

std::vector<torch::Tensor> pad_image(torch::Tensor img0){
  int64_t div = 16;
  int64_t extra = 1;
  torch::IntArrayRef img_shape = img0.sizes();
  int64_t Lpad = (div * (std::ceil(img_shape[1]/div)+1) - img_shape[1]);
  int64_t xpad1 = extra * div / 2 + Lpad / 2;
  int64_t xpad2 = extra * div / 2 + Lpad - Lpad / 2;
    
  Lpad = (div * (std::ceil(img_shape[2]/div)+1) - img_shape[2]);
  int64_t ypad1 = extra * div / 2 + Lpad / 2;
  int64_t ypad2 = extra * div / 2 + Lpad - Lpad / 2;
  torch::Tensor I = torch::constant_pad_nd(img0, {ypad1, ypad2, xpad1, xpad2, 0, 0}, 0);
  torch::Tensor ysub = torch::arange(xpad1, xpad1 + img_shape[1]);
  torch::Tensor xsub = torch::arange(ypad1, ypad1 + img_shape[2]);
  std::vector<torch::Tensor> outputs;
  outputs.push_back(I);
  outputs.push_back(ysub);
  outputs.push_back(xsub);
  return outputs;
}

torch::Tensor normalize_image(torch::Tensor img){
  // img.shape = [B, H, W]
  float eps = 0.001;
  // ##### Batch-wise normalization ####
  // int64_t area = img.size(0) * img.size(1) * img.size(2);
  // int64_t p99 = std::ceil(area*0.99);
  // int64_t p1 = std::ceil(area*0.01);
  // auto sortout = torch::sort(img.reshape({-1}));
  // torch::Tensor i99 = std::get<0>(sortout).index({p99-1});
  // torch::Tensor i1 = std::get<0>(sortout).index({p1-1});
  // img = img - i1;
  // return img.clip(eps) / (i99-i1).clip(eps);
  // ##### Patch-wise normalization ####
  int64_t area = img.size(1) * img.size(2);
  int64_t p99 = std::ceil(area*0.99);
  int64_t p1 = std::ceil(area*0.01);
  auto sortout = torch::sort(img.reshape({img.size(0), -1}));
  torch::Tensor flatten = std::get<0>(sortout);
  torch::Tensor i99 = flatten.index({torch::indexing::Slice(), p99-1});
  torch::Tensor i1 = flatten.index({torch::indexing::Slice(), p1-1});
  img = (img-(i1.unsqueeze(1).unsqueeze(1)));
  img = img.clip(eps) / (i99-i1).clip(eps).unsqueeze(1).unsqueeze(1);
  return img;
}

torch::Tensor preproc_image(torch::Tensor img){
  img = normalize_image(img);
  img = torch::cat({img, torch::zeros_like(img)});
  img = F::interpolate(
      img.unsqueeze(0), 
      F::InterpolateFuncOptions().scale_factor(std::vector<double>({1.437293, 1.437293})).mode(torch::kBilinear).align_corners(false)
  );
  return img[0];
}


void print_size(torch::Tensor tensor) {
  std::cout << "Tensor size: " << tensor.sizes() << ", Device: " << tensor.device() << "\n";
}


void print_with_time(std::string msg){
    auto now_time = std::chrono::high_resolution_clock::now();
    std::time_t current_time = std::chrono::high_resolution_clock::to_time_t(now_time);
    std::cout<<std::ctime(&current_time)<< " " << msg;
}