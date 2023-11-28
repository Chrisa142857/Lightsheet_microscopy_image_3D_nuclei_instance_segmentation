#include "utils.h"

namespace fs = std::filesystem;


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


void print_size(torch::Tensor tensor) {
  std::cout << "Tensor size: " << tensor.sizes() << ", Device: " << tensor.device() << "\n";
}


void print_with_time(std::string msg){
    auto now_time = std::chrono::high_resolution_clock::now();
    std::time_t current_time = std::chrono::high_resolution_clock::to_time_t(now_time);
    std::cout<<std::ctime(&current_time)<< " " << msg;
}