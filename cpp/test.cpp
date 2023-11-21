#include <torch/script.h> // One-stop header.
// #include <filesystem>
#include <future>

// #include <iostream>
#include <memory>
#include <string> 

#include "loop_unet.h"
#include "utils.h"
#include "flow_op.h"


std::vector<torch::Tensor> nis_obtain(torch::jit::script::Module flow_3DtoSeed, torch::Tensor dP, torch::Tensor cp_mask){
  torch::Tensor p = index_flow(dP * cp_mask / 5., dP.size(1), dP.size(2), dP.size(3), 139);
  std::vector<torch::Tensor> nis_outputs = flow_3DtoNIS(
    // meshgrider,
    flow_3DtoSeed,
    p, 
    cp_mask, 20
    );
  return nis_outputs;
}

int main(int argc, const char* argv[]) {
  std::cout << "LibTorch version: "
    << TORCH_VERSION_MAJOR << "."
    << TORCH_VERSION_MINOR << "."
    << TORCH_VERSION_PATCH << std::endl;
  std::string device = "cuda:0";
  int64_t chunk_depth = 30;
  float cellprob_threshold = 0.0;

  torch::jit::script::Module get_tile_param;
  torch::jit::script::Module preproc;
  torch::jit::script::Module nis_unet;
  torch::jit::script::Module postproc;
  torch::jit::script::Module grad_2d_to_3d;
  torch::jit::script::Module interpolater;
  // torch::jit::script::Module meshgrider;
  // torch::jit::script::Module index_flow3D;
  torch::jit::script::Module flow_3DtoSeed;
  get_tile_param = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/get_model_tileparam_cpu.pt");
  preproc = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/preproc_img1xLyxLx.pt");
  nis_unet = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/nis_unet_cpu.pt");
  postproc = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/postproc_unet.pt");
  grad_2d_to_3d = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/grad_2Dto3D.pt");
  interpolater = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/interpolate_ratio_1.6x1x1.pt");
  // meshgrider = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/meshgrider.pt");
  // index_flow3D = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/index_flow3D.pt");
  flow_3DtoSeed = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/flow_3DtoSeed.pt");
  std::string pair_tag = "pair15";
  std::string brain_tag = "L73D766P4";
  std::string img_dir = "/lichtman/Felix/Lightsheet/P4/"+pair_tag+"/output_"+brain_tag+"/stitched/";
  std::string mask_dir = "/lichtman/ziquanw/Lightsheet/roi_mask/"+pair_tag+"/"+brain_tag+"/";
  auto allimgs = listdir_sorted(img_dir);
  auto allmasks = listdir_sorted(mask_dir);
  std::vector<std::string> img_fns;
  std::vector<std::string> mask_fns;
  for (std::string file : allimgs ) {
    if (file.find("_C1_") != std::string::npos) {
      img_fns.push_back(file);
    }
  }
  for (std::string file : allmasks ) {
    mask_fns.push_back(file);
  }
  if (mask_fns.size() != img_fns.size()) {exit(0);}
  std::cout<<"There are "<<img_fns.size()<<" .tif images\n";
  torch::NoGradGuard no_grad;
  torch::Tensor pre_final_yx_flow;
  torch::Tensor pre_last_second;
  std::cout<<"Torch: no grad\n";
  std::future<std::vector<torch::Tensor>> nis_obtainer;
  std::vector<torch::Tensor> nis_outputs;
  for (int64_t i = 0; i < img_fns.size(); i+=chunk_depth) {
    print_with_time("Chunk has slice "+std::to_string(i)+"~"+std::to_string(i+chunk_depth)+"\n");
    std::vector<std::string> img_onechunk;
    std::vector<std::string> mask_onechunk;
    for (int64_t j = i; j < i+chunk_depth; j++){
      if (j >= img_fns.size()) {break;}
      img_onechunk.push_back(img_fns[j]);
      mask_onechunk.push_back(mask_fns[j]);
    }
    /*
    Loop slices of one chunk into Unet (GPU & IO)
    */
    torch::Tensor flow2d = loop_unet(
      img_onechunk, 
      mask_onechunk, 
      get_tile_param,
      preproc,
      nis_unet,
      postproc, device
    );
    /*
    Resample probability map to train-data resolution (GPU)
      train_resolution = (2.5, .75, .75)
      input_resolution = (4, .75, .75)
    */
    flow2d = flow2d.permute({3, 0, 1 ,2});
    // print_with_time("========== flow 2d before inter ==========\n");
    // std::cout<< flow2d.mean() << "," << flow2d.max() << "\n";
    std::vector<torch::jit::IValue> inter_inputs({flow2d});
    flow2d = interpolater(inter_inputs).toTensor();
    /*
    2D flow to 3D flow (GPU)
    */
    if (i > 0) {
     flow2d = torch::cat({pre_final_yx_flow.unsqueeze(1), flow2d}, 1);
    }
    pre_final_yx_flow = flow2d.index({torch::indexing::Slice(torch::indexing::None, 3, 1), -1}).clone();
    pre_last_second = flow2d.index({torch::indexing::Slice(torch::indexing::None, 2, 1), -2}).clone();
    bool is_first_chunk = i == 0;
    torch::Tensor flow3d = flow_2Dto3D(flow2d, pre_last_second, grad_2d_to_3d, device, is_first_chunk);
    // print_size(flow3d);
    /*
    Follow the 3D flow to obtain NIS (CPU)
    */
    torch::Tensor dP = flow3d.index({torch::indexing::Slice(torch::indexing::None, 3), "..."});
    torch::Tensor cellprob = flow3d.index({3, "..."}); 
    torch::Tensor cp_mask = cellprob > cellprob_threshold; 
    // print_with_time("========== flow 3d ==========\n");
    // std::cout<< flow3d.mean() << "," << flow3d.max() << "\n";
    if (cp_mask.any().item<bool>()) {
      // if (i + chunk_depth > img_fns.size()) {
      //   nis_outputs = nis_obtain(flow_3DtoSeed, dP, cp_mask);
      // } else {
      if (i > 0){
        nis_outputs = nis_obtainer.get();
      }
      nis_obtainer = std::async(std::launch::async, nis_obtain, flow_3DtoSeed, dP, cp_mask);
      // }
    } else {
      print_with_time("No instance, probability map is all zero, continue");
      continue;
    }
    /*
    TODO: Save NIS results (IO) 
    */
    if (i > 0){
      // Save to H5 database

      if (i + chunk_depth > img_fns.size()) {
        nis_outputs = nis_obtainer.get();
        // Save the last chunk
      }
    }
    
    for (auto& out : nis_outputs) {
      print_size(out);
    }
  }
  std::cout << "ok\n";
}
