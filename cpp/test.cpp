#include <torch/script.h> // One-stop header.
// #include <filesystem>
#include <future>

// #include <iostream>
#include <memory>
#include <string> 
#include <highfive/highfive.hpp>

#include "loop_unet.h"
#include "utils.h"
#include "flow_op.h"
#include "h5_writer.h"
#include "image_reader.h"
#include "gnn_stitch_gap.h"

using namespace HighFive;

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

std::vector<torch::Tensor> gpu_process(
  int64_t i, 
  int64_t chunk_depth, 
  std::vector<std::string> img_fns,
  // std::vector<std::string> mask_fns,
  torch::jit::script::Module get_tile_param, 
  torch::jit::script::Module preproc, 
  torch::jit::script::Module nis_unet, 
  // torch::jit::script::Module postproc, 
  torch::jit::script::Module interpolater,
  torch::jit::script::Module grad_2d_to_3d,
  torch::Tensor pre_final_yx_flow,
  torch::Tensor pre_last_second,
  std::string device
) {
  print_with_time("Chunk has slice "+std::to_string(i)+"~"+std::to_string(i+chunk_depth)+"\n");
  std::vector<std::string> img_onechunk;
  // std::vector<std::string> mask_onechunk;
  for (int64_t j = i; j < i+chunk_depth; j++){
    if (j >= img_fns.size()) {break;}
    img_onechunk.push_back(img_fns[j]);
    // mask_onechunk.push_back(mask_fns[j]);
  }
  /*
  Loop slices of one chunk into Unet (GPU & IO)
  */
  std::vector<torch::Tensor> unet_output = loop_unet(
    img_onechunk, 
    // mask_onechunk, 
    get_tile_param,
    preproc,
    nis_unet,
    // postproc, 
    device
  );
  torch::Tensor flow2d = unet_output[0];
  torch::Tensor first_img = unet_output[1];
  torch::Tensor last_img = unet_output[2];
  /*
  Resample probability map to train-data resolution (GPU)
    train_resolution = (2.5, .75, .75)
    input_resolution = (4, .75, .75)
  */
  flow2d = flow2d.permute({3, 0, 1 ,2});
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
  torch::Tensor first_flow = flow3d.index({torch::indexing::Slice(), 0, "..."});
  torch::Tensor last_flow = flow3d.index({torch::indexing::Slice(), -1, "..."});
  std::vector<torch::Tensor> output;
  output.push_back(flow3d);
  output.push_back(pre_final_yx_flow);
  output.push_back(pre_last_second);
  output.push_back(first_img);
  output.push_back(last_img);
  output.push_back(first_flow);
  output.push_back(last_flow);
  return output;
}

int main(int argc, const char* argv[]) {
  std::cout << "LibTorch version: "
    << TORCH_VERSION_MAJOR << "."
    << TORCH_VERSION_MINOR << "."
    << TORCH_VERSION_PATCH << std::endl;
  std::string device = "cuda:0";
  int64_t chunk_depth = 30;
  float cellprob_threshold = 0.1;

  torch::jit::script::Module get_tile_param;
  torch::jit::script::Module preproc;
  torch::jit::script::Module nis_unet;
  // torch::jit::script::Module postproc;
  torch::jit::script::Module grad_2d_to_3d;
  torch::jit::script::Module interpolater;
  torch::jit::script::Module gnn_message_passing;
  torch::jit::script::Module gnn_classifier;
  torch::jit::script::Module flow_3DtoSeed;
  get_tile_param = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/get_model_tileparam_cpu.pt");
  preproc = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/preproc_img1xLyxLx.pt");
  nis_unet = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/nis_unet_cpu.pt");
  // postproc = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/postproc_unet.pt");
  grad_2d_to_3d = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/grad_2Dto3D.pt");
  interpolater = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/interpolate_ratio_1.6x1x1.pt");
  gnn_message_passing = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/gnn_message_passing.pt");
  gnn_classifier = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/gnn_classifier.pt");
  flow_3DtoSeed = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/flow_3DtoSeed.pt");
  std::string pair_tag = "pair15";
  std::string brain_tag = "L73D766P4";
  std::string img_dir = "/lichtman/Felix/Lightsheet/P4/"+pair_tag+"/output_"+brain_tag+"/stitched/";
  std::string mask_dir = "/cajal/ACMUSERS/ziquanw/Lightsheet/roi_mask/"+pair_tag+"/"+brain_tag+"/";
  std::string h5fn = "/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/"+pair_tag+"/"+brain_tag+"/"+brain_tag+"_NIScpp_results.h5";
  std::string remapfn = "/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/"+pair_tag+"/"+brain_tag+"/"+brain_tag+"_remap.zip";
  auto allimgs = listdir_sorted(img_dir);
  auto allmasks = listdir_sorted(mask_dir);
  std::vector<std::string> img_fns;
  // std::vector<std::string> mask_fns;
  for (std::string file : allimgs ) {
    if (file.find("_C1_") != std::string::npos) {
      img_fns.push_back(file);
    }
  }
  // for (std::string file : allmasks ) {
  //   mask_fns.push_back(file);
  // }
  // if (mask_fns.size() != img_fns.size()) {exit(0);}
  std::cout<<"There are "<<img_fns.size()<<" .tif images\n";

  print_with_time("Initialize H5 database to store NIS results\n");
  std::vector<hsize_t> whole_brain_shape;
  torch::Tensor img0 = load_tif_as_tensor(img_fns[0]);
  whole_brain_shape.push_back(img_fns.size());
  whole_brain_shape.push_back(img0.size(0));
  whole_brain_shape.push_back(img0.size(1));
  std::vector<DataSet> dsetlist = init_h5data(h5fn, whole_brain_shape);
  hsize_t old_instance_n = 0;
  hsize_t old_contour_n = 0;
  hsize_t zmin = 0;
  torch::NoGradGuard no_grad;
  torch::Tensor masks;
  torch::Tensor pre_final_yx_flow;
  torch::Tensor pre_last_second;
  torch::Tensor pre_last_img;
  torch::Tensor pre_last_flow;
  torch::Tensor pre_last_mask;
  std::cout<<"Torch: no grad\n";
  // std::future<torch::Tensor> nis_saver;
  std::vector<torch::Tensor> remap_all;
  std::future<std::vector<torch::Tensor>> nis_obtainer;
  std::vector<torch::Tensor> nis_outputs;
  std::vector<torch::Tensor> gpu_outputs = gpu_process(
    0, 
    chunk_depth, 
    img_fns,
    // mask_fns,
    get_tile_param, 
    preproc, 
    nis_unet, 
    // postproc, 
    interpolater,
    grad_2d_to_3d,
    pre_final_yx_flow,
    pre_last_second,
    device
  );
  torch::Tensor flow3d = gpu_outputs[0];
  pre_final_yx_flow = gpu_outputs[1];
  pre_last_second = gpu_outputs[2];
  torch::Tensor first_img = gpu_outputs[3];
  torch::Tensor last_img = gpu_outputs[4];
  torch::Tensor first_flow = gpu_outputs[5];
  torch::Tensor last_flow = gpu_outputs[6];
  for (int64_t i = chunk_depth; i < img_fns.size(); i+=chunk_depth) {
    /*
    Follow the 3D flow to obtain NIS (CPU)
    */
    torch::Tensor dP = flow3d.index({torch::indexing::Slice(torch::indexing::None, 3), "..."});
    torch::Tensor cellprob = flow3d.index({3, "..."}); 
    torch::Tensor cp_mask = cellprob > cellprob_threshold; 
    if (cp_mask.any().item<bool>()) {
      nis_obtainer = std::async(std::launch::async, nis_obtain, flow_3DtoSeed, dP, cp_mask);
    } else {
      print_with_time("No instance, probability map is all zero, continue");
      // continue;
    }
    gpu_outputs = gpu_process(
      i, 
      chunk_depth, 
      img_fns,
      // mask_fns,
      get_tile_param, 
      preproc, 
      nis_unet, 
      // postproc, 
      interpolater,
      grad_2d_to_3d,
      pre_final_yx_flow,
      pre_last_second,
      device
    );
    if (i > chunk_depth){
      pre_last_mask = masks.index({-1, "..."});
    }
    if (cp_mask.any().item<bool>()) {
      nis_outputs = nis_obtainer.get();
      /*
      Save NIS results (IO) 
      */
      // Save the chunk to H5 database
      print_with_time("Save NIS results to H5 database\n");
      
      // if (i > chunk_depth){
      //   masks = nis_saver.get();
      // }
      hsize_t zmax = zmin + nis_outputs[0].size(0);
      // nis_saver = std::async(std::launch::async, save_h5data, dsetlist, nis_outputs, old_instance_n, old_contour_n, zmin, zmax, whole_brain_shape);
      masks = save_h5data(dsetlist, nis_outputs, old_instance_n, old_contour_n, zmin, zmax, whole_brain_shape);
      zmin += nis_outputs[0].size(0);
      old_instance_n += nis_outputs[2].size(0);
      old_contour_n += nis_outputs[1].size(0);
    }
    /*
    Run GNN to stitch the gap (GPU) 
    */
    if (i > chunk_depth){
      // Stitch the gap between two chunks
      print_with_time("Stitch the gap between two chunks\n");
      torch::Tensor tgt_img = pre_last_img;
      torch::Tensor tgt_mask = pre_last_mask;
      torch::Tensor tgt_flow = pre_last_flow;
      torch::Tensor src_img = first_img;
      torch::Tensor src_mask = masks.index({0, "..."});
      torch::Tensor src_flow = first_flow;
      torch::Tensor remap = gnn_stitch_gap(gnn_message_passing, gnn_classifier, tgt_img, tgt_mask, tgt_flow, src_img, src_mask, src_flow, device);
      remap_all.push_back(remap);
      auto bytes = torch::pickle_save(torch::cat(remap_all, -1)); //this is actually a std::vector of char
      std::ofstream fout(remapfn, std::ios::out | std::ios::binary);
      fout.write(bytes.data(), bytes.size());
      fout.close();
    }

    pre_last_img = last_img;
    pre_last_flow = last_flow;
    flow3d = gpu_outputs[0];
    pre_final_yx_flow = gpu_outputs[1];
    pre_last_second = gpu_outputs[2];
    first_img = gpu_outputs[3];
    last_img = gpu_outputs[4];
    first_flow = gpu_outputs[5];
    last_flow = gpu_outputs[6];
  }
  /*
  Follow the 3D flow to obtain NIS (CPU)
  */
  torch::Tensor dP = flow3d.index({torch::indexing::Slice(torch::indexing::None, 3), "..."});
  torch::Tensor cellprob = flow3d.index({3, "..."}); 
  torch::Tensor cp_mask = cellprob > cellprob_threshold; 
  if (cp_mask.any().item<bool>()) {
    nis_obtainer = std::async(std::launch::async, nis_obtain, flow_3DtoSeed, dP, cp_mask);
    nis_outputs = nis_obtainer.get();
    /*
    Save NIS results (IO) 
    */
    // Save the chunk to H5 database
    print_with_time("Save NIS results to H5 database\n");
    hsize_t zmax = zmin + nis_outputs[0].size(0);
    masks = save_h5data(dsetlist, nis_outputs, old_instance_n, old_contour_n, zmin, zmax, whole_brain_shape);
  } else {
    print_with_time("No instance, probability map is all zero, continue");
    // continue;
  }
  std::cout << "ok\n";
}
