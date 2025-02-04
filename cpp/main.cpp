#include <torch/script.h> // One-stop header.
#include <torch/nn/functional.h>
#include <future>

// #include <iostream>
#include <memory>
#include <string> 

#include "loop_unet.h"
#include "utils.h"
#include "flow_op.h"
#include "image_reader.h"
#include "gnn_stitch_gap.h"

#include "argparser.hpp"

void save_tensor(torch::Tensor tensor, std::string fn){
  print_with_time("Save tensor as .zip: ");
  print_size(tensor);
  auto bytes = torch::pickle_save(tensor); //this is actually a std::vector of char
  std::ofstream fout(fn, std::ios::out | std::ios::binary);
  fout.write(bytes.data(), bytes.size());
  fout.close();
}

std::vector<torch::Tensor> nis_obtain(torch::jit::script::Module flow_3DtoSeed, torch::Tensor flow3d, float cellprob_threshold, int64_t ilabel, std::string savefn){
  torch::Tensor p = index_flow(
    flow3d.index({torch::indexing::Slice(torch::indexing::None, 3), "..."}) * (flow3d.index({3, "..."})>cellprob_threshold) / 5., 
    flow3d.size(1), 
    flow3d.size(2), 
    flow3d.size(3), 139);
  std::vector<torch::Tensor> nis_outputs = flow_3DtoNIS(
    // meshgrider,
    flow_3DtoSeed,
    p, 
    (flow3d.index({3, "..."})>cellprob_threshold), 
    ilabel, 20
    );
  if (nis_outputs.size()>1) {
    std::vector<int64_t> seg_size = {nis_outputs[0].size(0), nis_outputs[0].size(1), nis_outputs[0].size(2)};
    save_tensor(torch::from_blob(seg_size.data(), seg_size.size(), torch::kLong), savefn+"_seg_meta.zip");
    save_tensor(nis_outputs[0].to(torch::kBool), savefn+"_binary_mask.zip");
    save_tensor(nis_outputs[1], savefn+"_instance_center.zip");
    save_tensor(nis_outputs[2], savefn+"_instance_coordinate.zip");
    save_tensor(nis_outputs[3], savefn+"_instance_label.zip");
    // torch::Tensor vols = nis_outputs[0].reshape(-1).bincount();
    // save_tensor(vols, savefn+"_instance_volume.zip");
    save_tensor(nis_outputs[4], savefn+"_instance_volume.zip");
  }
  return nis_outputs;
}

std::vector<torch::Tensor> gpu_process(
  int64_t i, 
  int64_t chunk_depth, 
  std::vector<std::string> img_fns,
  // torch::jit::script::Module get_tile_param, 
  // torch::jit::script::Module preproc, 
  torch::jit::script::Module* nis_unet, 
  bool do_fg_filter,
  // torch::jit::script::Module interpolater,
  torch::jit::script::Module grad_2d_to_3d,
  torch::Tensor pre_final_yx_flow,
  torch::Tensor pre_last_second,
  std::string device,
  std::vector<std::string> bg_img_fns = {}
) {
  namespace F = torch::nn::functional;
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
  std::vector<torch::Tensor> unet_output;
  if (bg_img_fns.size() != 4){
    unet_output = loop_unet(
      img_onechunk, 
      // &get_tile_param,
      // &preproc,
      nis_unet,
      do_fg_filter,
      device
    );
  } else {
    unet_output = loop_unet(
      img_onechunk, 
      // &get_tile_param,
      // &preproc,
      nis_unet,
      do_fg_filter,
      device,
      bg_img_fns[0],
      bg_img_fns[1],
      bg_img_fns[2],
      bg_img_fns[3]
    );
  }
  torch::Tensor flow2d = unet_output[0];
  torch::Tensor first_img = unet_output[1];
  torch::Tensor last_img = unet_output[2];
  /*
  Resample probability map to train-data resolution (GPU)
    train_resolution = (2.5, .75, .75)
    input_resolution = (4, .75, .75)
  */
  flow2d = flow2d.permute({3, 0, 1 ,2});
  flow2d = F::interpolate(flow2d.unsqueeze(0),
    F::InterpolateFuncOptions().scale_factor(std::vector<double>({4/2.5, 1, 1})).mode(torch::kNearestExact).align_corners(false).recompute_scale_factor(false)).squeeze();
  // std::vector<torch::jit::IValue> inter_inputs({flow2d});
  // flow2d = interpolater(inter_inputs).toTensor();
  /*
  2D flow to 3D flow (GPU)
  */
  if (i > 0) {
    flow2d = torch::cat({pre_final_yx_flow.unsqueeze(1), flow2d}, 1);
  }
  pre_final_yx_flow = flow2d.index({torch::indexing::Slice(torch::indexing::None, 3, 1), -1}).detach().clone();
  pre_last_second = flow2d.index({torch::indexing::Slice(torch::indexing::None, 2, 1), -2}).detach().clone();
  bool is_first_chunk = i == 0;

  torch::Tensor flow3d = flow_2Dto3D(flow2d, pre_last_second, &grad_2d_to_3d, device, is_first_chunk);
  torch::Tensor first_flow = flow3d.index({torch::indexing::Slice(), 0, "..."}).detach().clone();
  torch::Tensor last_flow = flow3d.index({torch::indexing::Slice(), -1, "..."}).detach().clone();
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

std::vector<torch::Tensor> stitch_process(
  std::vector<torch::Tensor> remap_all,
  torch::jit::script::Module* gnn_message_passing, 
  torch::jit::script::Module* gnn_classifier, 
  torch::Tensor pre_last_img,
  torch::Tensor pre_last_mask,
  torch::Tensor pre_last_flow,
  torch::Tensor first_img,
  torch::Tensor first_mask,
  torch::Tensor first_flow,
  std::string remapfn,
  std::string device
) {
    // Stitch the gap between two chunks
    print_with_time("Stitch the gap between two chunks\n");
    torch::Tensor tgt_img = pre_last_img;
    torch::Tensor tgt_mask = pre_last_mask;
    torch::Tensor tgt_flow = pre_last_flow;
    torch::Tensor src_img = first_img;
    torch::Tensor src_mask = first_mask;
    torch::Tensor src_flow = first_flow;
    torch::Tensor remap = gnn_stitch_gap(gnn_message_passing, gnn_classifier, tgt_img, tgt_mask, tgt_flow, src_img, src_mask, src_flow, device);
    if (remap.size(0)==0) {return remap_all;}
    remap_all.push_back(remap);
    save_tensor(torch::cat(remap_all, -1), remapfn);
    return remap_all;
}

int main(int argc, const char* argv[]) {
  std::cout << "LibTorch version: "
    << TORCH_VERSION_MAJOR << "."
    << TORCH_VERSION_MINOR << "."
    << TORCH_VERSION_PATCH << std::endl;

  argparse::ArgumentParser program("main");
  program.add_argument("--device")
    .help("GPU ID")
    .required()
    .default_value(std::string("cuda:0")); 
  program.add_argument("--chunk_depth")
    .help("chunk_depth")
    .required()
    .default_value(std::string("600")); 
  program.add_argument("--cellprob_threshold")
    .help("cellprob_threshold")
    .required()
    .default_value(std::string("0.1")); 
  program.add_argument("-mroot", "--model_root")
    .help("path to .pt models")
    .required()
    .default_value(std::string("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource")); 
  program.add_argument("-ptag", "--pair_tag")
    .help("pair_tag")
    .required()
    .default_value(std::string("4")); 
  program.add_argument("-btag", "--brain_tag")
    .help("brain_tag")
    .required()
    .default_value(std::string("100")); 
  program.add_argument("-in", "--data_root")
    .help("specify the data_root.")
    .required()
    .default_value(std::string("./outputs/"));
  program.add_argument("-out", "--save_root")
    .help("specify the save_root")
    .required()
    .default_value(std::string("")); 
  program.add_argument("--lt_corner_image_path")
    .help("left_top_corner_image_path for foreground detection")
    .required()
    .default_value(std::string("None")); 
  program.add_argument("--rt_corner_image_path")
    .help("right_top_corner_image_path for foreground detection")
    .required()
    .default_value(std::string("None")); 
  program.add_argument("--lb_corner_image_path")
    .help("left_bottom_corner_image_path for foreground detection")
    .required()
    .default_value(std::string("None")); 
  program.add_argument("--rb_corner_image_path")
    .help("right_bottom_corner_image_path for foreground detection")
    .required()
    .default_value(std::string("None")); 
  program.add_argument("-no_fg_det", "--no_foreground_detection")
    .help("skip foreground_detection")
    .default_value(false)
    .implicit_value(true); 
  
  try {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cerr << "Error throwed in argparse" << std::endl;
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  std::string device = program.get<std::string>("--device");
  int64_t chunk_depth = stoi(program.get<std::string>("--chunk_depth"));
  float cellprob_threshold = stof(program.get<std::string>("--cellprob_threshold"));

  // torch::jit::script::Module get_tile_param;
  // torch::jit::script::Module preproc;
  torch::jit::script::Module nis_unet;
  // torch::jit::script::Module postproc;
  torch::jit::script::Module grad_2d_to_3d;
  // torch::jit::script::Module interpolater;
  torch::jit::script::Module gnn_message_passing;
  torch::jit::script::Module gnn_classifier;
  torch::jit::script::Module flow_3DtoSeed;
  // get_tile_param = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/get_model_tileparam_cpu.pt");
  // preproc = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/preproc_img1xLyxLx_"+std::string(argv[3])+".pt");
  nis_unet = torch::jit::load(program.get<std::string>("--model_root")+"/nis_unet_cpu.pt");
  grad_2d_to_3d = torch::jit::load(program.get<std::string>("--model_root")+"/grad_2Dto3D_"+device+".pt");
  // interpolater = torch::jit::load("/ram/USERS/ziquanw/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/downloads/resource/interpolate_ratio_1.6x1x1.pt");
  gnn_message_passing = torch::jit::load(program.get<std::string>("--model_root")+"/gnn_message_passing_"+device+".pt");
  gnn_classifier = torch::jit::load(program.get<std::string>("--model_root")+"/gnn_classifier_"+device+".pt");
  flow_3DtoSeed = torch::jit::load(program.get<std::string>("--model_root")+"/flow_3DtoSeed.pt");
  // std::string pair_tag = "pair15";
  // std::string brain_tag = "L73D766P4";
  std::string pair_tag = program.get<std::string>("--pair_tag");
  std::string brain_tag = program.get<std::string>("--brain_tag");
  std::string data_root = program.get<std::string>("--data_root");
  std::string save_root = program.get<std::string>("--save_root");
  bool do_fg_filter = !program.get<bool>("--no_foreground_detection");
  std::vector<std::string> bg_img_fns = {};
  if (do_fg_filter){
    bg_img_fns.push_back(program.get<std::string>("--lt_corner_image_path"));
    bg_img_fns.push_back(program.get<std::string>("--rt_corner_image_path"));
    bg_img_fns.push_back(program.get<std::string>("--lb_corner_image_path"));
    bg_img_fns.push_back(program.get<std::string>("--rb_corner_image_path"));
  }
  // std::string img_dir = "/"+data_root+"/Felix/Lightsheet/P4/"+pair_tag+"/output_"+brain_tag+"/stitched/";
  // std::string img_dir = data_root+pair_tag+"/output_"+brain_tag+"/stitched/";
  std::string img_dir = data_root;
  // std::string mask_dir = "/cajal/ACMUSERS/ziquanw/Lightsheet/roi_mask/"+pair_tag+"/"+brain_tag+"/";
  // std::string savefn_prefix = "/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/"+pair_tag+"/"+brain_tag+"/"+brain_tag+"_NIScpp_results";
  // std::string remapfn = "/cajal/ACMUSERS/ziquanw/Lightsheet/results/P4/"+pair_tag+"/"+brain_tag+"/"+brain_tag+"_remap.zip";
  std::string savefn_prefix = save_root+brain_tag+"_NIScpp_results";
  std::string remapfn = save_root+brain_tag+"_remap.zip";
  auto allimgs = listdir_sorted(img_dir);
  // auto allmasks = listdir_sorted(mask_dir);
  std::vector<std::string> img_fns;
  // std::vector<std::string> mask_fns;
  for (std::string file : allimgs ) {
    if (file.find("_C1_") != std::string::npos) {
      img_fns.push_back(file);
    }
  }
  std::cout<<"There are "<<img_fns.size()<<" .tif images";

  // print_with_time("Initialize H5 database to store NIS results\n");
  std::vector<int> whole_brain_shape;
  torch::Tensor img0 = load_tif_as_tensor(img_fns[0]);
  whole_brain_shape.push_back(img_fns.size());
  whole_brain_shape.push_back(img0.size(1));
  whole_brain_shape.push_back(img0.size(2));
  std::cout<<", whole brain shape is "<<whole_brain_shape<<"\n";
  int64_t old_instance_n = 0;
  // int old_contour_n = 0;
  int zmin = 0;
  torch::NoGradGuard no_grad;
  torch::Tensor first_mask;
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
    // get_tile_param, 
    // preproc, 
    &nis_unet, 
    do_fg_filter,
    // interpolater,
    grad_2d_to_3d,
    pre_final_yx_flow,
    pre_last_second,
    device, bg_img_fns
  );
  // torch::Tensor flow3d = gpu_outputs[0];
  pre_final_yx_flow = gpu_outputs[1];
  pre_last_second = gpu_outputs[2];
  torch::Tensor first_img = gpu_outputs[3];
  torch::Tensor last_img = gpu_outputs[4];
  torch::Tensor first_flow = gpu_outputs[5];
  torch::Tensor last_flow = gpu_outputs[6];
  // torch::Tensor masks;
  std::vector<torch::Tensor> last_first_masks;
  int zmax;
  bool pre_has_nis;
  bool cur_has_nis;
  for (int64_t i = chunk_depth; i < img_fns.size(); i+=chunk_depth) {
    /*
    Follow the 3D flow to obtain NIS (CPU)
    */
    zmax = zmin + gpu_outputs[0].size(1);
    nis_obtainer = std::async(std::launch::async, nis_obtain, flow_3DtoSeed, gpu_outputs[0], cellprob_threshold, old_instance_n, savefn_prefix+"_zmin"+std::to_string(zmin));
    gpu_outputs.clear();
    if (i > chunk_depth & last_first_masks.size() > 0){
      pre_last_mask = last_first_masks[0];
      pre_has_nis = true;
    } else {
      pre_has_nis = false;
    }
    nis_outputs = nis_obtainer.get();
    /*
    Get last and first slice of output
    */
    if (nis_outputs.size() > 1){
      print_with_time("zmin: ");
      std::cout<<zmin<<", zmax: "<<zmax<<"\n";
      print_with_time("whole_brain_shape: ");
      std::cout<<whole_brain_shape<<"\n";
      last_first_masks.clear();
      last_first_masks.push_back(nis_outputs[0].index({-1, "..."}).detach().clone());
      last_first_masks.push_back(nis_outputs[0].index({0, "..."}).detach().clone());
      first_mask = last_first_masks[1];
      old_instance_n = nis_outputs[5].item<int64_t>();
      // old_contour_n += nis_outputs[2].size(0);
      cur_has_nis = true;
    } else {
      last_first_masks.clear();
      cur_has_nis = false;
    }
    nis_outputs.clear();
    // }
    /*
    Run GNN to stitch the gap (GPU) 
    */
    if (i > chunk_depth & pre_has_nis & cur_has_nis){
      remap_all = stitch_process(
        remap_all,
        &gnn_message_passing, 
        &gnn_classifier, 
        pre_last_img,
        pre_last_mask,
        pre_last_flow,
        first_img,
        first_mask,
        first_flow,
        remapfn,
        device
      );
    }

    gpu_outputs = gpu_process(
      i, 
      chunk_depth, 
      img_fns,
      // get_tile_param, 
      // preproc, 
      &nis_unet, 
      do_fg_filter,
      // interpolater,
      grad_2d_to_3d,
      pre_final_yx_flow,
      pre_last_second,
      device, bg_img_fns
    );
    pre_last_img = last_img;
    pre_last_flow = last_flow;
    // flow3d = gpu_outputs[0];
    pre_final_yx_flow = gpu_outputs[1];
    pre_last_second = gpu_outputs[2];
    first_img = gpu_outputs[3];
    last_img = gpu_outputs[4];
    first_flow = gpu_outputs[5];
    last_flow = gpu_outputs[6];
    zmin = zmax;
  }

  nis_obtainer = std::async(std::launch::async, nis_obtain, flow_3DtoSeed, gpu_outputs[0], cellprob_threshold, old_instance_n, savefn_prefix+"_zmin"+std::to_string(zmin));
  nis_outputs = nis_obtainer.get();
  if (last_first_masks.size() > 0){
    pre_last_mask = last_first_masks[0];
    pre_has_nis = true;
  } else {
    pre_has_nis = false;
  }
  /*
  Get last and first slice of output
  */
  if (nis_outputs.size() > 1){
    print_with_time("zmin: ");
    std::cout<<zmin<<", zmax: "<<zmax<<"\n";
    print_with_time("whole_brain_shape: ");
    std::cout<<whole_brain_shape<<"\n";
    first_mask = nis_outputs[0].index({0, "..."}).detach().clone();
    cur_has_nis = true;
  } else {
    cur_has_nis = false;
  }
  nis_outputs.clear();
  // }
  /*
  Run GNN to stitch the gap (GPU) 
  */
  if (pre_has_nis & cur_has_nis){
    remap_all = stitch_process(
      remap_all,
      &gnn_message_passing, 
      &gnn_classifier, 
      pre_last_img,
      pre_last_mask,
      pre_last_flow,
      first_img,
      first_mask,
      first_flow,
      remapfn,
      device
    );
  }

  std::cout << "ok\n";
}
