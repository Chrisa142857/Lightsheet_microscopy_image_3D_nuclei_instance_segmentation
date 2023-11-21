#include "loop_unet.h"


torch::Tensor loop_unet(
    std::vector<std::string> img_fns, 
    std::vector<std::string> mask_fns, 
    torch::jit::script::Module get_tile_param,
    torch::jit::script::Module preproc,
    torch::jit::script::Module nis_unet,
    torch::jit::script::Module postproc,
    std::string device
  ) {
    /*
      Loop Unet for one chunk
    */
    std::vector<torch::Tensor> flow2d_list;
    int64_t batch_size = 200;
    int64_t file_loaded = 0;
    // Loop the list of files
    torch::Tensor img;
    std::future<torch::Tensor> img_loader;
    for (int64_t i=0; i<img_fns.size(); i++) {
        std::string img_path = img_fns[i];
        std::string mask_path = mask_fns[i];
        print_with_time("Processing Image " + img_path + " \n");
        torch::Tensor mask = load_tif_as_tensor(mask_path).to(device)[0];
        // Load data in background    
        if (file_loaded == 0) {
            img = load_tif_as_tensor(img_path).to(device);
        } else {
            img = img_loader.get().to(device);
        }
        if (file_loaded + 1 < img_fns.size()){
            img_loader = std::async(std::launch::async, load_tif_as_tensor, img_fns[file_loaded + 1]);
        }
        file_loaded++;
        
        // Get image info
        torch::IntArrayRef org_img_shape = img.sizes();
        // Pre-process image
        torch::Tensor area = torch::tensor(org_img_shape[1] * org_img_shape[2]);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(img);
        inputs.push_back(area);
        img = preproc(inputs).toTensor();
        std::vector<torch::Tensor> padded_outputs = pad_image(img);
        img = padded_outputs[0];
        torch::IntArrayRef img_shape = img.sizes();
        torch::Tensor pad_ysub = padded_outputs[1];
        torch::Tensor pad_xsub = padded_outputs[2];
        // Get tiler params
        torch::Tensor Ly = torch::tensor(img_shape[1]);
        torch::Tensor Lx = torch::tensor(img_shape[2]);
        torch::Tensor bsize = torch::tensor(224);
        torch::Tensor overlap = torch::tensor(0.1);
        inputs.clear();
        inputs.push_back(Ly);
        inputs.push_back(Lx);
        inputs.push_back(bsize);
        inputs.push_back(overlap);
        inputs.push_back(mask);
        auto tile_param = get_tile_param(inputs).toTuple();
        auto tile_ysub = tile_param->elements()[0].toTensor();
        auto tile_xsub = tile_param->elements()[1].toTensor();
        auto tile_idx = tile_param->elements()[2].toTensor().to(device);
        print_size(tile_idx);
        // Batching the image to input to Unet
        int64_t tile_num = tile_ysub.size(0);
        nis_unet.eval();
        nis_unet.to(device);
        std::vector<torch::Tensor> unet_inputs;
        torch::Tensor unet_inbatch;
        torch::Tensor yf = torch::zeros({tile_num, 3, 224, 224});
        torch::Tensor irange;
        for (int64_t i = 0; i < tile_idx.size(0); i++) {
            auto tile = img
                .slice(1, tile_ysub[tile_idx[i]][0].item<int64_t>(), tile_ysub[tile_idx[i]][1].item<int64_t>())
                .slice(2, tile_xsub[tile_idx[i]][0].item<int64_t>(), tile_xsub[tile_idx[i]][1].item<int64_t>());
            unet_inputs.push_back(tile);
            if ((i+1) % batch_size == 0){ 
                /*
                TODO
                 The last batch was skipped, Debug here.
                */
                unet_inbatch = torch::stack(unet_inputs, 0);
                unet_inputs.clear();
                std::cout << "Batch " << (i+1) / batch_size << "\t";
                print_size(unet_inbatch);
                inputs.clear();
                inputs.push_back(unet_inbatch);
                auto unet_outputs = nis_unet(inputs).toTuple();
                irange = tile_idx.slice(0, i+1-batch_size, i+1);
                torch::Tensor y = unet_outputs->elements()[0].toTensor().cpu();
                yf.index_put_({irange, "..."}, y);
            }
        }
        yf = yf.to(device);
        tile_ysub = tile_ysub.to(torch::kLong).to(device);
        tile_xsub = tile_xsub.to(torch::kLong).to(device);
        yf = average_tiles(yf, tile_ysub, tile_xsub, img_shape[1], img_shape[2]);
        yf = yf.slice(1, pad_ysub[0].item<int64_t>(), pad_ysub[-1].item<int64_t>())
               .slice(2, pad_xsub[0].item<int64_t>(), pad_xsub[-1].item<int64_t>());
               
        inputs.clear();
        inputs.push_back(yf);
        yf = postproc(inputs).toTensor();
        flow2d_list.push_back(yf.detach().cpu());
    }
    torch::Tensor flow2d = torch::stack(flow2d_list, 0);
    return flow2d;
}


torch::Tensor average_tiles(torch::Tensor y, torch::Tensor ysub, torch::Tensor xsub, int64_t Ly, int64_t Lx) {
    torch::Tensor Navg = torch::zeros({Ly, Lx}).to(y.device());
    torch::Tensor yf = torch::zeros({y.size(1), Ly, Lx}).to(y.device());
    
    // Taper edges of tiles
    torch::Tensor mask = _taper_mask(y.device(), y.size(-2), 7.5);
    
    for (int64_t j = 0; j < ysub.size(0); ++j) {
        int64_t ys = ysub.index({j, 0}).item<int64_t>();
        int64_t ye = ysub.index({j, 1}).item<int64_t>();
        int64_t xs = xsub.index({j, 0}).item<int64_t>();
        int64_t xe = xsub.index({j, 1}).item<int64_t>();
        torch::Tensor yf_add = yf.index({torch::indexing::Slice(), torch::indexing::Slice(ys, ye), torch::indexing::Slice(xs, xe)}) + (y[j] * mask);
        yf.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(ys, ye), torch::indexing::Slice(xs, xe)},
            yf_add
        );
        torch::Tensor Navg_add = Navg.index({torch::indexing::Slice(ys, ye), torch::indexing::Slice(xs, xe)}) + mask;
        Navg.index_put_(
            {torch::indexing::Slice(ys, ye), torch::indexing::Slice(xs, xe)},
            Navg_add
        );
    }

    yf = yf / Navg;
    return yf;
}

torch::Tensor _taper_mask(torch::Device device, int64_t bsize, float sig) {
    torch::Tensor xm = torch::arange(bsize, torch::device(device).dtype(torch::kFloat));
    xm = torch::abs(xm - xm.mean());
    
    torch::Tensor mask = 1 / (1 + torch::exp((xm - (bsize / 2 - 20)) / sig));
    mask = mask * mask.unsqueeze(1);
    
    return mask;
}