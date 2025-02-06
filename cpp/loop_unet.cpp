#include "loop_unet.h"


std::vector<torch::Tensor> loop_unet(
    std::vector<std::string> img_fns, 
    // torch::jit::script::Module* get_tile_param,
    // torch::jit::script::Module* preproc,
    torch::jit::script::Module* nis_unet,
    bool do_fg_filter,
    std::string device,
    int64_t batch_size,
    std::string lefttop_fn,
    std::string righttop_fn,
    std::string leftbottom_fn,
    std::string rightbottom_fn
  ) {
    torch::NoGradGuard no_grad;
    namespace F = torch::nn::functional;
    int64_t bsize = 224;
    /*
      Loop Unet for one chunk
    */
    // preproc->to(device);
    // get_tile_param->to(device);
    std::vector<torch::Tensor> flow2d_list;
    // int64_t batch_size = 400;
    int64_t file_loaded = 0;
    
    torch::Tensor lefttop_img, righttop_img, leftbottom_img, rightbottom_img;
    // Loop the list of files
    torch::Tensor img;
    torch::Tensor first_img;
    torch::Tensor last_img;
    std::future<torch::Tensor> img_loader;
    for (int64_t i=0; i<img_fns.size(); i++) {
        std::string img_path = img_fns[i];
        print_with_time("Processing Image " + img_path + " \n");
        // Load corner in the slice to set fg threshold
        float given_fg_thres, fg_thres;
        if (lefttop_fn != "") {
            std::string cur_lefttopfn = lefttop_fn;
            std::string cur_righttopfn = righttop_fn;
            std::string cur_leftbottomfn = leftbottom_fn;
            std::string cur_rightbottomfn = rightbottom_fn;
            cur_lefttopfn = replaceWithFormattedNumbers(cur_lefttopfn, split_then_int(getFilename(img_path), 1), split_then_int(getFilename(lefttop_fn), 1), "_");
            cur_lefttopfn = replaceWithFormattedNumbers(cur_lefttopfn, split_then_int(getFilename(img_path), 1), split_then_int(getFilename(lefttop_fn), 1), "Z");
            cur_righttopfn = replaceWithFormattedNumbers(cur_righttopfn, split_then_int(getFilename(img_path), 1), split_then_int(getFilename(righttop_fn), 1), "_");
            cur_righttopfn = replaceWithFormattedNumbers(cur_righttopfn, split_then_int(getFilename(img_path), 1), split_then_int(getFilename(righttop_fn), 1), "Z");
            cur_leftbottomfn = replaceWithFormattedNumbers(cur_leftbottomfn, split_then_int(getFilename(img_path), 1), split_then_int(getFilename(leftbottom_fn), 1), "_");
            cur_leftbottomfn = replaceWithFormattedNumbers(cur_leftbottomfn, split_then_int(getFilename(img_path), 1), split_then_int(getFilename(leftbottom_fn), 1), "Z");
            cur_rightbottomfn = replaceWithFormattedNumbers(cur_rightbottomfn, split_then_int(getFilename(img_path), 1), split_then_int(getFilename(rightbottom_fn), 1), "_");
            cur_rightbottomfn = replaceWithFormattedNumbers(cur_rightbottomfn, split_then_int(getFilename(img_path), 1), split_then_int(getFilename(rightbottom_fn), 1), "Z");
            lefttop_img = load_tif_as_tensor(cur_lefttopfn).to(device);
            righttop_img = load_tif_as_tensor(cur_righttopfn).to(device);
            leftbottom_img = load_tif_as_tensor(cur_leftbottomfn).to(device);
            rightbottom_img = load_tif_as_tensor(cur_rightbottomfn).to(device);
            given_fg_thres = std::min({
                lefttop_img.slice(1, 0, bsize).slice(2, 0, bsize).max().item<float>(),
                lefttop_img.slice(1, -bsize, -1).slice(2, 0, bsize).max().item<float>(),
                lefttop_img.slice(1, -bsize, -1).slice(2, -bsize, -1).max().item<float>(),
                lefttop_img.slice(1, 0, bsize).slice(2, -bsize, -1).max().item<float>(),
                righttop_img.slice(1, 0, bsize).slice(2, 0, bsize).max().item<float>(),
                righttop_img.slice(1, -bsize, -1).slice(2, 0, bsize).max().item<float>(),
                righttop_img.slice(1, -bsize, -1).slice(2, -bsize, -1).max().item<float>(),
                righttop_img.slice(1, 0, bsize).slice(2, -bsize, -1).max().item<float>(),
                leftbottom_img.slice(1, 0, bsize).slice(2, 0, bsize).max().item<float>(),
                leftbottom_img.slice(1, -bsize, -1).slice(2, 0, bsize).max().item<float>(),
                leftbottom_img.slice(1, -bsize, -1).slice(2, -bsize, -1).max().item<float>(),
                leftbottom_img.slice(1, 0, bsize).slice(2, -bsize, -1).max().item<float>(),
                rightbottom_img.slice(1, 0, bsize).slice(2, 0, bsize).max().item<float>(),
                rightbottom_img.slice(1, -bsize, -1).slice(2, 0, bsize).max().item<float>(),
                rightbottom_img.slice(1, -bsize, -1).slice(2, -bsize, -1).max().item<float>(),
                rightbottom_img.slice(1, 0, bsize).slice(2, -bsize, -1).max().item<float>(),
            });
        }

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
        std::vector<int64_t> org_img_shape;
        org_img_shape.push_back(img.size(0));
        org_img_shape.push_back(img.size(1));
        org_img_shape.push_back(img.size(2));
        if (i==0){
            first_img = img[0].detach().cpu().clone();
        }
        if ((i+1)==img_fns.size()) {
            last_img = img[0].detach().cpu().clone();
        }
        // Pre-process image
        // img = preproc_image(img);
        img = torch::cat({img, torch::zeros_like(img)});
        img = F::interpolate(
            img.unsqueeze(0), 
            F::InterpolateFuncOptions().scale_factor(std::vector<double>({1.437293, 1.437293})).mode(torch::kBilinear).align_corners(false).recompute_scale_factor(true)
        );
        img = img[0];
        // torch::Tensor area = torch::tensor(org_img_shape[1] * org_img_shape[2]);
        std::vector<torch::jit::IValue> inputs;
        // inputs.push_back(img);
        // inputs.push_back(area);
        // img = preproc->forward(inputs).toTensor();
        std::vector<torch::Tensor> padded_outputs = pad_image(img);
        img = padded_outputs[0];
        
        torch::IntArrayRef img_shape = img.sizes();
        torch::Tensor pad_ysub = padded_outputs[1];
        torch::Tensor pad_xsub = padded_outputs[2];
        // Get tiler params
        torch::Tensor Ly = torch::tensor(img_shape[1]);
        torch::Tensor Lx = torch::tensor(img_shape[2]);
        // torch::Tensor bsize = torch::tensor(224);
        torch::Tensor overlap = torch::tensor(0.1);
        // inputs.clear();
        std::vector<torch::Tensor> tile_param = tile_image(img, bsize, overlap);
        auto tile_ysub = tile_param[0];
        auto tile_xsub = tile_param[1];
        // print_size(img);
        // print_size(tile_ysub);
        // Batching the image to input to Unet
        int64_t tile_num = tile_ysub.size(0);
        nis_unet->eval();
        nis_unet->to(device);
        std::vector<torch::Tensor> unet_inputs;
        torch::Tensor unet_inbatch;
        torch::Tensor yf = torch::zeros({tile_num, 3, bsize, bsize});
        std::vector<int64_t> irange;
        int64_t j;
        int64_t bi = 0;
        if (lefttop_fn != "") {
            fg_thres = given_fg_thres;
        }
        else if (do_fg_filter) {
            fg_thres = std::min({
                img.slice(1, 0, bsize).slice(2, 0, bsize).max().item<float>(),
                img.slice(1, -bsize, -1).slice(2, 0, bsize).max().item<float>(),
                img.slice(1, -bsize, -1).slice(2, -bsize, -1).max().item<float>(),
                img.slice(1, 0, bsize).slice(2, -bsize, -1).max().item<float>()});
        }
        for (j = 0; j < tile_ysub.size(0); j++) {
            auto tile = img
                .slice(1, tile_ysub[j][0].item<int64_t>(), tile_ysub[j][1].item<int64_t>())
                .slice(2, tile_xsub[j][0].item<int64_t>(), tile_xsub[j][1].item<int64_t>());
            // if (tile.max().item<float>() > fg_thres){
            if (std::get<0>(tile.max(-1)).mean().item<float>() > fg_thres || !do_fg_filter) {
                unet_inputs.push_back(tile);
                irange.push_back(j);
            }
            if (unet_inputs.size() >= batch_size){ 
                bi += 1;
                unet_inbatch = torch::stack(unet_inputs, 0);
                std::cout << "Batch " << bi << "\t";
                unet_inbatch.index_put_({torch::indexing::Slice(), 0}, normalize_image(unet_inbatch.index({torch::indexing::Slice(), 0})));
                unet_inputs.clear();
                inputs.clear();
                inputs.push_back(unet_inbatch);
                auto unet_outputs = nis_unet->forward(inputs).toTuple();
                torch::Tensor y = unet_outputs->elements()[0].toTensor().cpu();
                yf.index_put_({torch::tensor(irange), "..."}, y);
                irange.clear();
            }
        }
        /*
            The last batch was skipped, compute here.
        */
        if (unet_inputs.size() > 0){ 
            unet_inbatch = torch::stack(unet_inputs, 0);
            std::cout << "Batch " << bi + 1 << "\t";
            unet_inbatch.index_put_({torch::indexing::Slice(), 0}, normalize_image(unet_inbatch.index({torch::indexing::Slice(), 0})));
            print_size(unet_inbatch);
            inputs.clear();
            inputs.push_back(unet_inbatch);
            auto unet_outputs = nis_unet->forward(inputs).toTuple();
            torch::Tensor y = unet_outputs->elements()[0].toTensor().cpu();
            yf.index_put_({torch::tensor(irange), "..."}, y);
        }
        /*
            The last batch was computed.
        */
        yf = yf.to(device);
        tile_ysub = tile_ysub.to(torch::kLong).to(device);
        tile_xsub = tile_xsub.to(torch::kLong).to(device);
        yf = average_tiles(yf, tile_ysub, tile_xsub, img_shape[1], img_shape[2]);
        yf = yf.slice(1, 0, img_shape[1])
               .slice(2, 0, img_shape[2]);
        yf = yf.slice(1, pad_ysub[0].item<int64_t>(), pad_ysub[-1].item<int64_t>()+1)
               .slice(2, pad_xsub[0].item<int64_t>(), pad_xsub[-1].item<int64_t>()+1);
               
        yf = F::interpolate(
            yf.unsqueeze(0), 
            F::InterpolateFuncOptions().size(std::vector<int64_t>({org_img_shape[1], org_img_shape[2]})).mode(torch::kBilinear).align_corners(false)
        );
        yf = torch::permute(yf[0], {1, 2, 0});
        flow2d_list.push_back(yf.detach().cpu());
    }
    torch::Tensor flow2d = torch::stack(flow2d_list, 0);
    std::vector<torch::Tensor> output;
    output.push_back(flow2d);
    output.push_back(first_img);
    output.push_back(last_img);
    return output;
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

std::vector<torch::Tensor> tile_image(torch::Tensor image, int64_t bsize, torch::Tensor overlap) {
    auto Ly = image.size(1);
    auto Lx = image.size(2);
    auto tile_overlap = overlap.clip(0.05, 0.5);
    auto bsizeY = bsize;
    auto bsizeX = bsize;
    // tiles overlap by 10% tile size
    int ny = torch::ceil((1.+2*tile_overlap) * Ly / bsize).to(torch::kLong).item<int>();
    int nx = torch::ceil((1.+2*tile_overlap) * Lx / bsize).to(torch::kLong).item<int>();
    auto ystart = torch::linspace(0, Ly-bsizeY, ny).to(torch::kLong);
    auto xstart = torch::linspace(0, Lx-bsizeX, nx).to(torch::kLong);
    ny = ystart.size(0);
    nx = xstart.size(0);
    auto ystarts = ystart.unsqueeze(1).repeat({1,nx}).reshape(-1);
    auto xstarts = xstart.repeat({ny});
    auto yends = ystarts + bsizeY;
    auto xends = xstarts + bsizeX;
    auto ysub = torch::stack({ystarts, yends}, -1);
    auto xsub = torch::stack({xstarts, xends}, -1);
    std::vector<torch::Tensor> output;
    output.push_back(ysub);
    output.push_back(xsub);
    return output;
}