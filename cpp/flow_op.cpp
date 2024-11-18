#include "flow_op.h"

std::vector<torch::Tensor> get_large_fg_coord(torch::Tensor seg) {
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    auto seg_shape = seg.sizes();
    std::vector<torch::Tensor> meshgrid_tensors;

    for (int dimi = 0; dimi < seg_shape.size(); ++dimi) {
        meshgrid_tensors.push_back(torch::arange(seg_shape[dimi], options));
    }

    print_with_time("Start meshgrid");
    auto meshgrid_output = torch::meshgrid(meshgrid_tensors);
    torch::Tensor meshgrid_coord;
    print_with_time("Done meshgrid, start mask seg fg");
    torch::Tensor nis_mask = seg > 0;
    meshgrid_coord = torch::stack({
        meshgrid_output[0].masked_select(nis_mask),
        meshgrid_output[1].masked_select(nis_mask),
        meshgrid_output[2].masked_select(nis_mask)}, -1).to(torch::kInt16);
    torch::Tensor seg_fg = seg.masked_select(nis_mask);
    print_with_time("Done mask, start unique seg fg to get volume");
    auto unique_output = at::_unique2(seg_fg, true, false, true);
    torch::Tensor label = std::get<0>(unique_output);
    torch::Tensor vol = std::get<2>(unique_output);
    print_with_time("Done unique, start resort seg fg to group nis coordinate");

    auto sorted_nis = seg_fg.argsort();
    meshgrid_coord = meshgrid_coord.index_select(0, sorted_nis); // vol*N x 3
    // print_with_time("Done resort, start loop nis to get center of coordinate");
    print_with_time("Done resort, start mask big nis");

    torch::Tensor big_nis_mask = vol > 1000;
    label = label.masked_select(torch::logical_not(big_nis_mask));
    torch::Tensor splits = vol.cumsum(0);
    torch::Tensor big_ind = torch::where(big_nis_mask)[0];
    torch::Tensor mask_pt = torch::ones({meshgrid_coord.size(0)}, torch::kBool);
    for ( int bi=0; bi < big_ind.sizes()[0]; bi++){
        torch::Tensor big_pt_ind = torch::arange((splits[big_ind[bi]]-vol[big_ind[bi]]).item(), splits[big_ind[bi]].item());
        mask_pt.index_put_({big_pt_ind}, false);
    };
    meshgrid_coord = meshgrid_coord.index({mask_pt, torch::indexing::Slice()});
    vol = vol.masked_select(torch::logical_not(big_nis_mask));
    splits = vol.cumsum(0);
    print_with_time("Done mask big nis, start pad_sequence(pt) to get center of coordinate by median(pt)");
    
    auto pt_splits = torch::tensor_split(meshgrid_coord, splits); // [vol x 3,...]xN
    pt_splits.pop_back();
    torch::Tensor padded_pt = torch::nn::utils::rnn::pad_sequence(pt_splits, false); // max_vol x N x 3
    torch::Tensor ct = padded_pt[0].to(torch::kInt16); // N x 3

    print_with_time("Done get center, out to save tensors");
    std::vector<torch::Tensor> outputs;
    outputs.push_back(ct);
    outputs.push_back(meshgrid_coord);
    outputs.push_back(label);
    outputs.push_back(vol);
    return outputs;
}

torch::Tensor flow_2Dto3D(
    torch::Tensor flow_2d,  // [3 x Z x Y x X]
    torch::Tensor pre_last_second, 
    torch::jit::script::Module* sim_grad_z,
    std::string device,
    bool skip_first
    ) {
    sim_grad_z->to(device);
    flow_2d = torch::cat({torch::zeros_like(flow_2d[0]).unsqueeze(0), flow_2d}); // [4 x Z x Y x X]

    for (int64_t i = 0; i < flow_2d.size(1) - 1; ++i) {
        torch::Tensor pre_yx_flow;

        if (i > 0) {
            pre_yx_flow = flow_2d.index({torch::indexing::Slice(1, 3), i - 1});
        } else {
            pre_yx_flow = pre_last_second;
        }
        if (i == 0 & skip_first) {continue;}
        std::vector<torch::jit::IValue> inputs({flow_2d.index({3, i}).to(device), pre_yx_flow.to(device), flow_2d.index({torch::indexing::Slice(1, 3), i + 1}).to(device)});
        flow_2d.index_put_({0, i}, sim_grad_z->forward(inputs).toTensor().cpu());
    }
    if (skip_first){
        flow_2d = flow_2d.slice(1, 1, -1); // [4 x Z-2 x Y x X]
    } else {
        flow_2d = flow_2d.slice(1, 0, -1); // [4 x Z-1 x Y x X]
    }
    return flow_2d;
}

torch::Tensor index_flow(
    torch::Tensor dP, 
    int64_t Lz, int64_t Ly, int64_t Lx, int64_t niter = 139
    ) {
    std::vector<int64_t> shape = {Lz, Ly, Lx};
    auto p_vec = torch::meshgrid({torch::arange(shape[0]), torch::arange(shape[1]), torch::arange(shape[2])}, "ij");
    torch::Tensor p = torch::stack(p_vec, 0).to(torch::kFloat);
    torch::Tensor inds = torch::nonzero(dP[0].abs() > 1e-3).to(torch::kLong);
    auto z = inds.select(1, 0);
    auto y = inds.select(1, 1);
    auto x = inds.select(1, 2);
    auto pmin = torch::zeros(3);
    auto pmax = torch::stack({torch::tensor(Lz - 1, torch::kFloat), torch::tensor(Ly - 1, torch::kFloat), torch::tensor(Lx - 1, torch::kFloat)});
    auto pp = p.index({torch::indexing::Slice(), z, y, x});
    
    for (int64_t iter = 0; iter < niter; ++iter) {
        auto pz = torch::floor(pp.select(0, 0)).to(torch::kLong);
        auto py = torch::floor(pp.select(0, 1)).to(torch::kLong);
        auto px = torch::floor(pp.select(0, 2)).to(torch::kLong);
        auto pdP = dP.index({torch::indexing::Slice(), pz, py, px});
        pp = (pp + pdP).transpose(0, -1);
        pp = torch::clamp(pp, pmin, pmax).transpose(0, -1);
    }
    
    p.index_put_({torch::indexing::Slice(), z, y, x}, pp);
    return p;
}

std::vector<torch::Tensor> expand_pt_unique(std::vector<torch::Tensor> pix, int64_t iter){
    int64_t expand_sz = 1+iter*2;
    torch::Tensor expand_id = torch::arange(expand_sz*expand_sz*expand_sz);
    torch::Tensor expand = expand_id.reshape({expand_sz, expand_sz, expand_sz});
    torch::Tensor x = pix[0] - pix[0].min();
    torch::Tensor y = pix[1] - pix[1].min();
    torch::Tensor z = pix[2] - pix[2].min();
    auto unique_out = at::_unique(expand.index({x, y, z}), false, true);
    torch::Tensor unique_eloc = std::get<0>(unique_out);
    torch::Tensor unique_idx = std::get<1>(unique_out);
    torch::Tensor loc = torch::zeros_like(pix[0]).to(torch::kBool);
    torch::Tensor sorted_idx = std::get<1>(torch::sort(unique_idx, true, -1, false));
    loc.index_put_({sorted_idx.index({torch::cat({torch::tensor({0}), unique_idx.bincount().cumsum(0).slice(0, 0, -1)})})}, true);
    return std::vector<torch::Tensor>({pix[0].index({loc}), pix[1].index({loc}), pix[2].index({loc})});
}

std::vector<torch::Tensor> flow_3DtoNIS(
    // torch::jit::script::Module meshgrider,
    torch::jit::script::Module flow_3DtoSeed,
    torch::Tensor p, 
    torch::Tensor iscell, 
    int64_t ilabel,
    int64_t rpad = 20
) {
    std::cout<<"ilabel "<<ilabel<<"\n";
    int64_t zpad = 4;
    std::vector<int64_t> rpads({zpad, rpad, rpad});
    int64_t iter_num = 3; // Needs to be odd
    int64_t dims = 3;
    std::vector<torch::Tensor> edges(dims);
    std::vector<torch::Tensor> seeds;
    int64_t Lz = p.size(1);
    int64_t Ly = p.size(2);
    int64_t Lx = p.size(3);
    std::vector<int64_t> shape0 = {Lz, Ly, Lx};
    std::vector<int64_t> shape;
    std::vector<torch::Tensor> inds = torch::meshgrid({torch::arange(shape0[0]), torch::arange(shape0[1]), torch::arange(shape0[2])}, "ij");
    
    for (int64_t i = 0; i < dims; ++i) {
        p.index_put_({i, torch::logical_not(iscell)}, inds[i].index({torch::logical_not(iscell)}).to(torch::kFloat));
    }
    for (int64_t i = 0; i < dims; ++i) {
        edges[i] = torch::arange(-0.5 - rpads[i], shape0[i] + 0.5 + rpads[i]);
        shape.push_back(edges[i].numel()-1);
    }
    p = p.to(torch::kLong);
    p = torch::stack({p[0].view(-1), p[1].view(-1), p[2].view(-1)}, 0);
    torch::Tensor fg = torch::zeros(shape, torch::kBool);
    fg.index_put_({p[0] + rpads[0], p[1] + rpads[1], p[2] + rpads[2]}, torch::ones(1, torch::kBool));

    torch::Tensor h = histogramdd(p.transpose(0, 1).to(torch::kDouble).detach().clone(), edges);
    torch::Tensor pix = flow_3DtoSeed(std::vector<torch::jit::IValue>({h})).toTensor();
    int64_t seed_num = pix.size(0);
    print_with_time("There are ");
    std::cout << seed_num << " seeds of instance, extend them as mask\n";
    

    torch::Tensor expand = torch::nonzero(torch::ones({3, 3, 3})).transpose(0, 1);
    std::vector<std::vector<torch::Tensor>> pix_copy(pix.size(0));
    std::vector<torch::Tensor> newpix(expand.size(0));
    std::vector<torch::Tensor> iin(expand.size(0));
    torch::Tensor igood;
    torch::Tensor iin_all;
    for (int64_t iter = 0; iter < iter_num; ++iter) {
        print_with_time("Extend iter ");
        std::cout << iter+1 << ", "; 
        for (int64_t k = 0; k < pix_copy.size(); ++k) {
            if (iter == 0) {
                pix_copy[k] = pix[k].unbind(0);
            }
            newpix.clear();
            iin.clear();
            
            for (int64_t i = 0; i < expand.size(0); ++i) {
                // Extend
                newpix.push_back(expand[i].unsqueeze(1) + pix_copy[k][i].unsqueeze(0) - 1);
                newpix[i] = newpix[i].flatten();
                // Clip coordinates
                iin.push_back(torch::logical_and(newpix[i] >= 0, newpix[i] < shape[i]));
            }

            iin_all = torch::stack(iin).all(0);
            for (int64_t i = 0; i < expand.size(0); ++i) {
                newpix[i] = newpix[i].index({iin_all});
            }
            if (iter > 0){
                // Get unique coordinates
                newpix = expand_pt_unique(newpix, iter+1);
            }
            igood = h.index({newpix[0], newpix[1], newpix[2]})>2;
            for (int64_t i = 0; i < expand.size(0); ++i) {
                pix_copy[k][i] = newpix[i].index({igood});
            }

            if (k % 10000 == 0) {
                std::cout << "...";
            }
        }
        std::cout << ", Done" << "\n"; 
    }
     
    int64_t remove_c = 0;
    torch::Tensor M = torch::zeros(shape, torch::kLong);
    // std::vector<torch::Tensor> coords;
    std::vector<torch::Tensor> labels;
    // std::vector<int64_t> vols;
    std::vector<torch::Tensor> coords0;
    std::vector<torch::Tensor> coords1;
    std::vector<torch::Tensor> coords2;
    float fLz = Lz;
    float fLy = Ly;
    float fLx = Lx;
    float big = fLz * fLy * fLx * 0.001;
    print_with_time("Index masks, ");
    std::cout << "Ultra big mask threshold: " << big << ". Start ";
    for (int64_t k = 0; k < pix_copy.size(); ++k) {    
        if (pix_copy[k][0].size(0)==0) {
            remove_c += 1;
            continue;
        } 
        torch::Tensor is_fg = fg.index({pix_copy[k][0], pix_copy[k][1], pix_copy[k][2]});
        torch::Tensor is_bg = torch::logical_not(is_fg);
        if (is_bg.all().item<bool>()) {
            remove_c += 1;
            continue;
        } 
        std::vector<torch::Tensor> coord({pix_copy[k][0].index({is_fg}), pix_copy[k][1].index({is_fg}), pix_copy[k][2].index({is_fg})});
        if (coord[0].size(0) > big) {
            remove_c += 1;
            continue;
        }
        ilabel += 1;
        // vols.push_back(coord[0].size(0));
        // std::vector<torch::Tensor> center({
        //     (coord[0].max() + coord[0].min()) / 2,
        //     (coord[1].max() + coord[1].min()) / 2,
        //     (coord[2].max() + coord[2].min()) / 2
        // });
        labels.push_back(torch::tensor(ilabel, torch::kLong).repeat({coord[0].size(0)}));
        coords0.push_back(coord[0]);
        coords1.push_back(coord[1]);
        coords2.push_back(coord[2]);
        // M.index_put_({coord[0], coord[1], coord[2]}, torch::tensor(ilabel, torch::kLong));
        // coords.push_back(torch::stack(coord, 1));
        // labels.push_back(ilabel);
        // centers.push_back(torch::stack(center, 0));
        if (k % 10000 == 0) {
            std::cout << "...";
        }
    }
    std::cout << ", Done, removed " << remove_c << " ultra big or small masks, " << pix_copy.size() << " remain" << "\n"; 
    if (coords0.size() == 0){
        return {torch::zeros(0)};
    } else {
        torch::Tensor coord0 = torch::cat(coords0);
        torch::Tensor coord1 = torch::cat(coords1);
        torch::Tensor coord2 = torch::cat(coords2);
        M.index_put_({coord0, coord1, coord2}, torch::cat(labels));
        M = M.index({p[0]+rpads[0], p[1]+rpads[1], p[2]+rpads[2]}).view(shape0);
        std::vector<torch::Tensor> nis_profile = get_large_fg_coord(M);
        return {
            M, 
            nis_profile[0], 
            nis_profile[1], 
            nis_profile[2], 
            nis_profile[3],
            torch::tensor(ilabel, torch::kLong)
        };
    }
}
