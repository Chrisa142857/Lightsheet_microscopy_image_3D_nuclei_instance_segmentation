#include "flow_op.h"


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
    
    for (int64_t iter = 0; iter < iter_num; ++iter) {
        print_with_time("Extend iter ");
        std::cout << iter+1 << ", "; 
        for (int64_t k = 0; k < pix_copy.size(); ++k) {
            if (iter == 0) {
                pix_copy[k] = pix[k].unbind(0);
            }
            std::vector<torch::Tensor> newpix(expand.size(0));
            std::vector<torch::Tensor> iin(expand.size(0));
            
            for (int64_t i = 0; i < expand.size(0); ++i) {
                // Extend
                newpix[i] = expand[i].unsqueeze(1) + pix_copy[k][i].unsqueeze(0) - 1;
                newpix[i] = newpix[i].flatten();
                // Clip coordinates
                iin[i] = torch::logical_and(newpix[i] >= 0, newpix[i] < shape[i]);
            }

            torch::Tensor iin_all = torch::stack(iin).all(0);
            for (int64_t i = 0; i < expand.size(0); ++i) {
                newpix[i] = newpix[i].index({iin_all});
            }
            if (iter > 0){
                // Get unique coordinates
                newpix = expand_pt_unique(newpix, iter+1);
            }
            torch::Tensor igood = h.index({newpix[0], newpix[1], newpix[2]})>2;
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
    std::vector<torch::Tensor> coords;
    std::vector<int64_t> labels;
    std::vector<int64_t> vols;
    std::vector<torch::Tensor> centers;
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
        vols.push_back(coord[0].size(0));
        std::vector<torch::Tensor> center({
            (coord[0].max() + coord[0].min()) / 2,
            (coord[1].max() + coord[1].min()) / 2,
            (coord[2].max() + coord[2].min()) / 2
        });
        M.index_put_({coord[0], coord[1], coord[2]}, torch::tensor(ilabel, torch::kLong));
        coords.push_back(torch::stack(coord, 1));
        labels.push_back(ilabel);
        centers.push_back(torch::stack(center, 0));
        if (k % 10000 == 0) {
            std::cout << "...";
        }
    }
    std::cout << ", Done, removed " << remove_c << " ultra big or small masks, " << pix_copy.size() << " remain" << "\n"; 
    if (coords.size() == 0){
        return {torch::zeros(0)};
    } else {
        return {
            M.index({p[0]+rpads[0], p[1]+rpads[1], p[2]+rpads[2]}).view(shape0), 
            torch::cat(coords, 0), 
            torch::tensor(labels, torch::kLong), 
            torch::tensor(vols, torch::kLong), 
            torch::stack(centers)
        };
    }
}
