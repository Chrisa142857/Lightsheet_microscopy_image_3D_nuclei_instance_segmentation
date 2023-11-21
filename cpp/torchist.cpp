#include "torchist.h"

torch::Tensor histogramdd(
    torch::Tensor x,
    std::vector<torch::Tensor> edges
) {
    // Preprocess
    int64_t D = x.size(-1);
    x = x.view({-1, D}).squeeze(-1);
    int64_t minlength = 1;
    std::vector<torch::Tensor> edge_vec; 
    std::vector<int64_t> bins; 
    std::vector<torch::Tensor> low; 
    std::vector<torch::Tensor> upp; 
    int64_t max_bin = 0;
    for (torch::Tensor e : edges) {
        edge_vec.push_back(e.flatten());
        bins.push_back(e.numel() - 1);
        low.push_back(e[0]);
        upp.push_back(e[-1]);
        if ((e.numel()-1) > max_bin) {
            max_bin = (e.numel() - 1);
        }
        minlength *= (e.numel() - 1);
    }

    torch::Tensor pack = x.new_full({D, max_bin + 1}, std::numeric_limits<float>::infinity());
    for (int64_t i = 0; i < D; ++i) {
        pack.index_put_({i, torch::indexing::Slice(torch::indexing::None, edges[i].numel())}, edges[i]);
    }

    torch::Tensor new_edges = pack;
    torch::Tensor bin_tensor = torch::tensor(bins).squeeze().to(torch::kLong);
    torch::Tensor low_tensor = torch::stack(low).squeeze().to(x);
    torch::Tensor upp_tensor = torch::stack(upp).squeeze().to(x);

    // Filter out-of-bound values
    torch::Tensor mask = torch::logical_not(out_of_bounds(x, low_tensor, upp_tensor));
    x = x.index({mask, torch::indexing::Slice()});

    // Indexing
    torch::Tensor idx = torch::searchsorted(new_edges, x.t().contiguous(), false, true).t() - 1;

    // Histogram
    idx = ravel_multi_index(idx, bin_tensor);
    torch::Tensor hist = idx.bincount({}, minlength);
    hist = hist.reshape({bin_tensor[0].item<int64_t>(), bin_tensor[1].item<int64_t>(), bin_tensor[2].item<int64_t>()});

    return hist;
}

torch::Tensor ravel_multi_index(torch::Tensor coords, torch::Tensor shape) {
    torch::Tensor coefs = torch::stack({shape[1], shape[2], torch::tensor(1)}).to(coords).flipud().cumprod(0).flipud();
    return (coords * coefs).sum(-1);
}

torch::Tensor out_of_bounds(torch::Tensor x, torch::Tensor low, torch::Tensor upp) {
    torch::Tensor a = x < low;
    torch::Tensor b = x > upp;
    if (x.dim() > 1) {
        a = a.any(-1);
        b = b.any(-1);
    }
    return a.logical_or(b);
}
