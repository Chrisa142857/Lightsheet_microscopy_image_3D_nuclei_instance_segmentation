#include "gnn_stitch_gap.h"

torch::Tensor gnn_stitch_gap(
    torch::jit::script::Module gnn_message_passing, 
    torch::jit::script::Module gnn_classifier,
    torch::Tensor img1, // tgt
    torch::Tensor mask1, 
    torch::Tensor flow1,
    torch::Tensor img2, // src
    torch::Tensor mask2, 
    torch::Tensor flow2,
    std::string device
) {
    img1 = norm_tensor(img1);
    img2 = norm_tensor(img2);
    gnn_classifier.eval();
    gnn_message_passing.eval();
    gnn_classifier.to(device);
    gnn_message_passing.to(device);
    std::vector<torch::Tensor> graph = build_graph(img1, mask1, flow1, img2, mask2, flow2, device);
    std::vector<torch::jit::IValue> gnn_input;
    gnn_input.push_back(graph[0]);
    gnn_input.push_back(graph[1]);
    gnn_input.push_back(graph[2]);
    torch::Tensor feat = gnn_message_passing(gnn_input).toTensor();
    std::vector<torch::Tensor> feat_vec;
    for(int64_t nid=0; nid < graph[5].size(0); nid++){
        feat_vec.push_back(feat.index({graph[1][0]==nid}).reshape(-1));
    }
    feat = torch::stack(feat_vec);
    gnn_input.clear();
    gnn_input.push_back(feat);
    torch::Tensor logits = gnn_classifier(gnn_input).toTensor();
    torch::Tensor pred = logits.argmax(1);
    torch::Tensor not_new_cell = pred != 0;
    pred = pred.index({not_new_cell}) - 1;
    logits = logits.index({not_new_cell});
    torch::Tensor node0_ind = torch::arange({graph[5].size(0)}).index({not_new_cell});
    torch::Tensor node1_ind = graph[5].index({not_new_cell, pred});
    
    std::vector<torch::Tensor> oldid_vec;
    std::vector<torch::Tensor> newid_vec;
    std::tuple<torch::Tensor, torch::Tensor> unique_out = at::_unique(node1_ind);
    torch::Tensor nid1 = std::get<0>(unique_out);
    // If node1 matched to multiple node0, choose max score
    for (int64_t i=0; i<nid1.size(0); i++){
        oldid_vec.push_back(graph[4].index({nid1[i]})); // get org id of src mask
        torch::Tensor nid_place = (node1_ind==nid1[i]);
        torch::Tensor scores = logits.index({nid_place, torch::indexing::Slice(1, torch::indexing::None)}); // [X x topn]
        scores = std::get<0>(scores.max(1));
        torch::Tensor matched_nid0 = node0_ind.index({nid_place, scores.argmax()});
        newid_vec.push_back(graph[3].index({matched_nid0})); // get org id of tgt mask
    }
    // [2 x X]
    return torch::stack({
        torch::stack(oldid_vec), torch::stack(newid_vec)
    }).detach().cpu();
}

std::vector<torch::Tensor> build_graph(
    torch::Tensor img1, 
    torch::Tensor mask1, 
    torch::Tensor flow1,
    torch::Tensor img2, 
    torch::Tensor mask2, 
    torch::Tensor flow2,
    std::string device
) {
    // feat = {id_old2new, node_feat, edge_feat, bbox}
    std::vector<torch::Tensor> feat1 = get_feat(img1.to(device), mask1.to(device), flow1.to(device));
    std::vector<torch::Tensor> feat2 = get_feat(img2.to(device), mask2.to(device), flow2.to(device));
    // edges = {edge_id, edge_attr, topn_id}
    std::vector<torch::Tensor> edges = get_edge(feat1[3], feat2[3], feat1[2], feat2[2], device);
    // graph = {x, edge_id, edge_attr, node0_oldid2new, node1_oldid2new, topn_id}
    std::vector<torch::Tensor> graph;
    graph.push_back(torch::cat({feat1[1], feat2[1]}));
    graph.push_back(edges[1]);
    graph.push_back(edges[2]);
    graph.push_back(feat1[0]);
    graph.push_back(feat2[0]);
    graph.push_back(edges[3]);
    return graph;
}

std::vector<torch::Tensor> get_edge(
    torch::Tensor bbox1, //[N x 4]
    torch::Tensor bbox2, // [N x 4]
    torch::Tensor zflow1, //[N]
    torch::Tensor zflow2, // [N]
    std::string device
) {
    int topn=2;
    int64_t N1 = bbox1.size(0);
    torch::Tensor center1 = bbox1.slice(1, 0, 2);
    torch::Tensor center2 = bbox2.slice(1, 0, 2);
    std::vector<torch::Tensor> edge_id0;
    std::vector<torch::Tensor> edge_id1;
    std::vector<torch::Tensor> edge_attr;
    std::vector<torch::Tensor> topn_ind;
    for (int64_t i=0; i<bbox1.size(0); i++){
        torch::Tensor d = center1[i] - center2;
        torch::Tensor dist = (d*d).sum(1).sqrt();
        torch::Tensor ind0 = torch::tensor(std::vector<int64_t>({i})).repeat(topn).to(device); // [topn]
        torch::Tensor ind1 = torch::argsort(dist).slice(0, 0, topn); // [topn]
        topn_ind.push_back(ind1);
        edge_id0.push_back(ind0); 
        edge_id1.push_back(ind1+N1); 
        edge_attr.push_back(
            torch::stack({
                zflow1.index({ind0}), 
                zflow2.index({ind1}), 
                bbox1.index({ind0, 0}) - bbox2.index({ind1, 0}), 
                bbox1.index({ind0, 1}) - bbox2.index({ind1, 1}), 
                bbox1.index({ind0, 2}) / bbox2.index({ind1, 2}), 
                bbox1.index({ind0, 3}) / bbox2.index({ind1, 3})
            }, -1) // [topn x 6]
        );
    }
    std::vector<torch::Tensor> outputs;
    outputs.push_back(torch::stack({torch::cat(edge_id0), torch::cat(edge_id1)})); // [2 x N*topn]
    outputs.push_back(torch::cat(edge_attr)); // [N*topn x 6]
    outputs.push_back(torch::stack(topn_ind)); // [N x topn]
    return outputs;
}

std::vector<torch::Tensor> get_feat(
    torch::Tensor img, // 2D img [Ly x Lx]
    torch::Tensor mask, // 2D mask [Ly x Lx]
    torch::Tensor flow // 3D flow [3 x Ly x Lx]
) {
    int64_t node_feat_size=200;
    namespace F = torch::nn::functional;
    torch::Tensor oldid;
    std::tuple<torch::Tensor, torch::Tensor> unique_out;
    // torch::Tensor newid;
    unique_out = at::_unique(mask);
    oldid = std::get<0>(unique_out);
    if (oldid[0].item<int64_t>()==0) {oldid = oldid.slice(0, 1, oldid.size(0));}
    // newid = torch::arange(oldid.size(0));
    // std::vector<torch::Tensor> id_old2new;
    // id_old2new.push_back(oldid);
    // id_old2new.push_back(newid);
    std::vector<torch::Tensor> zflows;
    std::vector<torch::Tensor> hists;
    std::vector<torch::Tensor> feats;
    std::vector<torch::Tensor> bboxs;
    for (int64_t i=0; i<oldid.size(0); i++){
        torch::Tensor m = mask==oldid[i];
        torch::Tensor feat = F::interpolate(
            img.index({m}).unsqueeze(0).unsqueeze(0), 
            F::InterpolateFuncOptions().size(std::vector<int64_t>({node_feat_size})).mode(torch::kLinear).align_corners(false)
        ).squeeze();
        // std::cout<<"----image index\n";
        torch::Tensor zflow = flow.index({-1, m}).mean();
        // std::cout<<"----zflow index\n";
        torch::Tensor xyflow = torch::stack({flow.index({0, m}), flow.index({1, m})});
        // std::cout<<"----xyflow index\n";
        torch::Tensor hist = hist_flow_vector(xyflow);
        zflows.push_back(zflow);
        hists.push_back(hist);
        feats.push_back(feat);
        bboxs.push_back(get_bbox(m));
    }
    std::vector<torch::Tensor> outputs;
    // **ID remap dict** [N]
    outputs.push_back(oldid);
    // **Node feature** [200+9, N]
    outputs.push_back(torch::cat({torch::stack(feats), torch::stack(hists)}, -1));
    // **Edge feature** [N]
    outputs.push_back(torch::stack(zflows));
    // **Bbox** for building edge 
    outputs.push_back(torch::stack(bboxs));
    return outputs;
}

torch::Tensor get_bbox(torch::Tensor mask){
    std::vector<torch::Tensor> xy = torch::where(mask);
    torch::Tensor xmin = xy[0].min();
    torch::Tensor xmax = xy[0].max();
    torch::Tensor ymin = xy[1].min();
    torch::Tensor ymax = xy[1].max();
    torch::Tensor bbox = torch::stack({(xmax+xmin)/2, (ymax+ymin)/2, xmax-xmin, ymax-ymin});
    return bbox;
}

torch::Tensor norm_tensor(torch::Tensor tensor){
    return (tensor - tensor.min()) / (tensor.max() - tensor.min());
}

torch::Tensor hist_flow_vector(torch::Tensor flow){
    // Get the dimensions of the input tensor
    auto sizes = flow.sizes();
    int num_dims = sizes.size();

    // Create a vector to store the binary dimensions
    std::vector<std::vector<torch::Tensor>> dims;
    for (int i = 0; i < num_dims; ++i) {
        std::vector<torch::Tensor> dim_vec;
        dim_vec.push_back(flow[i] < 0);
        dim_vec.push_back(flow[i] == 0);
        dim_vec.push_back(flow[i] > 0);
        dims.push_back(dim_vec);
    }

    // Create a vector to store the histogram
    std::vector<torch::Tensor> hist;

    // Calculate the histogram based on the number of dimensions
    if (num_dims == 2) {
        for (torch::Tensor d1 : dims[0]) {
            for (torch::Tensor d2 : dims[1]) {
                hist.push_back((d1 & d2).sum());
            }
        }
    } else {
        for (torch::Tensor d1 : dims[0]) {
            for (torch::Tensor d2 : dims[1]) {
                for (torch::Tensor d3 : dims[2]) {
                    hist.push_back((d1 & d2 & d3).sum());
                }
            }
        }
    }

    // Convert the histogram vector to a torch tensor
    torch::Tensor hist_tensor = torch::stack(hist);

    return hist_tensor;
}