#include "h5_writer.h"


DataSet create_extensible_dataset(File file, std::string DATASET_NAME, hsize_t ncol, bool dtype_is_long){
    // Create a dataspace with initial shape and max shape
    DataSpace dataspace = DataSpace({0, ncol}, {DataSpace::UNLIMITED, ncol});
    // Use chunking
    DataSetCreateProps props;
    props.add(Chunking(std::vector<hsize_t>{16, ncol}));
    // Create the dataset
    if (dtype_is_long){
        DataSet dataset = file.createDataSet(DATASET_NAME, dataspace, create_datatype<int64_t>(), props);
        return dataset;
    } else {
        DataSet dataset = file.createDataSet(DATASET_NAME, dataspace, create_datatype<float>(), props);
        return dataset;
    }    
}

void init_h5data(std::string FILE_NAME, std::vector<hsize_t> whole_brain_shape){
    hsize_t Lz = whole_brain_shape[0];
    hsize_t Ly = whole_brain_shape[1];
    hsize_t Lx = whole_brain_shape[2];
    File file(FILE_NAME, File::ReadWrite | File::Create | File::Truncate);
    DataSetCreateProps props1;
    props1.add(Chunking(std::vector<hsize_t>{16, 128, 128}));
    // DataSetCreateProps props2;
    // props2.add(Chunking(std::vector<hsize_t>{2, 2, 2}));
    DataSet seg_dset = file.createDataSet("nuclei_segmentation", DataSpace({Lz, Ly, Lx}, {DataSpace::UNLIMITED, Ly, Lx}), create_datatype<int64_t>(), props1);
    // DataSet bmask_dset = file.createDataSet("binary_mask", DataSpace({Lz, Ly, Lx}, {DataSpace::UNLIMITED, DataSpace::UNLIMITED, DataSpace::UNLIMITED}), create_datatype<int>(), props2);
    DataSet coord_dset = create_extensible_dataset(file, "coordinate", 3, true);
    DataSet ilabel_dset = create_extensible_dataset(file, "instance_label", 1, true);
    DataSet ivolume_dset = create_extensible_dataset(file, "instance_volume", 1, true);
    DataSet icenter_dset = create_extensible_dataset(file, "instance_center", 3, false);
    // std::vector<DataSet> dsetlist;
    // dsetlist.push_back(seg_dset);
    // // dsetlist.push_back(bmask_dset);
    // dsetlist.push_back(coord_dset);
    // dsetlist.push_back(ilabel_dset);
    // dsetlist.push_back(ivolume_dset);
    // dsetlist.push_back(icenter_dset);
    // return dsetlist;
    // return file;
}

std::vector<torch::Tensor> save_h5data(std::string FILE_NAME, std::vector<torch::Tensor> datalist, hsize_t old_instance_n, hsize_t old_contour_n, hsize_t zmin, hsize_t zmax, std::vector<hsize_t> whole_brain_shape){
    // File h5file(FILE_NAME, File::ReadWrite);
    // hsize_t Lz = whole_brain_shape[0];
    // hsize_t Ly = whole_brain_shape[1];
    // hsize_t Lx = whole_brain_shape[2];
    // DataSet seg_dset = h5file.getDataSet("nuclei_segmentation");
    // DataSet coord_dset = h5file.getDataSet("coordinate");
    // DataSet ilabel_dset = h5file.getDataSet("instance_label");
    // DataSet ivolume_dset = h5file.getDataSet("instance_volume");
    // DataSet icenter_dset = h5file.getDataSet("instance_center");
    // // torch::Tensor seg = datalist[0]; // Mask IDs, which are in the range [1, new_instance_n] 
    // // // torch::Tensor bmask = (seg > 0); //.to(torch::kLong);
    // // torch::Tensor coord = datalist[1];
    // // torch::Tensor ilabel = datalist[2];
    // // torch::Tensor ivolume = datalist[3];
    // // torch::Tensor icenter = datalist[4];
    // hsize_t new_contour_n = datalist[1].size(0);
    // hsize_t new_instance_n = datalist[2].size(0);
    // // Mask IDs extend
    // // datalist[0].index_put_({datalist[0] > 0}, datalist[0].index({datalist[0] > 0}) + (int64_t)old_instance_n);
    // // datalist[2] = datalist[2] + (int64_t)old_instance_n;
    // // Extend datasets
    // print_with_time("Start to extend datasets of H5 database\n");
    // if (zmax > Lz) {
    //     seg_dset.resize({zmax, Ly, Lx});
    //     // bmask_dset.resize({zmax, Ly, Lx});
    // }
    // coord_dset.resize({old_contour_n+new_contour_n, 3});
    // ilabel_dset.resize({old_instance_n+new_instance_n, 1});
    // ivolume_dset.resize({old_instance_n+new_instance_n, 1});
    // icenter_dset.resize({old_instance_n+new_instance_n, 3});
    // print_with_time("Done\n");
    // // Write data
    // seg_dset.select({zmin, 0, 0}, {zmax-zmin, Ly, Lx}).write_raw(datalist[0].data_ptr<int64_t>());
    // coord_dset.select({old_contour_n, 0}, {new_contour_n, 3}).write_raw(datalist[1].data_ptr<int64_t>());
    // ilabel_dset.select({old_instance_n, 0}, {new_instance_n, 1}).write_raw(datalist[2].data_ptr<int64_t>());
    // ivolume_dset.select({old_instance_n, 0}, {new_instance_n, 1}).write_raw(datalist[3].data_ptr<int64_t>());
    // icenter_dset.select({old_instance_n, 0}, {new_instance_n, 3}).write_raw(datalist[4].data_ptr<float>());
    std::vector<torch::Tensor> output;
    output.push_back(datalist[0].index({-1, "..."}).detach().clone());
    output.push_back(datalist[0].index({0, "..."}).detach().clone());
    return output;
}