#include "image_reader.h"

using namespace std;
using namespace cv;

torch::Tensor load_tif_as_tensor(string path) {
    Mat image = imread(path, IMREAD_UNCHANGED);
    int64_t width = image.cols;
    int64_t height = image.rows;
    // Check if the image was loaded successfully
    if (image.empty()) {
        cout << "Error loading image" << width << "," << height << "\n" << "path: "+path;
        exit(0);
    }
    // cout << width << "," << height << "\n";
    // Convert the image data to a torch::Tensor
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt16);
    torch::Tensor tensor_image = torch::from_blob(image.data, {1, height, width}, options);//.permute({0, 3, 1, 2})
    tensor_image = tensor_image.to(torch::kFloat32);// / 255.0;
    // cout << tensor_image.max() << tensor_image.min() << tensor_image.mean() << "\n";
    return tensor_image;
}

