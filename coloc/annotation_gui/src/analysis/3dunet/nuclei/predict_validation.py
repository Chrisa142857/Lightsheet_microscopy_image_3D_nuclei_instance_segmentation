"""
Run validation cases in ./data/validation

"""
import os
import nibabel as nib
import numpy as np
import argparse
from unet3d.training import load_old_model
from unet3d.prediction import patch_wise_prediction
from nuclei.img_utils import patch_wise_prediction2, prediction_to_centroids

parser = argparse.ArgumentParser(description='Predict from validation dataset.')
parser.add_argument('--r', metavar='r', type=str, nargs='+',
                    help='Resolution tag')
parser.add_argument('--g', metavar='g', type=str, nargs='+',
                    help='GPU tag')
parser.add_argument('--m', metavar='m', type=str, nargs='+',
                    help='Model tag')
args = parser.parse_args()

if args.g:
    gpu_idx = args.g[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if args.r:
    resolution = args.r[0]
else:
    resolution = '075'

if args.m:
    model_idx = args.m[0]
else:
    model_idx = '2'

model_path = '/home/ok37/repos/3dunet-centroid/nuclei/isensee_2017_model' + model_idx + '.h5'
#model_path = '/home/ok37/repos/3dunet-centroid/nuclei/128_model_121.h5'

# %% Run
def main(resolution, model_path):
    model = load_old_model(model_path)

    # Get images
    images = get_validation_folders(resolution)

    # Run prediction
    predictions = []
    for i in range(len(images)):
        data = images[i].get_fdata()
        data = np.squeeze(data)
        data = np.asarray(data, dtype=np.float64)

        patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])

        if patch_shape == data.shape[-3:]:
            img_shape = (1, 1) + data.shape
            data_reshaped = np.reshape(data, img_shape)
            prediction = model.predict(data_reshaped)
        else:
            img_shape = (1, 1) + data.shape
            data_reshaped = np.reshape(data, img_shape)
            prediction = patch_wise_prediction2(model=model, data=data_reshaped, overlap=[16, 16, 8],
                                                pred_threshold=0.5, int_threshold=0)

        predictions.append(np.squeeze(prediction))

    # Save results
    save_directory = os.path.join(os.getcwd(), 'results', 'validation_results')
    save_prediction_results(images, predictions, save_directory)

    return


# %% Read Images
def get_validation_folders(resolution):
    img_path = os.path.join(os.getcwd(), 'data', 'validation', resolution)
    img_folders = os.listdir(img_path)

    img_folders = [os.path.join(img_path, f) for f in sorted(img_folders)]

    images = []
    for f in range(len(img_folders)):
        files = os.listdir(img_folders[f])
        file = os.path.join(img_folders[f], files[0])

        img = nib.load(file)
        images.append(img)

    return images


# %% Save Results
def save_prediction_results(images, predictions, save_directory):
    for i in range(len(images)):
        save_folder_name = os.path.join(save_directory, 'f' + str(i + 1))

        if not os.path.isdir(save_folder_name):
            os.mkdir(save_folder_name)

        save_name = os.path.join(save_folder_name, 'image.nii')
        nib.save(nib.Nifti1Image(images[i].get_data().astype('uint16'), affine=np.eye(4)), save_name)

        save_name = os.path.join(save_folder_name, 'prediction.nii')
        nib.save(nib.Nifti1Image(predictions[i].astype('uint8'), affine=np.eye(4)), save_name)

    return


# %% Predict individual case
if __name__ == "__main__":
    main(resolution, model_path)
