import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import numpy as np
import cc3d
from datetime import datetime
from scipy.io import loadmat

from unet3d.training import load_old_model
from nuclei.multi_gpu import make_parallel
from scipy import ndimage
from skimage.transform import resize, rescale
from skimage.exposure import rescale_intensity

import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from nuclei.img_utils import overlay_centroids

model_file ='/home/ok37/repos/3dunet-centroid/nuclei/img_test/7_imgs.h5'
model = load_old_model(model_file)

mask_file = '/media/SteinLab2/WT7R/output/variables/I_mask_scaled.mat'
mask = loadmat(mask_file)
mask = mask['I_mask']

input_img_directory = '/media/SteinLab2/WT7R/output/stitched/topro'

chunk_size = [160, 160, 48]
overlap = [2, 2, 1]

acquired_img_resolution = [1.208, 1.208, 4]
trained_img_resolution = [0.9375, 0.9375, 3.125]
mask_resolution = [25, 25, 4]

#%% Determine chunk sizes while considering scaling and chunk overlap
res = [i / j for i, j in zip(acquired_img_resolution, trained_img_resolution)]

load_chunk_size = [i / j for i, j in zip(chunk_size, res)]
load_chunk_size = [round(s) for s in load_chunk_size]

effective_chunk_size = load_chunk_size[0:2] - np.array(overlap[0:2])*2

#%% Read images
img_directory = os.path.join(os.getcwd(), input_img_directory)

# Taking only channel 1 images (assumed to be cell nuclei)
files = os.listdir(img_directory)
matching = [s for s in files if "C1" in s]
matching = sorted(matching)

# Read only the number of images for each chunk
img_list = [img_directory + '/' + s for s in matching]
total_images = len(img_list)
n_z_chunks = np.ceil(total_images/load_chunk_size[2])

#z_start = np.linspace(1, total_images, n_z_chunks+1)
#z_start = z_start[1:len(z_start)]
#z_end = z_start - 1

#%%
images = []
z_start = 0
for n in range(int(n_z_chunks)):
    startTime = datetime.now()
    images = []
    z_end = z_start + load_chunk_size[2]
    if z_end > total_images:
        z_end = total_images-1

    # Read images
    images = [cv2.imread(file, -1) for file in img_list[z_start:z_end]]
    images = np.asarray(images, dtype=np.float32)
    images = images.swapaxes(1, 2).swapaxes(0, 2)
    images = rescale_intensity(images,in_range=(0, 8000))

    # Take the mask for the respective z positions
    mask_chunk = mask[:, :, z_start:z_end]
    mask_projection = np.amax(mask_chunk, axis=2)

    mask_scale = [i / j for i, j in zip(images.shape, mask.shape)]
    mask_scale[2] = 1

    y_start_m = np.where(mask_projection)[0].min()
    y_start = int(round(y_start_m * mask_scale[0]))
    y_end_m = np.where(mask_projection)[0].max()
    y_end = int(round(y_end_m * mask_scale[0]))

    x_start_m = np.where(mask_projection)[1].min()
    x_start = int(round(x_start_m * mask_scale[1]))
    x_end_m = np.where(mask_projection)[1].max()
    x_end = int(round(x_end_m * mask_scale[1]))

    cropped = images[y_start:y_end, x_start:x_end, :]
    mask_cropped = mask_chunk[y_start_m:y_end_m, x_start_m:x_end_m, :]

    #cropped_projection = np.amax(cropped, axis=2)*10
    #mask_cropped_projection = np.amax(mask_cropped, axis=2)

    print('Working on slices', z_start, 'through', z_end)

    # Rescale images to match unet resolution
    rescaled_shape = np.array(cropped.shape)*res
    rescaled_shape[2] = chunk_size[2]
    rescaled = resize(cropped, rescaled_shape.round())
    mask_chunk_r = resize(mask_cropped, rescaled.shape, order=0)
    [rows, cols, slices] = rescaled.shape

    print('Images cropped and rescaled in: ', datetime.now() - startTime)

    # We're going to pad the outside to make chunks fit
    # First determine number of chunks
    # Calculate number of chunks required to contain entire image + padding
    n_chunks_r = int(np.ceil(rows/(chunk_size[0]-overlap[0])))
    n_chunks_c = int(np.ceil(cols/(chunk_size[1]-overlap[1])))
    total_chunks = n_chunks_r * n_chunks_c

    grid_size_r = n_chunks_r * chunk_size[0]
    grid_size_c = n_chunks_c * chunk_size[1]

    # Determine how much padding required
    pad_r = (grid_size_r - rows)/ 2
    pad_r_top = int(np.ceil(pad_r))
    pad_r_bottom = int(np.floor(pad_r))

    pad_c = (grid_size_c - cols)/ 2
    pad_c_left = int(np.ceil(pad_c))
    pad_c_right = int(np.floor(pad_c))

    # Pad full stack with 0s
    padded_mask = np.pad(mask_chunk_r, ((pad_r_top, pad_r_bottom), (pad_c_left, pad_c_right), (0, 0)), 'constant')
    padded = np.pad(rescaled, ((pad_r_top, pad_r_bottom), (pad_c_left, pad_c_right), (0, 0)), 'constant')

    # Now determine x and y grid point positions for rescaled coordinates
    y_positions = np.linspace(0, padded.shape[0], n_chunks_r+1)
    y_positions = y_positions.astype(int)
    x_positions = np.linspace(0, padded.shape[1], n_chunks_c+1)
    x_positions = x_positions.astype(int)

    # Determine positions from where to sample from including padding and overlap for rescaled coordinates
    sampling_y_start = np.repeat(overlap[0], n_chunks_r)
    sampling_y_start[0] = pad_r_top
    sampling_y_end = np.repeat(chunk_size[0]-1, n_chunks_r)
    sampling_y_end[-1] = chunk_size[0] - pad_r_bottom

    sampling_x_start = np.repeat(overlap[1], n_chunks_c)
    sampling_x_start[0] = pad_c_left
    sampling_x_end = np.repeat(chunk_size[1]-1, n_chunks_c)
    sampling_x_end[-1] = chunk_size[1] - pad_c_right

    # Append everything to list
    startTime = datetime.now()
    img_chunks = []
    msk_chunks = []
    for i in range(n_chunks_r):
        for j in range(n_chunks_c):
            msk_chunks.append(padded_mask[y_positions[i]:y_positions[i+1], x_positions[j]:x_positions[j+1], :])
            img_chunks.append(padded[y_positions[i]:y_positions[i+1], x_positions[j]:x_positions[j+1], :])

    #%% Run prediction
    # The input shape should be 5 dimensions: (m, n, x, y, z)
    # x, y, z represent the image shape, as you would expect. n is the number of
    # channels. In a standard color video image, you would have 3 channels (red,
    # green, blue). In medical imaging these channels can be separate imaging
    # modalities. m is the batch size or number of samples being passed to the
    # model for training.

    img_data = np.asarray(img_chunks, dtype=np.float32)
    img_shape = [1, 1] + chunk_size

    startTime = datetime.now()

    output = []
    empty = 0
    for idx in range(len(img_data)):
        img_reshaped = np.reshape(img_data[idx], img_shape)
        if msk_chunks[idx].sum() > 0:
            output.append(model.predict(img_reshaped))
        else:
            output.append([])
            empty = empty + 1

    print('Cell prediction time elapsed: ', datetime.now() - startTime)

    #im1 = plt.imshow(msk_chunks[j][:,:,10])
    #im2 = plt.imshow(img_chunks[0][:, :, 50], cmap='gray', alpha=.2, interpolation='bicubic')

    #%% Post-processing to get cell masks
    threshold = 0.25
    startTime = datetime.now()
    centroids = []
    mask_reshaped = []
    for i in range(len(output)):
        if len(output[i]) > 0:
            output_thresh = np.where(output[i] > threshold, 1, 0)
            mask_reshape = np.reshape(msk_chunks[i], img_shape)
            output_mask = np.where(mask_reshape > 0, 1, 0)

            #output_thresh = output_thresh * output_mask
            labels_out = cc3d.connected_components(output_thresh.squeeze())
            labels_idx = np.unique(labels_out)
            centroids.append(ndimage.measurements.center_of_mass(labels_out, labels_out, index=labels_idx[1:len(labels_idx)]))
        else:
            centroids.append([])

    #im1 = plt.imshow(mask_reshaped[100][0,0,:, :, 45], cmap='inferno')
    #im2 = plt.imshow(img_chunks[-1][:, :, 45], cmap='gray', alpha=.3, interpolation='bicubic')

    #%% Centroid trimming and arrangement
    a = 0
    centroids_final = []
    for i in range(n_chunks_r):
        for j in range(n_chunks_c):
            if len(centroids[a]) > 0:
                cen = np.asarray(centroids[a])

                idx = cen.round().astype(int)
                intensity = img_chunks[a][idx[:,0],idx[:,1],idx[:,2]]

                cen = cen[intensity > 0.02]

                cen = cen[cen[:, 0] > sampling_y_start[i]]
                cen = cen[cen[:, 0] <= sampling_y_end[i]]
                cen[:, 0] = cen[:, 0] + y_positions[i] - sampling_y_start[0]

                cen = cen[cen[:, 1] > sampling_x_start[j]]
                cen = cen[cen[:, 1] <= sampling_x_end[j]]
                cen[:, 1] = cen[:, 1] + x_positions[j] - sampling_x_start[0]

                cen = cen/res
                cen = cen[cen[:, 2] > overlap[2]]
                #cen = cen[cen[:, 2] <= (load_chunk_size[2]-overlap[2])]

                cen[:, 0] = cen[:, 0] + y_start
                cen[:, 1] = cen[:, 1] + x_start
                cen[:, 2] = cen[:, 2] + z_start

                centroids_final.append(cen)
            a = a + 1

    centroids_final = np.vstack(centroids_final).round()

    print('Postprocessing time elasped: ', datetime.now() - startTime)
    print('Cells detected:', len(centroids_final))

    #np.savetxt('centroids_' + str(z_start) + '_' + str(z_end) + '.csv', centroids_final.astype(int), delimiter=",")
    with open("centroids2.csv", "ab") as f:
        np.savetxt(f, centroids_final.astype(int), delimiter=",")

    z_start = z_end


#%% Save results to .csv file




