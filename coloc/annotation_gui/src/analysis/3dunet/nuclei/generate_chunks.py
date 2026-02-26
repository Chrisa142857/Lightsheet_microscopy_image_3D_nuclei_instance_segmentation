import os
import cv2
import numpy as np
import cc3d
import argparse
import mat73

from datetime import datetime
from scipy.io import loadmat
from scipy import ndimage
from skimage.transform import resize
from skimage.exposure import rescale_intensity

from unet3d.training import load_old_model
from nuclei.img_utils import calculate_rescaling_intensities, measure_colocalization, remove_touching_df

parser = argparse.ArgumentParser(description='Predict from validation dataset.')
parser.add_argument('--mat', metavar='mat', type=str, nargs='+',
                    help='Whether called by matlab function')
parser.add_argument('--i', metavar='i', type=str, nargs='+',
                    help='Input image directory')
parser.add_argument('--o', metavar='i', type=str, nargs='+',
                    help='Output image directory')
parser.add_argument('--m', metavar='m', type=str, nargs='+',
                    help='Input mask location')
parser.add_argument('--r', metavar='r', type=str, nargs='+',
                    help='Resolution tag')
parser.add_argument('--g', metavar='g', type=str, nargs='+',
                    help='GPU tag')
args = parser.parse_args()

if args.g:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.g[0])
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

################################
## Extra options for running model on GPU with limited memory
#import tensorflow as tf
#from keras import backend as k
 
#config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
#config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
#config.gpu_options.per_process_gpu_memory_fraction = 0.25
 
# Create a session with the above options specified.
#k.tensorflow_backend.set_session(tf.Session(config=config))

################################
##### Set parameters

# Image and model configuration parameters
# Dimensions should correspond to [x,y,z]
chunk_size = [112, 112, 32]  # Model chunk size in voxels
overlap = [16, 16, 8]  # Overlap between chunks in voxels
pred_threshold = 0.5  # Prediction threshold. Default is 0.5
int_threshold = 400  # Minimum intensity of positive cells. Otherwise set to None
normalize_intensity = True  # Whether to normalize intensities using min/max. Recommended: true
resample_chunks = False  # Whether to resample image to match trained image resolution. Note: increases computation time
tree_radius = 2     # Pixel radius for removing centroids near each other

measure_coloc = False  # Measure intensity of co-localizaed channels
n_channels = 3  # Total number of channels
img_list = []   # Empty image list

##### Load parameters from config.mat matlab structure
if args.mat:
    # Load parameters from matlab structure
    from_matlab = True
    config = mat73.loadmat(args.mat[0])

    input_img_directory = config['config']['img_directory']
    output_directory = config['config']['output_directory']
    img_list = config['config']['img_list']
    img_list = [[item for sublist in img_list for item in sublist]]

    model_file = os.path.join(os.getenv('PYTHONPATH'), 'nuclei', 'models', config['config']['model_file'])

    chunk_size = [int(s) for s in config['config']['chunk_size']]
    overlap = [int(s) for s in config['config']['chunk_overlap']]

    acquired_img_resolution = list(config['config']['resolution'])
    trained_img_resolution = list(config['config']['trained_resolution'])
    
    # Set acquired to trained resolution if calling from NuMorph
    trained_img_resolution = acquired_img_resolution

    int_threshold = config['config']['min_intensity']

    use_mask = config['config']['use_annotation_mask']
    mask_resolution = config['config']['resample_resolution']
    if use_mask == 'true':
        use_mask = True
        mask_file = config['config']['mask_file']
    else:
        use_mask = False

    resample_chunks = config['config']['resample_chunks']
    if resample_chunks == 'true':
        resample_chunks = True
    else:
        resample_chunks = False

    save_name = config['config']['path_save']
else:
    from_matlab = False

#####
save_name = os.path.join(output_directory, save_name)
print('Saving results to: ', save_name)

#####
# Load model
model = load_old_model(model_file)

# Load mask
if use_mask:
    print('Loading mask...')
    #mask = loadmat(mask_file)
    mask = mat73.loadmat(mask_file)
    mask = mask['I_mask']
else:
    mask = np.ones((1, 1, 1))

# Determine chunk sizes while considering scaling and chunk overlap
res = [i / j for i, j in zip(acquired_img_resolution, trained_img_resolution)]
mask_res = [i / j for i, j in zip(acquired_img_resolution, mask_resolution)]

load_chunk_size = [round(i / j) for i, j in zip(chunk_size, res)]

# Taking only channel 1 images (assumed to be cell nuclei)
files = os.listdir(input_img_directory)

# Take img_list for each channel. Nuclei is channel should be first
if not measure_coloc:
    n_channels = 1

if not img_list:
    for i in range(n_channels):
        matching = [s for s in files if "C" + str(i + 1) in s]
        matching = sorted(matching)
        # Read only the number of images for each chunk
        img_list.append([input_img_directory + '/' + s for s in matching])

total_slices = len(img_list[0])

# z_pad = np.ceil(total_images / load_chunk_size[2])
# n_chunks = np.ceil((total_images + z_pad) / load_chunk_size[2]).astype(int)

# chunk_start = np.arrange(0, n_chunks * (load_chunk_size[2] - overlap[2]), load_chunk_size[2] - overlap[2])
# chunk_end = chunk_start[1:] + overlap[2]
# chunk_end = np.append(chunk_end, total_images - 1)

chunk_start = np.arange(0, total_slices, step=(chunk_size[2] - overlap[2]))
chunk_end = chunk_start + chunk_size[2]
n_chunks = len(chunk_start)

# Resize mask to match the number of slices in input image directory
mask = resize(mask, (mask.shape[0], mask.shape[1], total_slices), order=0)
mask_res[-1] = 1

# Calculate rescaling intensity values
if normalize_intensity:
    intensity_values = calculate_rescaling_intensities(img_list[0], sampling_range=10)
    if int_threshold is not None:
        int_threshold = rescale_intensity(np.asarray(int_threshold, dtype=np.float32),
                                          in_range=intensity_values)

# Calculate rescaling factor if resolutions are different
if acquired_img_resolution != trained_img_resolution and resample_chunks:
    rescale_factor = [i / j for i, j in zip(acquired_img_resolution, trained_img_resolution)]
else:
    rescale_factor = None

# Read first image to get sizes
tempI = cv2.imread(img_list[0][0], -1)
[rows, cols] = tempI.shape

# Begin cell counting
total_cells = 0
total_time = datetime.now()

for n in range(n_chunks):
    print('Working on chunk', n + 1, 'out of', n_chunks)
    startTime = datetime.now()
    z_start = chunk_start[n]
    z_end = chunk_end[n]

    # If last chunk, add padding
    if z_end > total_slices:
        z_end = total_slices
        add_end_padding = True
    else:
        add_end_padding = False

    # Skip chunk if nothing is present
    if not mask[:, :, z_start:z_end].any():
        continue

    # Take the mask for the respective z positions
    mask_chunk1 = [cv2.resize(mask[:, :, z], (cols, rows), interpolation=0) for z in range(z_start, z_end)]
    mask_chunk1 = np.asarray(mask_chunk1, dtype=bool)
    mask_chunk = np.swapaxes(mask_chunk1, 0, 1).swapaxes(1, 2)

    # Read images
    print('Reading slices', z_start, 'through', z_end)
    images = [cv2.imread(file, -1) for file in img_list[0][z_start:z_end]]
    images = np.asarray(images, dtype=np.float32)
    images = np.swapaxes(images, 0, 1).swapaxes(1, 2)

    # If last chunk, then pad bottom to make it fit into the model
    if add_end_padding:
        print('Padding Chunk End...')
        end_pad = chunk_size[2] - images.shape[2]
        mask_chunk = np.pad(mask_chunk, ((0, 0), (0, 0), (0, end_pad)), 'constant')
        images = np.pad(images, ((0, 0), (0, 0), (0, end_pad)), 'mean')

    # Rescale intensity
    if normalize_intensity:
        print('Rescaling Intensity...')
        images_rescaled = rescale_intensity(images, in_range=intensity_values)
    else:
        images_rescaled = images

    # Rescale image size
    if rescale_factor is not None:
        print('Rescaling Size...')
        images_rescaled = ndimage.zoom(images_rescaled, rescale_factor, order=1)

    [rows, cols, slices] = images_rescaled.shape

    # Calculate y and x positions to sample image chunks
    x_positions = np.arange(0, cols, step=(chunk_size[0] - overlap[0]))
    y_positions = np.arange(0, rows, step=(chunk_size[1] - overlap[1]))

    n_chunks_c = len(x_positions)
    n_chunks_r = len(y_positions)

    # Append image and mask chunks to list
    img_chunks = []
    msk_chunks = []
    for i in range(n_chunks_r):
        for j in range(n_chunks_c):
            msk_chunk = mask_chunk[y_positions[i]:y_positions[i] + chunk_size[0],
                        x_positions[j]:x_positions[j] + chunk_size[1], :]
            img_chunk = images_rescaled[y_positions[i]:y_positions[i] + chunk_size[0],
                        x_positions[j]:x_positions[j] + chunk_size[1], :]

            if img_chunk.shape != tuple(load_chunk_size):
                pad_c_right = int(load_chunk_size[1] - img_chunk.shape[1])
                pad_r_bottom = int(load_chunk_size[0] - img_chunk.shape[0])

                msk_chunk = np.pad(msk_chunk, ((0, pad_r_bottom), (0, pad_c_right), (0, 0)), 'constant')
                img_chunk = np.pad(img_chunk, ((0, pad_r_bottom), (0, pad_c_right), (0, 0)), 'constant')

            msk_chunks.append(msk_chunk)
            img_chunks.append(img_chunk)

    print('Images prepared in: ', datetime.now() - startTime)

    # Run prediction
    # The input shape should be 5 dimensions: (m, n, x, y, z)
    # x, y, z represent the image shape, as you would expect. n is the number of
    # channels. In a standard color video image, you would have 3 channels (red,
    # green, blue). In medical imaging these channels can be separate imaging
    # modalities. m is the batch size or number of samples being passed to the
    # model for training.
    startTime = datetime.now()
    output = []
    img_shape = (1, 1) + tuple(chunk_size)
    empty_chunk = np.zeros(img_shape)
    empty_idx = np.zeros(len(img_chunks))
    img_reshaped = []
    msk_reshaped = []
    for idx, chunk in enumerate(img_chunks):
        img_reshaped.append(np.reshape(img_chunks[idx], img_shape))
        msk_reshaped.append(np.reshape(msk_chunks[idx], img_shape))
        if msk_reshaped[idx].any() and (img_reshaped[idx] > int_threshold).any():
            output.append(model.predict(img_reshaped[idx]) * msk_reshaped[idx])
        else:
            output.append(empty_chunk)
            empty_idx[idx] = 1
    output = [np.squeeze(chunk) for chunk in output]
    print('Object mask prediction time elapsed: ', datetime.now() - startTime)

    # Calculate connected components and determine centroid positions
    startTime = datetime.now()
    a = 0
    cen = []
    for i in range(n_chunks_r):
        for j in range(n_chunks_c):
            if empty_idx[a] != 1:
                # Calculate final prediction mask
                output_thresh = np.where(output[a] > pred_threshold, 1, 0)

                # Label connected components
                labels_out = cc3d.connected_components(output_thresh)
                n_cells = np.max(labels_out)

                # Find centroids
                centroids = ndimage.measurements.center_of_mass(output_thresh, labels_out,
                                                                index=np.arange(1, n_cells + 1))
                centroids = np.asarray(centroids).round()

                # Remove cells with low intensity
                if int_threshold is not None:
                    img_chunk = img_chunks[a]
                    int_high = [img_chunk[tuple(centroids[c].astype(int))] > int_threshold for c in
                                range(len(centroids))]
                    centroids = centroids[int_high]

                if centroids.any():
                    # Remove centroids along borders
                    centroids = centroids[centroids[:, 0] > overlap[0] / 2]
                    centroids = centroids[centroids[:, 0] <= chunk_size[0] - overlap[0] / 2]

                    centroids = centroids[centroids[:, 1] > overlap[1] / 2]
                    centroids = centroids[centroids[:, 1] <= chunk_size[1] - overlap[1] / 2]

                    centroids = centroids[centroids[:, 2] > overlap[2] / 2]
                    centroids = centroids[centroids[:, 2] <= chunk_size[2] - overlap[2] / 2]

                    # Adjust centroid positions
                    centroids[:, 0] += y_positions[i]
                    centroids[:, 1] += x_positions[j]
                    centroids[:, 2] += chunk_start[n]

                    # Append to array
                    cen.append(centroids)
            a += 1

    if not cen:
        continue


    cent = np.concatenate(cen)
    cent = cent[cent[:, 2].argsort()]

    print('Postprocessing time elapsed: ', datetime.now() - total_time)
    print('Nuclei counted: ', cent.shape[0])

    total_cells += cent.shape[0]
    print('Total nuclei counted: ', total_cells)
    
    # Continue if no nuclei present
    if total_cells == 0:
        continue

    # Get mask structure id's
    if use_mask:
        structure_idx = [mask[tuple(np.floor(c * mask_res).astype(int))] for c in cent]
    else:
        structure_idx = np.ones(cent.shape[0])
    cent = np.append(cent, np.array(structure_idx)[:, None], axis=1)

    # Remove stray cells with no id
    rm_idx = cent[:, 3] == 0
    cent = cent[~rm_idx, :]
    total_cells += -sum(rm_idx)
    print('Removed ' + str(sum(rm_idx)) + ' empty nuclei')

    # Continue if no nuclei present
    if not cent.any():
        continue

    # Remove touching cells
    cent_rm = remove_touching_df(cent, radius=tree_radius)
    ncent_rm = cent.shape[0] - cent_rm.shape[0]
    total_cells += -ncent_rm
    print('Removed ' + str(ncent_rm) + ' touching nuclei')
    cent = cent_rm

    # Measure intensities in other channels
    if measure_coloc and cent.any():
        print('Measuring co-localization...')
        cent[:, 2] -= z_start
        for i in range(n_channels):
            intensities = measure_colocalization(cent, img_list[i][z_start:z_end])

            # Throwing errors
            cent = np.append(cent, intensities[:, None], axis=1)
        cent[:, 2] += z_start

    # If called by MATLAB, adjust for base 1 indexing
    if from_matlab:
        cent += 1

    # Write .csv file
    if n == 0:
        np.savetxt(save_name, cent.round().astype(int), delimiter=",", fmt='%u')
    else:
        with open(save_name, "ab") as f:
            np.savetxt(f, cent.round().astype(int), delimiter=",", fmt='%u')

# One final pass on all centroids to remove potentially touching nuclei in z



print('Total nuclei counted: ', total_cells)
print('Total time elapsed: ', datetime.now() - total_time)
