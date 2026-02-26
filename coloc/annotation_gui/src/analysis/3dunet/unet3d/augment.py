import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img
import random
import itertools
from skimage.filters import gaussian, unsharp_mask
from scipy import ndimage


def scale_image(image, scale_factor):
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(image.affine)
    new_affine[:3, :3] = image.affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
    return new_img_like(image, data=image.get_data(), affine=new_affine)


def flip_image(image, axis):
    try:
        new_data = np.copy(image.get_data())
        for axis_index in axis:
            new_data = np.flip(new_data, axis=axis_index)
    except TypeError:
        new_data = np.flip(image.get_data(), axis=axis)
    return new_img_like(image, data=new_data)


def random_flip_dimensions(n_dimensions):
    axis = list()
    for dim in range(n_dimensions):
        if random_boolean():
            axis.append(dim)
    return axis


def random_scale_factor(n_dim=3, mean=1, std=0.25):
    return np.random.normal(mean, std, n_dim)


def random_boolean():
    return np.random.choice([True, False])


def distort_image(image, flip_axis=None, scale_factor=None, noise_sigma=0.25, blur_max=0,
                  intensity_factors=None, sharp_factor=0):
    # Scale intensity first
    if intensity_factors is not None:
        image = vary_intensity(image, intensity_factors)

    if flip_axis is not None:
        image = flip_image(image, flip_axis)
    if scale_factor is not None:
        image = scale_image(image, scale_factor)
    if noise_sigma > 0:
        image = gaussian_noise(image, noise_sigma)
    if blur_max > 0:
        image = gaussian_blur(image, blur_max)
    if sharp_factor > 0:
        image = sharpen_image(image, sharp_factor)
    return image


def augment_data(data, truth, affine, scale_distortion=None, flip=False, blur=False, scale_intensity=None):
    n_dim = len(truth.shape)

    if flip:
        flip_axis = random_flip_dimensions(n_dim)
    else:
        flip_axis = None

    aug_t = random.randint(1, 5)
    noise_sigma = 0
    blur_max = 0
    intensity_factors = None
    sharp_factor = 0

    if aug_t == 1:
        noise_sigma = random_scale_factor(1, std=scale_distortion)
    elif aug_t == 2:
        blur_max = 1
    elif aug_t == 3:
        intensity_factors = random_scale_factor(2, std=scale_intensity)
    elif aug_t == 4:
        scale_factor = random_scale_factor(n_dim, std=0.25)
    elif aug_t == 5:
        sharp_factor = 1
    
    scale_factor = None
    data_list = list()
    for data_index in range(data.shape[0]):
        image = get_image(data[data_index], affine)
        data_list.append(resample_to_img(distort_image(image,
                                                       flip_axis=flip_axis,
                                                       scale_factor=scale_factor,
                                                       noise_sigma=noise_sigma,
                                                       blur_max=blur_max,
                                                       intensity_factors=intensity_factors,
                                                       sharp_factor=sharp_factor),
                                         image, interpolation="continuous").get_data())
    data = np.asarray(data_list)
    truth_image = get_image(truth, affine)
    truth_data = resample_to_img(distort_image(truth_image, flip_axis=flip_axis, scale_factor=scale_factor,
                                               noise_sigma=0, blur_max=0, intensity_factors=None, sharp_factor=0),
                                 truth_image, interpolation="nearest").get_data()
    return data, truth_data


def get_image(data, affine, nib_class=nib.Nifti1Image):
    return nib_class(dataobj=data, affine=affine)


def generate_permutation_keys():
    """
    This function returns a set of "keys" that represent the 48 unique rotations &
    reflections of a 3D matrix.

    Each item of the set is a tuple:
    ((rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)

    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.

    48 unique rotations & reflections:
    https://en.wikipedia.org/wiki/Octahedral_symmetry#The_isometries_of_the_cube
    """
    return set(itertools.product(
        itertools.product(range(1), range(1)), range(2), range(2), range(2), range(1)))


def random_permutation_key():
    """
    Generates and randomly selects a permutation key. See the documentation for the
    "generate_permutation_keys" function.
    """
    return random.choice(list(generate_permutation_keys()))


def permute_data(data, key):
    """
    Permutes the given data according to the specification of the given key. Input data
    must be of shape (n_modalities, x, y, z).

    Input key is a tuple: (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)

    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.
    """
    data = np.copy(data)
    (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key

    if rotate_y != 0:
        data = np.rot90(data, rotate_y, axes=(1, 3))
    if rotate_z != 0:
        data = np.rot90(data, rotate_z, axes=(2, 3))
    if flip_x:
        data = data[:, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_z:
        data = data[:, :, :, ::-1]
    if transpose:
        for i in range(data.shape[0]):
            data[i] = data[i].T
    return data


def random_permutation_x_y(x_data, y_data):
    """
    Performs random permutation on the data.
    :param x_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :param y_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :return: the permuted data
    """
    key = random_permutation_key()
    return permute_data(x_data, key), permute_data(y_data, key)


def reverse_permute_data(data, key):
    key = reverse_permutation_key(key)
    data = np.copy(data)
    (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key

    if transpose:
        for i in range(data.shape[0]):
            data[i] = data[i].T
    if flip_z:
        data = data[:, :, :, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_x:
        data = data[:, ::-1]
    if rotate_z != 0:
        data = np.rot90(data, rotate_z, axes=(2, 3))
    if rotate_y != 0:
        data = np.rot90(data, rotate_y, axes=(1, 3))
    return data


def reverse_permutation_key(key):
    rotation = tuple([-rotate for rotate in key[0]])
    return rotation, key[1], key[2], key[3], key[4]


def gaussian_noise(image, noise_sigma=0.25):
    new_data = np.copy(image.get_data())
    dim = new_data.shape

    noise_sigma = (np.random.rand(1) + 0.5) * noise_sigma

    noise = np.random.normal(1, noise_sigma, dim)
    new_data = np.multiply(new_data, noise)

    return new_img_like(image, data=new_data)


def gaussian_blur(image, blur_max=3):
    new_data = np.copy(image.get_data())

    blur_sigma = random.randint(1, blur_max) / 2

    new_data = gaussian(new_data, sigma=(blur_sigma, blur_sigma, blur_sigma / 3),
                        preserve_range=True)

    return new_img_like(image, data=new_data)


def vary_intensity(image, intensity_factors=[1, 1]):
    new_data = np.copy(image.get_data())

    # Choose axis
    axis = random.randint(0, 2)
    dims = new_data.shape

    center = random.randint(0, dims[axis])
    prof = gaussian_prof(np.linspace(1, dims[axis], dims[axis]), center, intensity_factors[0] * dims[axis])
    prof = prof * intensity_factors[1]

    if axis is 0:
        data_adj = new_data * prof[:, np.newaxis, np.newaxis]
    elif axis is 1:
        data_adj = new_data * prof[np.newaxis, :, np.newaxis]
    else:
        data_adj = new_data * prof[np.newaxis, np.newaxis, :]

    return new_img_like(image, data=data_adj)


def gaussian_prof(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def sharpen_image(image, sharp_factor=1):
    new_data = np.copy(image.get_data())

    sharpen_radius = random.randint(1, 1)
    sharpen_amount = random.randint(1, sharp_factor)

    new_data = unsharp_mask(new_data, radius=sharpen_radius, amount=sharpen_amount)

    return new_img_like(image, data=new_data)
