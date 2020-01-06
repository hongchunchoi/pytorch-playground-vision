import os
import pickle
import numpy as np


data_path = "data/"
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file



def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.
    If filename=="" then return the directory of the files.
    """

    return os.path.join(data_path, "cifar-10-batches-py/", filename)


def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """
    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_batch_file(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """
    # Load the pickled data-file.
    data = _unpickle(filename)
    # Get the raw images.
    raw_images = data[b'data']
    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])
    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls

def _load_data() :
    '''batch 단위 파일 불러와 한번에 합치기 / Test dataset 포함'''
    train_X = []
    train_Y = []

    for i in range(5):
        X, Y = _load_batch_file("data_batch_" + str(i + 1))
        train_X.append(X)
        train_Y.append(Y)

    train_data = np.concatenate(train_X)
    train_labels = np.concatenate(train_Y)
    del X, Y
    test_data, test_labels = _load_batch_file('test_batch')
    return train_data, train_labels, test_data, test_labels



def _split_by_label(n_label, data_X, data_Y) :
    '''load 된 데이터 label별로 나누는 함수
    '''
    list = []
    for k in range(n_label):
        # list[k] = []
        k_label = np.where(data_Y == k)
        data = data_X[k_label]
        label = data_Y[k_label]
        new_list = [data, label]
        list.append(new_list)

    return list


