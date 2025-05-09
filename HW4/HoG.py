import os
import numpy as np
import gzip
from skimage.feature import hog # you can install skimage by using  conda install scikit-image
from tqdm import tqdm

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as impath:
        images = np.frombuffer(impath.read(), np.uint8, offset=16).reshape(len(labels),28,28)

    return images, labels

def preprocess_save_hog():
    X_train = np.load('X_train_sampled.npy')
    train_hog_features = []
    for image in tqdm(X_train):
        arr = hog(image,
                  orientations=9,
                  pixels_per_cell=(2, 2),
                  cells_per_block=(1, 1),
                  visualize=False)
        train_hog_features.append(arr)
    train_hog_features = np.array(train_hog_features)
    np.save('X_train_hog.npy', train_hog_features)

    X_test = np.load('X_test_sampled.npy')
    test_hog_features = []
    for image in tqdm(X_test):
        arr = hog(image,
                  orientations=9,
                  pixels_per_cell=(2, 2),
                  cells_per_block=(1, 1),
                  visualize=False)
        test_hog_features.append(arr)
    test_hog_features = np.array(test_hog_features)
    np.save('X_test_hog.npy', test_hog_features)

if __name__ == '__main__':

    preprocess_save_hog()

    # The shape of the processed training set X_train is 12000x28x28, where 12000 is the sample size. Replace the following address with your own dataset address.
    # The hog features of the test set can be obtained in the same way.
    # X_train = np.load('X_train_sampled.npy')

    # Expand dimensions to 28x28x1 for an image x.
    # The following is an example of hog feature and visualization for an image. Replace the image with the training set or the test set (Set the hog function "visualize=False").
    # image = X_train[0]

    # Get HoG feature h_x of an image
    # arr: 28/2 * 28/2 * 9 = 1764
    # hog_image: 28 * 28 = 784
    # arr, hog_image = hog(image, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualize=True)

    # Reshape HoG feature h_x into 784 dimensions
    # hog_vector = hog_image.reshape(-1)
    # print('hog_vector shape is', hog_vector.shape)