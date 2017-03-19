import numpy as np
import cv2
from skimage.feature import hog


def get_spatial_features(img, size=(32, 32)):
    """
    Resize input image to specified size and return pixel values as a features
    :param img: Input image
    :param size: Size that image should be transformed to
    :return: Return stacked feature vector of the three color channel
    """
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()

    return np.hstack((color1, color2, color3))


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Calculate histogram of gradient features
    :param img:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param vis:
    :param feature_vec:
    :return: If vis=False return features, for vis=True return features and visualization
    """
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def get_color_hist_features(img, n_bins):
    """
    Calculate histogram for each color channel and create a feature vector
    :param img: Image with 3 color channels
    :param n_bins: Number of bins used for histogram
    :return: Feature vector containing a concatenate feature vector for all 3 channels
    """
    channel1_hist = np.histogram(img[:, :, 0], bins=n_bins)
    channel2_hist = np.histogram(img[:, :, 1], bins=n_bins)
    channel3_hist = np.histogram(img[:, :, 2], bins=n_bins)

    # Concatenate the histograms into a single feature vector
    return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
