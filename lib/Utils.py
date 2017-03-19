import numpy as np
import os
import glob
import cv2
import csv
import matplotlib.pyplot as plt

def apply_threshold(img, threshold):
    """
    Apply threshold to image
    :param img: Single channel 2d array
    :param threshold:
    :return:
    """
    # Zero out pixels below the threshold
    img[img <= threshold] = 0
    return img


def one_channel_to_gray(image):
    """
    Convert single channel grayscale image to 3 channel color image
    :param image: Grayscale image with one channel
    :return: Grayscale image with three channels
    """
    return np.dstack((image, image, image))


def read_project_data(folder, label):
    """
    Read labeled from KTTI dataset
    :param folder: Folder where to look for images
    :param label: Label that these image have (0: no vehicle, 1: vehicle)
    :return: tuple consisting of path to image and label
    """
    data = []
    for root, dirs, files in os.walk(folder):
        for d in dirs:
            all_images = glob.glob(os.path.join(root, d, "*.png"))
            for img in all_images:
                data.append((img, label))

    return data


def read_udacity_data(folder):
    """
    Read labeled vehicle and non_vehicle data from udacity data
    :param folder: Location of data
    :return: tuple consisting of path to image and label for non_vehicle and vehicle data
    """
    non_vehicle = []
    vehicle = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith("labels.csv"):

                with open(os.path.join(root, file), 'r') as f:

                    # Check which folder is processed and adapt data processing to this
                    crowdai_structure = ("object-detection-crowdai" in root)

                    # Skip header for object-detection-crowdai data
                    if crowdai_structure:
                        reader = csv.reader(f)
                        all_data = list(reader)
                        all_data = all_data[1:]
                    else:
                        reader = csv.reader(f, delimiter=' ')
                        all_data = list(reader)

                    for ind in range(0, len(all_data)):
                        data = all_data[ind]

                        if crowdai_structure:
                            bbox = data[0:4]
                            filename = data[4]
                            label = data[5]
                        else:
                            bbox = data[1:5]
                            filename = data[0]
                            label = data[6]

                        bbox = list(map(int, bbox))
                        bbox = (tuple(bbox[0:2]), tuple(bbox[2:4]))

                        if ("car" in label) or ("Car" in label):
                            vehicle.append((os.path.join(root, filename), 1, bbox))
                        else:
                            non_vehicle.append((os.path.join(root, filename), 0, bbox))

    return non_vehicle, vehicle


def draw_boxes(img, boxes, color=(0, 0, 255), thickness=6):
    """
    Draw
    :param img: Image in which bounding boxes should be drawn
    :param boxes: Bounding boxes that should be drawn
    :param color: Color used for bounding box as 3-RGB-tupel (Default: (0, 0, 255) / Blue)
    :param thickness: Thickness used for lines of bounding box
    :return:
    """
    # Make a copy of the image
    draw_img = np.copy(img)

    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    for el1, el2 in boxes:
        cv2.rectangle(draw_img, el1, el2, color, thickness)

    return draw_img


def label_to_bbox(label, car_number):
    """
    Convert output from scipy.ndimage.measurements's label function to bounding box
    :param label: Current label
    :param car_number: Car index for which bounding box should be generated
    :return: bounding box for this car
    """
    # Find pixels with each car_number label value
    nonzero = (label == car_number).nonzero()

    # Identify x and y values of those pixels
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Define a bounding box based on min/max x and y
    return (np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))


def draw_labeled_box(img, labels):
    """
    Draw bounding box around labeled vehicles detected in an image
    :param img: Image in which bounding boxes should be drawn
    :param labels: Vehicle labels as returned by scipy.ndimage.measurements's label function
    :return: Image with detected vehicles marked by a bounding box
    """

    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        bbox = label_to_bbox(labels[0], car_number)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    # Return the image
    return img


def cvt_color_string_to_cv2(input_string):
    """
    Convert input color string to cv2 conversion method
    :param input_string: Color space that should be used
    :return: cv2 conversion method
    """
    if input_string == "RGB":
        return cv2.COLOR_RGB2BGR
    elif input_string == "HLS":
        return cv2.COLOR_RGB2HLS
    elif input_string == "HSV":
        return cv2.COLOR_RGB2HSV
    elif input_string == "YCRCB":
        return cv2.COLOR_RGB2YCR_CB
    else:
        return None


def plot_images(filename, images, title, rows=None, cols=None, font_size=6):
    """
    Create a graphic for given list of images
    :param filename: Store the graphics with this file name
    :param images: List of images
    :param title: List of titles for each image
    :param rows: Number of rows
    :param cols: Number of columns
    :param font_size: Font size used in graphics
    """

    if (rows is None) and (cols is None):
        rows = len(images)
        cols = 1
    elif rows is None:
        rows = np.int(np.round(np.float(len(images)) / np.float(cols)))
    elif cols is None:
        cols = np.int(np.round(np.float(len(images)) / np.float(rows)))

    for i, img in enumerate(images):
        fig = plt.subplot(rows, cols, i+1)

        # Use hot colormap if image is grayscale
        if len(img.shape) < 3:
            plt.imshow(img, cmap='hot')
        else:
            plt.imshow(img)

        # Set title and deactivate axis visualization
        plt.title(title[i], fontsize=font_size)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.savefig(filename, bbox_inches='tight')
