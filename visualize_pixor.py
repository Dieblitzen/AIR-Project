from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import matplotlib.ticker as plticker
import get_bounding_boxes
from shapely.geometry.polygon import Polygon
import scipy.misc
import tile
from data_extract import extract_data


# TODO: add option to graph on image at certain path (or maybe just give np array?)

def extract_positive_labels(bboxes):
    return [box for box in bboxes if box[-1] != 0. ]
#applies translation, rotation, then un-translation of four points
def rotate_points(point, center_x, center_y, cos_angle, sin_angle):
    temp_x = point[0] - center_x
    temp_y = point[1] - center_y
    rotated_x = temp_x*cos_angle - temp_y*sin_angle
    rotatedY = tempX*sin(theta) + tempY*cos(theta)
#Converts pixor description of a box into four coordinates.
def pixor_to_corners(box):
    # float tempX = x - cx;
    # float tempY = y - cy;
    #
    # // now apply rotation
    # float rotatedX = tempX*cos(theta) - tempY*sin(theta);
    # float rotatedY = tempX*sin(theta) + tempY*cos(theta);
    #
    # // translate back
    # x = rotatedX + cx;
    # y = rotatedY + cy;
    center_x, center_y, cos_angle, sin_angle, width, length = box
    four_corners = [(center_x+width, center_y+height),
        (center_x+width, center_y-height),
        (center_x-width, center_y-height),
        (center_x-width, center_y+height)]
    rotated_corners = [rotate_point(corner, center_x, center_y, cos_angle, sin_angle) for corner in four_corners]


def visualize_bounding_boxes(image_array, bboxes):
    # Visualize bounding boxes on an image with bb_pixels either as horizontal boxes
    # print(image_array[50])
    plt.imshow(image_array)

    # non_zero = 0
    # for r in range(0, bb_pixels.shape[0]):
    #       for c in range(0, bb_pixels.shape[1]):
    #           pot_box = bb_pixels[r, c]
    #           if not (pot_box[0] == 228.0 and pot_box[1] == 228.0):
    #               non_zero += 1
    # print("number of pixor boxes: " + str(non_zero))


    for box in bboxes:
        coordinates = pixor_to_corners(box)
        poly = Polygon(coordinates)
        x, y = poly.exterior.xy
        plt.plot(x, y)
    for r in range(0, bb_pixels.shape[0]):
      for c in range(0, bb_pixels.shape[1]):
        pot_box = bb_pixels[r, c]
        if not (pot_box[0] == 228.0 and pot_box[1] == 228.0):
          print([pot_box[0], pot_box[1]])
          poly = Polygon([pot_box[0], pot_box[1]])
          x, y = poly.exterior.xy
          plt.plot(x, y)

    # fig, ax = plt.subplots()
    # myInterval=228
    # loc = plticker.MultipleLocator(base=myInterval)
    # ax.xaxis.set_major_locator(loc)
    # ax.yaxis.set_major_locator(loc)

    # # Add the grid
    # ax.grid(which='major', axis='both', linestyle='-')
    # ax.show()
    plt.grid()
    # plt.xticks(np.arange(0, 6000, 228), range(0, 23))
    # plt.yticks(np.arange(0, 6000, 228), range(0, 23))
    plt.show()


# tile the image, which returns a list
images = extract_data("images.pkl")
images = np.asarray(images)
boxlabels = extract_data("box_labels.pkl")
boxlabels = np.asarray(boxlabels)
classlabels = extract_data("class_labels.pkl")
classlabels = np.asarray(classlabels)

for i in range(0, classlabels.shape[0]):
    print("row" + str(i))
    for r in range(0, classlabels.shape[1]):
        for c in range(0, classlabels.shape[2]):
            if classlabels[i, r, c] != 0:
                print(classlabels[i, r, c])

# for each tile, visualize boxes on it
for i in range(len(images)):
    image = images[i]
    boxes_in_image = extract_positive_labels(boxlabels[i])

    visualize_bounding_boxes(tile_image, boxes_in_image)


# # Size of image (3648, 5280, 3)
# # We are aiming for (3648, 5244, 3), as this will divide by 228 (our target image size)
