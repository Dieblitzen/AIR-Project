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
from data_extract import extract_data, image_to_np_array
import math
from get_bounding_boxes import get_two_closest_points, convert_coord_to_pixel, corner_boxes_in_pixels, LON_WIDTH, LAT_HEIGHT, LAT_MAX, LON_MIN

ENTIRE_IMG_SIZE = (3648, 5280, 3)

# TODO: add option to graph on image at certain path (or maybe just give np array?)

def extract_positive_labels(bboxes):
    unique_boxes_set = set()
    unique_boxes = []
    for r in range(0, bboxes.shape[0]):
        for c in range(0, bboxes.shape[1]):
            if bboxes[r,c][-1] != 0. and tuple(bboxes[r,c][2:]) not in unique_boxes_set:
                unique_boxes_set.add(tuple(bboxes[r,c][2:]))
                unique_boxes.append(bboxes[r,c])

    return unique_boxes
#applies translation, rotation, then un-translation of four points
def rotate_point(point, center_x, center_y, cos_angle, sin_angle):
    temp_x = point[0] - center_x
    temp_y = point[1] - center_y
    cos_angle = np.cos(2*math.pi - np.arccos(cos_angle))
    sin_angle = np.sin(2*math.pi - np.arcsin(sin_angle))

    rotated_x = temp_x*cos_angle - temp_y*sin_angle
    rotated_y = temp_x*sin_angle + temp_y*cos_angle
    x = rotated_x + center_x
    y = rotated_y + center_y
    return x, y
#Converts pixor description of a box into four coordinates.
def pixor_to_corners(box):

    center_x, center_y, cos_angle, sin_angle, width, length = box
    four_corners = [(center_x+width/2, center_y+length/2),
        (center_x+width/2, center_y-length/2),
        (center_x-width/2, center_y-length/2),
        (center_x-width/2, center_y+length/2)]
    rotated_corners = [rotate_point(corner, center_x, center_y, cos_angle, sin_angle) for corner in four_corners]
    return rotated_corners

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
# for i in range(0, classlabels.shape[0]):
#     print("row" + str(i))
#     for r in range(0, classlabels.shape[1]):
#         for c in range(0, classlabels.shape[2]):
#             if classlabels[i, r, c] != 0:
#                 print(classlabels[i, r, c])


# im_array = image_to_np_array("./downloads/")
# for box in boxlabels:
#     for r in range(0, box.shape[0]):
#         for c in range(0, box.shape[1]):
#             if box[r,c][-1] != 0.:
#                 print(box[r,c])

#VISUALIZING BASED OFF OF LABELS IN PICKLE, ONLY DOING UNIQUE BOXES
# boxes_in_image = [extract_positive_labels(b) for b in boxlabels]
# visualize_bounding_boxes(image, boxes_in_image)

#VISUALIZING BASED OFF OF ONLY OSM DATA, NOT LABELS:
# get bounding boxes from OSM
# bboxes = get_bounding_boxes.get_bounding_boxes(YOLO = False)
# print(len(bboxes)) # We have 3975 buildings. Seems like more than enough
# pixel_boxes = get_bounding_boxes.OSM_to_pixels(im_array.shape[:2], bboxes, YOLO=False, PIXOR = True)
# converted_corner_boxes = corner_boxes_in_pixels(im_array.shape, bboxes)
# print(len(converted_corner_boxes))
# visualize_bounding_boxes(im_array, converted_corner_boxes)


# VISUALIZING FOR EACH TILE
counter = 0
for i in range(len(images)):
    image = images[i]
    boxes_in_image = extract_positive_labels(boxlabels[i])
    counter+= len(boxes_in_image)
    # print(len(boxes_in_image))

    # visualize_bounding_boxes(image, boxes_in_image)
print(counter)

# # Size of image (3648, 5280, 3)
# # We are aiming for (3648, 5244, 3), as this will divide by 228 (our target image size)
