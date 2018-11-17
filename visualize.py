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


def visualize_bounding_boxes(image_array, bb_pixels, YOLO=True):
    # Visualize bounding boxes on an image with bb_pixels either as horizontal boxes
    # print(image_array[50])
    plt.imshow(image_array)

    if (not YOLO):
        for bbox_coords in bb_pixels:
            poly = Polygon(bbox_coords)
            x, y = poly.exterior.xy
            plt.plot(x, y)
    else:
        for bbox_coords in bb_pixels:
            bottom_left = (bbox_coords[0]-(bbox_coords[2]/2),
                           bbox_coords[1] - (bbox_coords[3]/2))
            bottom_right = (
                bbox_coords[0]+(bbox_coords[2]/2), bbox_coords[1] - (bbox_coords[3]/2))
            top_left = (bbox_coords[0]-(bbox_coords[2]/2),
                        bbox_coords[1]+(bbox_coords[3]/2))
            top_right = (bbox_coords[0]+(bbox_coords[2]/2),
                         bbox_coords[1]+(bbox_coords[3]/2))

            coords = [bottom_left, top_left, top_right, bottom_right]
            poly = Polygon(coords)
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
tile_list = extract_data("tiles.pkl")

# for each tile, visualize boxes on it
for i in range(len(tile_list)):
    elts = tile_list[i]
    tile_image = elts[0]
    bboxes_list = elts[1]
    # print("bounding boxes list for image: " + str(i))
    # print(bboxes_list)
    visualize_bounding_boxes(tile_image, bboxes_list)


# # Size of image (3648, 5280, 3)
# # We are aiming for (3648, 5244, 3), as this will divide by 228 (our target image size)
