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

    # non_zero = 0
    # for r in range(0, bb_pixels.shape[0]):
    #       for c in range(0, bb_pixels.shape[1]):
    #           pot_box = bb_pixels[r, c]
    #           if not (pot_box[0] == 228.0 and pot_box[1] == 228.0):
    #               non_zero += 1
    # print("number of pixor boxes: " + str(non_zero))

    if (not YOLO):
        for r in range(0, bb_pixels.shape[0]):
          for c in range(0, bb_pixels.shape[1]):
            pot_box = bb_pixels[r, c]
            if not (pot_box[0] == 228.0 and pot_box[1] == 228.0):
              print(pot_box)
              poly = Polygon([pot_box[0], pot_box[1]])
              x, y = poly.exterior.xy
              plt.plot(x, y)

        # for bbox_coords in bb_pixels:
        #     poly = Polygon(bbox_coords)
        #     x, y = poly.exterior.xy
        #     plt.plot(x, y)

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
tile_list = extract_data("tiles2.pkl")

# for each tile, visualize boxes on it
for i in range(len(tile_list)):
    elts = tile_list[i]
    tile_image = elts[0]
    pixel_to_box_matrix = elts[1]

    visualize_bounding_boxes(tile_image, pixel_to_box_matrix, YOLO=False)


# # Size of image (3648, 5280, 3)
# # We are aiming for (3648, 5244, 3), as this will divide by 228 (our target image size)
