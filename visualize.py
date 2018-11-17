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


# Takes file name and converts to np array
def tiff2array(filename):
    # Special gdal dataset
    dataset = gdal.Open(filename)
    array = np.array(dataset.GetRasterBand(1).ReadAsArray(), np.uint8)

    return array

# TODO Make sure rgb layers are in the right order in returned value.


def image_to_np_array(image_folder):  # Fetches images from download folder
    # Will go through filenames and put the image arrays in images_dict
    images_arr = []
    # Loop through files in downloads directory (if multiple)
    for filename in os.listdir(image_folder):
        if filename.endswith(".tiff"):
            path_to_file = image_folder + '/' + filename
            images_arr.append(tiff2array(path_to_file))
    # Return rgb image in np array format
    im_arr = np.dstack(images_arr)
    arr2jpg(im_arr, 'images')
    return im_arr


def arr2jpg(arr, folder):
    # Turns np array into jpg and saves into folder specified by folder
    scipy.misc.imsave(f'./{folder}/PAIRS_Area.jpg', arr)

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
    plt.xticks(np.arange(0, 6000, 228), range(0, 23))
    plt.yticks(np.arange(0, 6000, 228), range(0, 23))
    plt.show()


# get the image as an array
im_array = image_to_np_array("./downloads")

# get bounding boxes from OSM
bboxes = get_bounding_boxes.get_bounding_boxes(YOLO=True)
# print(len(bboxes)) # We have 2990 buildings. Seems like more than enough
pixels = get_bounding_boxes.OSM_to_pixels(
    im_array.shape[:2], bboxes, YOLO=True)

visualize_bounding_boxes(im_array, pixels, YOLO=True)


# # tile the image, which returns a list
# tile_list = tile.tile_image(im_array, pixels, 228)

# # for each tile, visualize boxes on it
# for i in range(len(tile_list)):
#     elts = tile_list[i]
#     tile_image = elts[0]
#     bboxes_list = elts[1]
#     # print("bounding boxes list for image: " + str(i))
#     # print(bboxes_list)
#     visualize_bounding_boxes(tile_image, bboxes_list)


# # Size of image (3648, 5280, 3)
# # We are aiming for (3648, 5244, 3), as this will divide by 228 (our target image size)
