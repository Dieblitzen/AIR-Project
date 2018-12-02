## Data extraction and saving

import pickle
import tile
import get_bounding_boxes
import os
from osgeo import gdal
import numpy as np
import scipy.misc

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
    file_names = os.listdir(image_folder)
    file_names.sort(reverse=True)
    for filename in file_names:
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



def save_data(tiles):
    with open("tiles3.pkl", "wb") as f:
        pickle.dump(tiles,f)


def extract_data(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    # get the image as an array
    im_array = image_to_np_array("./downloads")

    # get bounding boxes from OSM
    bboxes = get_bounding_boxes.get_bounding_boxes(YOLO=True)
    # print(len(bboxes)) # We have 2990 buildings. Seems like more than enough
    pixels = get_bounding_boxes.OSM_to_pixels(
        im_array.shape[:2], bboxes, YOLO=True)

    # visualize_bounding_boxes(im_array, pixels, YOLO=True)
    indices_to_remove = list(range(4,24)) + list(range(3+23,24+23)) + [x+(23*2) for x in [2,3,4,5,6,7,8,12,13,14,15,16,17,17,22]] +[x+(23*3) for x in [0,10,11,12,13,14,15,16,22]] +[x+(23*4) for x in [10,11,13,14,15]] +[x+(23*5) for x in [0,8,9,10,11,14,15,16,17,18]] +[x+(23*6) for x in [6,7,8,9,10,15,16,17,18]] +[x+(23*7) for x in [10,11,17,18,19,22]] +[x+(23*8) for x in [18,19,20,21]] +[x+(23*9) for x in [20,21,22]] +[x+(23*10) for x in [4,5,6,20,21,22]] +[x+(23*11) for x in [5,6,7,20,21,22]] +[x+(23*12) for x in [1,2,20,21,22]] + [x+(23*13) for x in [1,2]] + [x+(23*16) for x in [0,1]] + [x+(23*17) for x in [0,1]] + [x+(23*18) for x in [0,1,2]] + [x+(23*19) for x in [0,1,2,3,17,18]] + [x+(23*20) for x in [0,1,2,3,4,17,18]] + [x+(23*21) for x in [0,1,2,3,4,5]] + [x+(23*22) for x in [0,1,2,3,4,5]]

    # tile the image, which returns a list
    tile_list = tile.tile_image(im_array, pixels, 228, indices_to_remove)
    save_data(tile_list)
