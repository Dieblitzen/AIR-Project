from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches

## Takes file name and converts to np array
def tiff2array(filename):
    ## Special gdal dataset
    dataset = gdal.Open(filename)
    array = np.array(dataset.GetRasterBand(1).ReadAsArray(), np.uint8)
    return array

### TODO Make sure rgb layers are in the right order in returned value. 
def image_to_np_array(image_folder): # Fetches images from download folder 
    ## Will go through filenames and put the image arrays in images_dict
    images_dict = {}
    ## Loop through files in downloads directory (if multiple)
    ind = 0
    for filename in os.listdir(image_folder):
        if filename.endswith(".tiff"): 
            path_to_file = path + '/' + filename
            images_dict[ind] = tiff2array(path_to_file)
            ind += 1
    ## Return rgb image in np array format
    return np.transpose(np.array(list(images_dict.values())))

## TODO: add option to graph on image at certain path (or maybe just give np array?)
def visualize_bounding_boxes(image_array, bb_pixels): 
    plt.imshow(im_array)
    for bbox_coords in bb_pixels: 
        poly = Polygon(bbox_coords)
        x,y = poly.exterior.xy
        plt.plot(x,y)

    plt.show()