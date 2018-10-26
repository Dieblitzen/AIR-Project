from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import get_bounding_boxes
import merge_data
from shapely.geometry.polygon import Polygon

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
            path_to_file = image_folder + '/' + filename
            images_dict[ind] = tiff2array(path_to_file)
            ind += 1
    ## Return rgb image in np array format
    return np.transpose(np.array(list(images_dict.values())))

## TODO: add option to graph on image at certain path (or maybe just give np array?)
def visualize_bounding_boxes(image_array, bb_pixels): 
    ## Visualize bounding boxes on an image with bb_pixels either as horizontal boxes (horizontal=True)
    plt.imshow(image_array)
    for bbox_coords in bb_pixels: 
        poly = Polygon(bbox_coords)
        x,y = poly.exterior.xy
        plt.plot(x,y)

    plt.show()


lat_min = 41.0100756423
lat_max = 41.0338409682
lon_min = -73.7792749922
lon_max = -73.7582464736

bboxes = get_bounding_boxes.get_bounding_boxes(lat_min,lon_min,lat_max,lon_max,horizontal=True)
im_array = image_to_np_array("./downloads")

visualize_bounding_boxes(np.rot90(im_array,1), 
               merge_data.OSM_to_pixels([lat_min,lon_min,lat_max,lon_max],im_array.shape[:2], bboxes))

