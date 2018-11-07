from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import get_bounding_boxes
import merge_data
from shapely.geometry.polygon import Polygon
import scipy.misc


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
    im_arr = np.transpose(np.array(list(images_dict.values())))
    arr2jpg(im_arr, 'images')
    return im_arr

def arr2jpg(arr, folder):
    ## Turns np array into jpg and saves into folder specified by folder
    scipy.misc.imsave(f'./{folder}/PAIRS_Area.jpg', arr)

## TODO: add option to graph on image at certain path (or maybe just give np array?)
def visualize_bounding_boxes(image_array, bb_pixels, YOLO=True): 
    ## Visualize bounding boxes on an image with bb_pixels either as horizontal boxes 
    print(image_array[50])
    plt.imshow(image_array)

    if (not YOLO) :
        for bbox_coords in bb_pixels: 
            poly = Polygon(bbox_coords)
            x,y = poly.exterior.xy
            plt.plot(x,y)
    else: 
        for bbox_coords in bb_pixels: 
            bottom_left = (bbox_coords[0]-(bbox_coords[2]/2), bbox_coords[1]- (bbox_coords[3]/2))
            bottom_right = (bbox_coords[0]+(bbox_coords[2]/2), bbox_coords[1]- (bbox_coords[3]/2))
            top_left = (bbox_coords[0]-(bbox_coords[2]/2), bbox_coords[1]+(bbox_coords[3]/2))
            top_right = (bbox_coords[0]+(bbox_coords[2]/2), bbox_coords[1]+(bbox_coords[3]/2))

            coords = [bottom_left,top_left,top_right,bottom_right]
            poly = Polygon(coords)
            x,y = poly.exterior.xy
            plt.plot(x,y)
    plt.show()

lat_min, lon_min, lat_max, lon_max = 41.0155, -73.7792749922, 41.03, -73.7582464736

bboxes = get_bounding_boxes.get_bounding_boxes(lat_min,lon_min,lat_max,lon_max,YOLO=True)
# print(len(bboxes)) # We have 2990 buildings. Seems like more than enough

# Size of image (3648, 5280, 3)

im_array = image_to_np_array("./downloads")
print()

pixels = merge_data.OSM_to_pixels([lat_min,lon_min,lat_max,lon_max],im_array.shape[:2], bboxes, YOLO=True)

visualize_bounding_boxes(np.rot90(im_array,1), 
               pixels,
               YOLO=True)

