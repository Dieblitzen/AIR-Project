import os
import pandas as pd
import numpy as np
import geojson
from pytz import UTC
import requests
from requests.auth import HTTPBasicAuth
import zipfile, io
from osgeo import gdal
from time import sleep


## Takes file name and converts to np array
def tiff2array(fn):
    ## Special gdal dataset
    dataset = gdal.Open(fn)
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

print(image_to_np_array("./downloads"))

## TODO: add option to graph on image at certain path (or maybe just give np array?)
def visualize_bounding_boxes(bb_pixels): 
    for bbox_coords in bb_pixels: 
        poly = Polygon(bbox_coords)
        x,y = poly.exterior.xy
        plt.plot(x,y)

    plt.show()



## TODO: Convert Access_Pairs to a function, Filter OSM data for data outside the box.