## This is the superclass for the dataset generation pipeline

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
import overpy
import pickle 
import scipy.misc
import math
import matplotlib.pyplot as plt


class DataPipeline:
  """ 
  DataPipeline fetches image and bounding box data from the source APIs in pixel format.
  Minimal processing is done.
  """

  # im_arr_filename is the name for the queried image data to be stored to (in an np array)
  im_arr_filename="im_arr.pkl"
  # osm_filename is the name for the queried OSM coordinates to be stored to
  osm_filename="OSM_bbox.pkl"
  # download_path is the path to the file where the queried image will be stored.
  download_path='./downloads'

  
  def __init__(self, coords, source="IBM"):
    """
    Initialises the query image coordinates, query source and image download path. 

    [coords] is an array of 4 latitude and longitude coordinates in the following format 
    [[LAT_MIN, LON_MIN, LAT_MAX, LON_MAX]]

    [source] is the source API of the data (eg. IBM, Google, etc.)


    If [source=="IBM"], then [(user, password)] is also required.
    """
    self.coordinates = coords
    self.source = source
    self.im_size = None

    if self.source == "IBM":
      user = input("Input your PAIRS Username: ")
      password = input("Input your PAIRS Password: ")
      # Pairs auth for making requests to server
      self.pairs_auth = (user, password)
  

  def query_image(self):
    """
    Sends a request to PAIRS server and downloads the images in the area specified
    by self.coordinates
    """

    # PAIRS server
    pairs_server = "https://pairs.res.ibm.com"

    # Make request to IBM server for images from area within coordinates
    response = requests.post(
      json = {
        "layers" : [
          {
              "type": "raster",
              "id": 36431
          },
          {
              "type": "raster",
              "id": 35445
          },
          {
              "type": "raster",
              "id": 36432
          },
      ],
      "spatial": {
          "type": "square",
          "coordinates": [str(self.coordinates[0]), str(self.coordinates[1]), str(self.coordinates[2]), str(self.coordinates[3])]
      },
      "temporal": {
          "intervals": [{"start": "2014-12-31T00:00:00Z","end": "2015-01-01T00:00:00Z"}]
      }
    },
        url=f'{pairs_server}/v2/query',
        auth=self.pairs_auth,
    )   

    res = response.json()
    id = res['id'] # each request has an id we can access it with
    

    # Check status of query, make sure it's finished before downloading
    response = requests.get(
        url=f'{pairs_server}/v2/queryjobs/{id}',
        auth=self.pairs_auth,
    )

    # Do not download until the query is succeeded
    while response.json()["status"] != "Succeeded" and response.json()["status"] != "Failed":
      sleep(3)
      # Recheck status after waiting 3 seconds
      response = requests.get(
          url=f'{pairs_server}/v2/queryjobs/{id}',
          auth=self.pairs_auth,
      )
      print("Query status... ")
      print(response.json()["status"])

    # Status updates
    print("json eventual response: ")
    print(response.json()["status"])

    #Download and extract to files
    download = requests.get(
        f'{pairs_server}/download/{id}', auth=self.pairs_auth, stream=True,
    )
    z = zipfile.ZipFile(io.BytesIO(download.content))

    ## Extract to download path specified
    z.extractall(DataPipeline.download_path)
      

  def query_OSM(self):
    """
    Sends a request to OSM server and saves an array of all the building nodes
    in the area specified by self.coordinates to DataPipeline.download_path
    """
    api = overpy.Overpass()
    query_result = api.query(("""
        way
            ({}, {}, {}, {}) ["building"];
        (._;>;);
        out body;
        """).format(self.coordinates[0], self.coordinates[1], self.coordinates[2], self.coordinates[3]))
    
    # Unprocessed building data from the query
    buildings = query_result.ways

    # The list of each building's coordinates.
    # Each item in this list is a list of points in (lat,lon) for each building's nodes.
    building_coords = []

    for building in buildings:
      points = [(float(str(n.lat)), float(str(n.lon))) for n in building.nodes]
      building_coords.append(points)
    
    with open(f"{DataPipeline.download_path}/{DataPipeline.osm_filename}", "wb") as filename:
      pickle.dump(building_coords, filename)




  def image_to_array(self):
    """
    Takes the image(s) downloaded in DataPipeline.download_path and converts them into 
    np arrays. Returns a list of these arrays. 
    """

    # Fetches images from download folder
    images_arr = []
    # Loop through files in downloads directory (if multiple)
    file_names = os.listdir(DataPipeline.download_path)
    file_names.sort(reverse=True)
    for filename in file_names:
        if filename.endswith(".tiff"):
            path_to_file = DataPipeline.download_path + '/' + filename
            dataset = gdal.Open(path_to_file)
            array = np.array(dataset.GetRasterBand(1).ReadAsArray(), np.uint8)
            images_arr.append(array)
    # Return rgb image in np array format
    im_arr = np.dstack(images_arr)

    self.im_size = im_arr.shape

    with open(f"{DataPipeline.download_path}/{DataPipeline.im_arr_filename}", "wb") as filename:
      pickle.dump(im_arr, filename)

  def remove_indices(self, indices_to_remove):
    """
    Removes OSM data at the indices specified
    """
    pass

  def coords_to_pixels(self):
    """
    Converts the OSM coordinates to pixels relative to the image data
    """

    assert self.im_size != None, "Image array has not yet been saved. Try image_to_array first"

    building_coords = []

    # Open pickle file with osm data
    with open(f"{DataPipeline.download_path}/{DataPipeline.osm_filename}", "rb") as filename:
      building_coords = pickle.load(filename)

    lat_min, lon_min, lat_max, lon_max = self.coordinates
    width = lon_max - lon_min # width in longitude of image
    height = lat_max - lat_min # height in latitude of image

    # Replaces lat,lon building coordinates with x,y coordinates relative to image array
    for b_ind in range(len(building_coords)):
      for n_ind in range(len(building_coords[b_ind])):
        lat, lon = building_coords[b_ind][n_ind]
        nodeX = math.floor(((lon-lon_min)/width)*self.im_size[1])
        nodeY = math.floor(((lat_max - lat)/height)*self.im_size[0])
        building_coords[b_ind][n_ind] = (nodeX, nodeY) 

    print(building_coords)

    # Save pixel data to pkl file
    with open(f"{DataPipeline.download_path}/{DataPipeline.osm_filename}", "wb") as filename:
      pickle.dump(building_coords, filename)
     
    

  
  def visualize_data(self):
    """
    Provides a visualization of the OSM and image data in DataPipeline.download_path
    """

  
  def create_bbox(self):
    """
    Creates formatted bounding box data from OSM data in DataPipeline.download_path.
    Function defined specifically for each model/subclass. 
    """
    pass
  
  def boxes_in_tile(self):
    """
    """
    pass
  
  def tile_image(self):
    """
    """
    pass
  


