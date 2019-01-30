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

class DataPipeline:
  """ 
  DataPipeline fetches image and bounding box data from the source APIs in pixel format.
  Minimal processing is done.
  """

  
  def __init__(self, coords, source="IBM", download_path='./downloads'):
    """
    Constructor takes [coords] as an array of 4 latitude and longitude coordinates in 
    the following format [[LAT_MIN, LON_MIN, LAT_MAX, LON_MAX]]
    Constructor also takes the [source] of the data (eg. IBM, Google, etc.)
    Constructor takes the location
    """
    self.coordinates = coords
    self.source = source
    self.download_path = download_path

    if self.source == "IBM":
      user = input("Input your PAIRS Username: ")
      password = input("Input your PAIRS Password: ")
      # Pairs auth for making requests to server
      self.pairs_auth = (user, password)
  

  def query_image(self):
    """
    Sends a request to OSM server and downloads the images in the area specified
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
    z.extractall(self.download_path)
      




  
  def query_OSM(self):
    """
    Sends a request to OSM server and saves an array of all the building nodes
    in the area specified by self.coordinates to self.download_path
    """
    pass

  def image_to_array(self):
    """
    Takes the image(s) downloaded in self.download_path and converts them into 
    np arrays. Returns a list of these arrays
    """
    pass

  def remove_indices(self, indices_to_remove):
    """
    Removes OSM data at the indices specified
    """
    pass

  def coords_to_pixels(self):
    """
    Converts the OSM coordinates to pixels relative to the image data
    """
    pass
  
  def visualize_data(self):
    """
    Provides a visualization of the OSM and image data in self.download_path
    """
    pass
  
  def save_data(self):
    """
    Saves OSM and image data saved in self.download_path to a .pkl file in the 
    same location
    """
    pass
  
  def create_bbox(self):
    """
    Creates formatted bounding box data from OSM data in self.download_path.
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
  


