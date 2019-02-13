## This is the superclass for the dataset generation pipeline

import os
import pandas as pd
import numpy as np
import geojson
import requests
import zipfile, io
from osgeo import gdal
from time import sleep
import overpy
import pickle 
import scipy.misc
import math
from PIL import Image

# Visualising
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from shapely.geometry.polygon import Polygon


class DataPipeline:
  """ 
  DataPipeline fetches image and bounding box data from the source APIs in pixel format.
  Minimal processing is done.
  """

  # im_arr_filename is the name for the queried image data to be stored to (in an np array)
  im_arr_filename="im_arr.pkl"
  # osm_filename is the name for the queried OSM coordinates to be stored to
  osm_filename="OSM_bbox.pkl"
  # name of tiles pkl file containing img tiles and bounding boxes
  tiles_filename="tiles.pkl"
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
    
    # Save the bounding boxes (in lat,lon coordinates) to a pickle file
    with open(f"{DataPipeline.download_path}/{DataPipeline.osm_filename}", "wb") as filename:
      pickle.dump(building_coords, filename)


  def image_to_array(self):
    """
    Takes the image(s) downloaded in DataPipeline.download_path and converts them into 
    np arrays. Returns a list of these arrays. 
    """

    # # Fetches images from download folder
    # images_arr = []
    # # Loop through files in downloads directory (if multiple)
    # file_names = os.listdir(DataPipeline.download_path)
    # file_names.sort(reverse=True)
    # for filename in file_names:
    #     if filename.endswith(".tiff"):
    #         path_to_file = DataPipeline.download_path + '/' + filename
    #         dataset = gdal.Open(path_to_file)
    #         array = np.array(dataset.GetRasterBand(1).ReadAsArray(), np.uint8)
    #         images_arr.append(array)
    # # Return rgb image in np array format
    # im_arr = np.dstack(images_arr)

    im_arr = np.array(Image.open(f"{DataPipeline.download_path}/im_arr.jpg"))

    # Update the [im_size] attribute to the correct image shape.
    self.im_size = im_arr.shape

    

    # scipy.misc.imsave(f"{DataPipeline.download_path}/im_arr.jpg",im_arr)

    # Save the image np array in a pickle file
    with open(f"{DataPipeline.download_path}/{DataPipeline.im_arr_filename}", "wb") as filename:
      pickle.dump(im_arr, filename)


  def coords_to_pixels(self):
    """
    Converts the OSM coordinates to pixels relative to the image data.

    Return format is [[building1_node, ...], [building2_node, ...], ...]
    where each building_node is (pixel_x, pixel_y)
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
        nodeY = math.floor(((lat_max-lat)/height)*self.im_size[0])
        building_coords[b_ind][n_ind] = (nodeX, nodeY) 

    # Save pixel data to pkl file
    with open(f"{DataPipeline.download_path}/{DataPipeline.osm_filename}", "wb") as filename:
      pickle.dump(building_coords, filename)
     
    
  def boxes_in_tile(self, building_coords, col_start, col_end, row_start, row_end):
    """
    Helper function that returns the list of boxes that are in the tile specified by
    col_start..col_end (the x range) and row_start..row_end (the y range). 

    Precondition: building_coords are in pixels not in lat,lon

    Returns [[building1_node, ...], [building2_node, ...], ...] of the buildings inside the 
    given tile range, with coordinates of building_nodes converted so that they are relative to tile.
    """
    
    # Output buildings that are in the tile
    buildings_in_tile = []

    for building in building_coords:

      # All the x and y coordinates of the nodes in a building, separated
      x_coords = [node[0] for node in building]
      y_coords = [node[1] for node in building]

      min_x = min(x_coords)
      max_x = max(x_coords)

      min_y = min(y_coords)
      max_y = max(y_coords)

      centre_x = (min_x + max_x) / 2
      centre_y = (min_y + max_y) / 2

      if col_start <= centre_x < col_end and row_start <= centre_y < row_end:
        
        # Goes through each node in building, converts coords relative to entire image to 
        # coords relative to tile
        new_building = list(map(lambda pos: (pos[0] - col_start, pos[1] - row_start), building))

        buildings_in_tile.append(new_building)
      
    
    return buildings_in_tile

        
  def tile_image(self, tile_size):
    """
    Tiles image array saved in DataPipeline.im_array_filename and saves tiles of 
    size [tile_size x tile_size] and corresponding bounding boxes in DataPipeline.tiles_filename

    pkl file contains a list of tuples in the following format:
    [(tile_im_array, bboxes_in_tile), ...]
    """

    tiles_and_boxes = []

    # Open pickle file with osm data, assumes building coordinates are in pixels not lat, lon
    building_coords = []
    with open(f"{DataPipeline.download_path}/{DataPipeline.osm_filename}", "rb") as filename:
      building_coords = pickle.load(filename)

    # Open pickle file with entire image np array
    im_arr = []
    with open(f"{DataPipeline.download_path}/{DataPipeline.im_arr_filename}", "rb") as filename:
      im_arr = pickle.load(filename)

    height, width, depth = self.im_size
    total_rows = height//tile_size
    total_cols = width//tile_size

    for row in range(total_rows):
      for col in range(total_cols):
        # row_start, row_end, col_start, col_end in pixels relative to entire img
        row_start = row*tile_size
        row_end = (row+1)*tile_size
        col_start = col*tile_size
        col_end = (col+1)*tile_size

        # All the building bounding boxes in the tile range
        buildings_in_tile = self.boxes_in_tile(building_coords, col_start, col_end, row_start,row_end)

        tile = im_arr[row_start:row_end, col_start:col_end, :]

        tiles_and_boxes.append((tile, buildings_in_tile))
    
    with open(f"{DataPipeline.download_path}/{DataPipeline.tiles_filename}", "wb") as filename:
      pickle.dump(tiles_and_boxes, filename)


  def remove_indices(self, indices_to_remove):
    """
    Removes tiles associated with index in indices_to_remove.
    indices_to_remove is a list of indices.

    Saves the edited [(tile, buildings), ...] in the pickle file. 
    """

    tiles_and_boxes = []

    with open(f"{DataPipeline.download_path}/{DataPipeline.tiles_filename}", "rb") as filename:
      tiles_and_boxes = pickle.load(filename)
    
    # Iterates through all tiles, keeps those whose index is not in indices_to_remove
    edited = [tiles_and_boxes[i] for i in range(len(tiles_and_boxes)) if i not in indices_to_remove]

    with open(f"{DataPipeline.download_path}/{DataPipeline.tiles_filename}", "wb") as filename:
      pickle.dump(edited, filename)


  
  def create_bbox(self):
    """
    Creates formatted bounding box data from OSM data in DataPipeline.download_path.
    Function defined specifically for each model/subclass. 
    """
    pass

  def test(self):
    """
    Use this function to test any of the other functions.
    """
    print("Lat height" + str(self.coordinates[2] - self.coordinates[0]))
    print("Lon width" + str(self.coordinates[3] - self.coordinates[1]))
    print(self.im_size)
  

## Tests with coordinates
# Perfect Square: [41.009, -73.779, 41.03, -73.758]
