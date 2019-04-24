## DataPipeline fetches image and bounding box data from the source APIs in pixel format.
## Minimal processing is done.
import json
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

## Previous Query Coordinates:
# White Plains: [41.009, -73.779, 41.03, -73.758] (returns image of approx 5280x5280)

# DATA_PATH is the path to the directory where the processed and queried data will be saved
DATA_PATH = './data_path_dallas'

# RAW_DATA_PATH is the path to the directory where the raw queried image(s) will be saved
RAW_DATA_PATH = f'{DATA_PATH}/raw_data'

# IMAGES_PATH is the path to the directory where the images will be saved
IMAGES_PATH = f'{DATA_PATH}/images'

# ANNOTATIONS_PATH is the path to the directory where the bounding box labels will be saved
ANNOTATIONS_PATH = f'{DATA_PATH}/annotations'

# OSM_FILENAME is the name of the pickle file where the queried raw OSM data will be saved
OSM_FILENAME = 'OSM_bbox.pkl'

# TILE_SIZE is the size of the tile the entire image will be cut up into
TILE_SIZE = 224
  
def create_dataset(query_path, source="IBM"):
  """
  Initialises the query image coordinates, query source and image download path. 
  This is the main function to be called from this module.

  [coords] is an array of 4 latitude and longitude coordinates in the following format 
  [[LAT_MIN, LON_MIN, LAT_MAX, LON_MAX]]

  [source] is the source API of the data (eg. IBM, Google, etc.)

  If [source=="IBM"], then [(user, password)] is also required.
  """

  # Read the query file, exit if wrong format
  with open(f'{query_path}.json', 'r') as query_file:
    try:
      query = json.load(query_file)
    except:
      print("Your query file is not in proper json format (or is empty).")
      return
  
  # Extract coordinates from query [lat_min, lon_min, lat_max, lon_max]
  try:
    coords = query['spatial']['coordinates']
  except:
    print("Your .json query does not have coordinates specified in the right manner.")

  # If directory for dataset does not exist, create directory
  create_directories()

  # First query image layers from API
  print("Querying raw image from PAIRS using coordinates given.")
  query_PAIRS(query)

  print("")
  
  # Convert the raw image layers into a numpy array (delete the raw image layers)
  print("Converting raw image to numpy array.\nDeleting raw images, saving jpeg instead.")
  im_arr = image_to_array()

  # Then query bounding box data from OSM
  print("Querying raw bounding box data from OpenStreetMap using coordinates given. ")
  raw_OSM = query_OSM(coords)

  # Size of the image
  im_size = im_arr.shape

  # Bounding box data in pixel format
  building_coords = coords_to_pixels(raw_OSM, coords, im_size)

  # Finally, tile the image and save it in the DATA_PATH
  print("Tiling image and saving .jpeg files (for tile) and .json files (for bounding boxes)")
  tile_image(TILE_SIZE, building_coords, im_arr, im_size)

  print("Success! Your raw dataset is now ready!")


def create_directories():
  """
  Creates directory to store dataset specified by DATA_PATH, if one does not exist.
  Nested inside DATA_PATH, creates directories for the raw data, images and annotations.
  """
  if not os.path.isdir(DATA_PATH):
    print(f"Creating directory to store your dataset.")
    os.mkdir(DATA_PATH)

  if not os.path.isdir(RAW_DATA_PATH):
    print(f"Creating directory to store raw data, including queried image.")
    os.mkdir(RAW_DATA_PATH)

  if not os.path.isdir(IMAGES_PATH):
    print(f"Creating directory to store jpeg images.")
    os.mkdir(IMAGES_PATH)

  if not os.path.isdir(ANNOTATIONS_PATH):
    print(f"Creating directory to store json annotations.")
    os.mkdir(ANNOTATIONS_PATH)

  print(f"Your dataset's directory is {DATA_PATH} and the raw data is stored in {RAW_DATA_PATH}")

def query_PAIRS(query_json):
  """
  Sends a request to PAIRS server and downloads the images in the area specified
  by coords. The raw images are saved in RAW_DATA_PATH
  """

  # PAIRS server
  pairs_server = "https://pairs.res.ibm.com"

  # Pairs auth for making requests to server
  user = input("Input your PAIRS Username: ")
  password = input("Input your PAIRS Password: ")
  pairs_auth = (user,password)


  # Make request to IBM server for images from area within coordinates
  response = requests.post(
    json = query_json,
    url=f'{pairs_server}/v2/query',
    auth=pairs_auth,
  )   

  res = response.json()
  id = res['id'] # each request has an id we can access it with
  

  # Check status of query, make sure it's finished before downloading
  response = requests.get(
      url=f'{pairs_server}/v2/queryjobs/{id}',
      auth=pairs_auth,
  )

  # Do not download until the query is succeeded
  while response.json()["status"] != "Succeeded" and response.json()["status"] != "Failed":
    sleep(3)
    # Recheck status after waiting 3 seconds
    response = requests.get(
        url=f'{pairs_server}/v2/queryjobs/{id}',
        auth=pairs_auth,
    )
    print("Query status: " + response.json()["status"])

  # Status updates
  print("json eventual response: " + response.json()["status"])

  #Download and extract to files
  download = requests.get(
      f'{pairs_server}/download/{id}', auth=pairs_auth, stream=True,
  )
  z = zipfile.ZipFile(io.BytesIO(download.content))

  ## Extract to download path specified
  z.extractall(RAW_DATA_PATH)
    

def image_to_array():
  """
  Takes the image(s) downloaded in RAW_DATA_PATH and converts them into an 
  np array. Deletes the raw images in the process.

  Returns: 
  A numpy array of the entire image.
  """

  # Fetches images from download folder
  images_arr = []
  # Loop through files in downloads directory (if multiple)
  file_names = os.listdir(RAW_DATA_PATH)
  file_names.sort(reverse=True)
  for filename in file_names:

    # Remove output.info
    if filename.endswith(".info"):
      path_to_file = RAW_DATA_PATH + '/' + filename
      os.remove(path_to_file)

    if filename.endswith(".tiff"):
      path_to_file = RAW_DATA_PATH + '/' + filename
      dataset = gdal.Open(path_to_file)
      raw_array = np.array(dataset.GetRasterBand(1).ReadAsArray())

      # Remove masked rows and transpose so we can do the same to cols
      row_mask = (raw_array > -128).any(axis=1)
      arr_clean_rows = raw_array[row_mask].T
      col_mask = (arr_clean_rows > -128).any(axis=1)
      clean_array = (arr_clean_rows[col_mask]).T

      # Append clean image array 
      images_arr.append(clean_array)
      
      # Remove the raw .tiff image
      os.remove(path_to_file)
      os.remove(path_to_file + '.json')

  # Return rgb image in np array format
  im_arr = np.dstack(images_arr)

  # Turns np array into jpg and saves into RAW_DATA_PATH
  scipy.misc.imsave(f'{RAW_DATA_PATH}/Entire_Area.jpg', im_arr)

  return im_arr


def query_OSM(coords):
  """
  Sends a request to OSM server and returns an array of all the building nodes
  in the area specified by [coords]

  Returns: 
  [[building1_node, ...], [building2_node, ...], ...] where each building_node
  is in (lat,lon) format.
  """
  api = overpy.Overpass()
  query_result = api.query(("""
      way
          ({}, {}, {}, {}) ["building"];
      (._;>;);
      out body;
      """).format(coords[0], coords[1], coords[2], coords[3]))
  
  # Unprocessed building data from the query
  buildings = query_result.ways

  # The list of each building's coordinates.
  # Each item in this list is a list of points in (lat,lon) for each building's nodes.
  building_coords = []

  for building in buildings:
    points = [(float(str(n.lat)), float(str(n.lon))) for n in building.nodes]
    building_coords.append(points)
  
  return building_coords

  # # Save the bounding boxes (in lat,lon coordinates) to a pickle file
  # with open(f"{RAW_DATA_PATH}/{OSM_FILENAME}", "wb") as filename:
  #   pickle.dump(building_coords, filename)


def coords_to_pixels(raw_OSM, coords, im_size):
  """
  Converts the OSM coordinates to pixels relative to the image data.
  Also stores the returned list of buildings in a pickle file called 'annotations.pkl'

  Requires:
  [coords] is is in [LAT_MIN, LON_MIN, LAT_MAX, LON_MAX] format
  [im_size] is the shape of the shape of the entire image numpy array

  Returns:
  [[building1_node, ...], [building2_node, ...], ...] where each building_node 
  is in (pixel_x, pixel_y) format
  """

  building_coords = raw_OSM

  lat_min, lon_min, lat_max, lon_max = coords
  width = lon_max - lon_min # width in longitude of image
  height = lat_max - lat_min # height in latitude of image

  # Replaces lat,lon building coordinates with x,y coordinates relative to image array
  for b_ind in range(len(building_coords)):
    for n_ind in range(len(building_coords[b_ind])):
      lat, lon = building_coords[b_ind][n_ind]
      nodeX = math.floor(((lon-lon_min)/width)*im_size[1])
      nodeY = math.floor(((lat_max-lat)/height)*im_size[0])
      building_coords[b_ind][n_ind] = (nodeX, nodeY) 
    
  with open(f"{RAW_DATA_PATH}/annotations.pkl", "wb") as filename:
    pickle.dump(building_coords, filename)

  # Reutrn the pixel building coords
  return building_coords
    
  
def boxes_in_tile(building_coords, col_start, col_end, row_start, row_end):
  """
  Helper function that returns the dictionary of boxes that are in the tile specified by
  col_start..col_end (the x range) and row_start..row_end (the y range). 

  Requires: 
  [building_coords] are in pixels not in lat,lon

  Returns:
  {0:[building1_node, ...], 1:[building2_node, ...], ...} of the buildings inside the 
  given tile range, with coordinates of building_nodes converted so that they are relative to tile.
  """
  
  # Output buildings that are in the tile
  buildings_in_tile = {}

  building_index = 0

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

      buildings_in_tile[building_index] = new_building

      building_index += 1
    
  return buildings_in_tile

def save_tile_and_bboxes(tile, building_coords, file_index):
  """
  Saves the tile as an indexed .jpeg image and the building_coords as an indexed .json file.

  Requires: 
  [tile] is a numpy array, 
  [building_coords] is a dictionary of int, list pairs of the building coordinates
    associated with the tile
  [file_index] is an integer.
  """

  img_name = "img_" + str(file_index) + '.jpg'
  bbox_name = "annotation_" + str(file_index) + '.json'

  # save jpeg
  with open(f'{IMAGES_PATH}/{img_name}', 'w') as filename:
    scipy.misc.imsave(filename, tile)

  # save json
  with open(f'{ANNOTATIONS_PATH}/{bbox_name}', 'w') as filename:
    json.dump(building_coords, filename, indent=2)


      
def tile_image(tile_size, building_coords, im_arr, im_size):
  """
  Tiles image array [im_arr] and saves tiles of size [tile_size x tile_size] 
  and corresponding bounding boxes in [DATA_PATH] as individual .jpeg and .json files

  Requires: 
  [tile_size] is a positive integer
  [building_coords] is [[building1_node, ...], [building2_node, ...], ...] where each 
   building_node is (pixel_x, pixel_y)
  [im_arr] is a numpy array of the entire queried image
  [im_size] is the shape of the numpy array
  """

  height, width, depth = im_size
  total_rows = height//tile_size
  total_cols = width//tile_size

  index = 0

  for row in range(total_rows):
    for col in range(total_cols):
      # row_start, row_end, col_start, col_end in pixels relative to entire img
      row_start = row*tile_size
      row_end = (row+1)*tile_size
      col_start = col*tile_size
      col_end = (col+1)*tile_size

      # All the building bounding boxes in the tile range
      buildings_in_tile = boxes_in_tile(building_coords, col_start, col_end, row_start,row_end)

      tile = im_arr[row_start:row_end, col_start:col_end, :]

      save_tile_and_bboxes(tile, buildings_in_tile, index)
      
      index += 1
  
  # with open(f"{DataPipeline.download_path}/{DataPipeline.tiles_filename}", "wb") as filename:
  #   pickle.dump(tiles_and_boxes, filename)


if __name__ == "__main__":
  print("\nPipeline usage: ")
  print(f"1) Make sure your data path is what you want it to be (it is currently: '{DATA_PATH}')")
  print("2) Call the function: create_dataset(query_path)")
  print("   query_path is the path to the json query that will be input when querying PAIRS")
  print("   Eg: query_path can be './PAIRS_Queries/Query_WhitePlains' (file extension not needed)\n")




     
