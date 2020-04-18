## DataPipeline fetches image and bounding box data from the source APIs in pixel format.
## Minimal processing is done.
import json
import os
import numpy as np
import zipfile, io
from ibmpairs import paw
from time import sleep
import overpy
import pickle 
import math
import argparse
from PIL import Image
from Dataset import Dataset


class DataInfo:
  def __init__(self, data_path, tile_size, overlap, pairs_query_path, classes_path):
    # A Dataset object handling directory names and paths
    # It will also create directories for the data if they don't exist.
    self.ds = Dataset(data_path, classes_path=classes_path)
    print(f"Your dataset's directory is {self.ds.data_path}")
    print(f"The raw data is stored in {self.ds.raw_data_path}")

    # Square tile size that the large image will be broken up into
    self.tile_size = tile_size

    # Number of overlapping pixels between adjacent tiles
    assert overlap < tile_size, "Can't overlap pixels beyond actual tile_size"
    self.overlap = overlap

    # Path to the file where json PAIRS query is stored.
    self.pairs_query_path = pairs_query_path

    # Path to json file containing the OSM classes.
    self.classes_path = classes_path

    # Name of file where raw OSM data will be dumped as a dictionary.
    self.osm_filename = 'OSM_bbox.pkl'

  
def create_dataset(data_info, source="IBM"):
  """
  Initialises the query image coordinates, query source and image download path. 
  This is the main function to be called from this module.

  [coords] is an array of 4 latitude and longitude coordinates in the following format 
  [[LAT_MIN, LON_MIN, LAT_MAX, LON_MAX]]

  [source] is the source API of the data (eg. IBM, Google, etc.)
  If [source=="IBM"], then [(user, password)] is also required.
  """

  # Read the query file, exit if wrong format
  query_path = data_info.pairs_query_path
  query_path = query_path if query_path.endswith('.json') else query_path + '.json'
  with open(f'{query_path}', 'r') as query_file:
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

  # Extract dictionary of OSM label classes from file.
  with open(data_info.classes_path, 'r') as classes_file:
    try:
      classes = json.load(classes_file)
    except:
      print("Your classes .json file is not in the right json format (or is empty)")
      return

  print("Querying raw image from PAIRS using coordinates given:\n")
  images = query_PAIRS(query, data_info.ds.raw_data_path)

  print("\nConverting raw image to numpy array.\nDeleting raw images, saving jpeg instead.")
  im_arr = image_to_array(data_info.ds.raw_data_path, images)

  print("Querying raw bounding box data from OpenStreetMap using coordinates given. ")
  raw_OSM = query_OSM(coords, classes)

  # Bounding box data in pixel format
  im_size = im_arr.shape
  label_coords = coords_to_pixels(raw_OSM, coords, im_size, data_info.ds.raw_data_path)

  print("Tiling image and saving .jpeg files (for tile) and .json files (for bounding boxes)")
  tile_image(label_coords, im_arr, im_size, data_info)

  print("Success! Your raw dataset is now ready!")


def query_PAIRS(query_json, raw_data_path, path_to_credentials='./ibmpairspass.txt'):
  """
  Sends a request to PAIRS server and downloads the images in the area specified
  by coords. The raw images are saved in RAW_DATA_PATH
  """
  with open(path_to_credentials, 'r') as creds:
    creds = creds.read().split(':')
  
  # PAIRS server, and authentication
  pairs_server, user_name, password = creds
  pairs_server = 'https://' + pairs_server
  pairs_auth = (user_name, password)


  # Make request to IBM server for images from area within coordinates
  query = paw.PAIRSQuery(
    query_json,
    pairs_server,
    pairs_auth,
    baseURI='/',
    downloadDir=raw_data_path
  )
  
  # Submit query and wait until downloaded
  query.submit()
  query.poll_till_finished()
  query.download()
  query.create_layers()

  # Sort in reverse to get channels in R, G, B order.
  data_keys = sorted(query.data.keys(), reverse=True)
  images = [query.data[k] for k in data_keys]

  # Delete the zip file.
  zip_file_path = os.path.join(raw_data_path, query.zipFilePath)
  os.remove(zip_file_path)

  return images
    

def image_to_array(raw_data_path, images):
  """
  Takes the list of raw image(s) downloaded from the query in RGB order, and converts 
  them to an np array. Stores the entire area's image in raw_data_path.

  Returns: 
  A numpy array of the entire image.
  """
  # If query doesn't return list of images, then extract images from download folder.
  if images is None or images == []:
    images = []
    file_names = sorted(os.listdir(raw_data_path), reverse=True)

    # Import gdal for .tiff images.
    from osgeo import gdal

    for filename in file_names:
      # Remove output.info
      if filename.endswith(".info"):
        path_to_file = os.path.join(raw_data_path, filename)
        os.remove(path_to_file)

      if filename.endswith(".tiff"):
        path_to_file = os.path.join(raw_data_path, filename)
        dataset = gdal.Open(path_to_file)
        raw_array = np.array(dataset.GetRasterBand(1).ReadAsArray())
        images.append(raw_array)

        # Remove the raw .tiff image
        os.remove(path_to_file)
        os.remove(path_to_file + '.json')
  
  # Return rgb image in np array format
  im_arr = np.dstack(images)
  
  # Clean the image of invalid values. 
  # We use Red channel (0 index) to find -128 values (shouldn't matter which channel)
  im_arr[np.isnan(im_arr)] = -128
  row_mask = (im_arr[..., 0] > -128).any(axis=1)
  clean_arr = im_arr[row_mask, :, :]
  col_mask = (clean_arr[..., 0] > -128).any(axis=0)
  clean_arr = clean_arr[:, col_mask, :]
 
  # Add 128 to make all values positive.
  im_arr = (clean_arr + 128).astype(np.uint8)

  filename = os.path.join(raw_data_path, 'Entire_Area.jpg')
  im = Image.fromarray(im_arr)
  im.save(filename)

  return im_arr


def query_OSM(coords, classes):
  """
  Sends a request to OSM server and returns a dictionary of all the buildings
  and roads nodes along with their sub classes in the area specified by [coords].
  Those buildings and roads not in specified sub-classes are of sub-class "other".

  Returns: 
  {building:
    building_class1: [[building_class1_way1_node1, ...], [way2_node1, ...], ...],
  ..., 
  road:
    road_class_1: [[road_class1_way1_node1, ...], [way2_node1, ...], ...],
  ...} 
  where each node is in (lat,lon) format.
  """
  api = overpy.Overpass()
  coords_string = f"{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}"

  # The dictionary of queried OSM labels for all classes
  query_data = {}
  for super_class, sub_classes in classes.items():
    query_data[super_class] = {}
    for sub_class in sub_classes + ["other"]:
      query_data[super_class][sub_class] = []

  # Query each super class, and then process data.
  for super_class, sub_classes in classes.items():
    sub_classes = set(sub_classes)
    
    query_string =\
      f"""
      way({coords_string})["{super_class}"];
      (._;>;);
      out body;
      """
    super_class_query_result = api.query(query_string)

    # Go through each way and append nodes to corresponding subclass list of points.
    for way in super_class_query_result.ways:
      points = [(float(str(n.lat)), float(str(n.lon))) for n in way.nodes]
      # "amenity" is current building status (eg: building that is hospital now vs was in past)
      sub_class_key = "amenity" if super_class == "building" else super_class

      # subclass is "other" if it doesn't exist in defined set of subclasses
      sub_class_tag = way.tags.get(sub_class_key, "other")
      sub_class = sub_class_tag if sub_class_tag in sub_classes else "other"

      query_data[super_class][sub_class].append(points)
  
  return query_data


def coords_to_pixels(raw_OSM, coords, im_size, raw_data_path):
  """
  Converts the OSM coordinates to pixels relative to the image data.
  Also stores the returned list of buildings in a pickle file called 'annotations.pkl'

  Requires:
  [coords] is is in [LAT_MIN, LON_MIN, LAT_MAX, LON_MAX] format
  [im_size] is the shape of the shape of the entire image numpy array

  Returns: 
  {building:
    building_class1: [[building_class1_way1_node1, ...], [way2_node1, ...], ...], ..., 
  road:
    road_class_1: [[road_class1_way1_node1, ...], [way2_node1, ...], ...], ...} 
  where each node is in (pixel_x, pixel_y) format.
  """

  label_coords = raw_OSM

  lat_min, lon_min, lat_max, lon_max = coords
  width = lon_max - lon_min # width in longitude of image
  height = lat_max - lat_min # height in latitude of image

  # Replaces lat,lon label coordinates with x,y coordinates relative to image array
  for super_class, sub_class_ways in label_coords.items():
    for sub_class, ways in sub_class_ways.items():
      for w_index, way in enumerate(ways):
        for n_index, (lat, lon) in enumerate(way):
          nodeX = math.floor(((lon-lon_min)/width)*im_size[1])
          nodeY = math.floor(((lat_max-lat)/height)*im_size[0])
          label_coords[super_class][sub_class][w_index][n_index] = (nodeX, nodeY)
    
  with open(os.path.join(raw_data_path, "annotations.pkl"), "wb") as filename:
    pickle.dump(label_coords, filename)

  # Reutrn the pixel building coords
  return label_coords


def save_tile_and_bboxes(tile, label_coords, file_index, data_info):
  """
  Saves the tile as an indexed .jpeg image and the label_coords as an indexed .json file.

  Requires: 
  [tile] is a numpy array, 
  [label_coords] is a dictionary of label coordinates (in pixel value) associated with tile.
  [file_index] is an integer.
  """
  img_name = "img_" + str(file_index) + '.jpg'
  bbox_name = "annotation_" + str(file_index) + '.json'

  # save jpeg
  filename = os.path.join(data_info.ds.images_path, img_name)
  im = Image.fromarray(tile)
  im.save(filename)

  # save json
  with open(os.path.join(data_info.ds.annotations_path, bbox_name), 'w') as filename:
    json.dump(label_coords, filename, indent=2)
  
  
def boxes_in_tile(label_coords, col_start, col_end, row_start, row_end):
  """
  Helper function that returns the dictionary of boxes that are in the tile specified by
  col_start..col_end (the x range) and row_start..row_end (the y range). 

  Requires: 
  [label_coords] are in pixels not in lat,lon

  Returns:
  {building: 
    building_class1: [label1_nodes, label2_nodes, ...], ...} 
  of the labels inside given tile range, with coords of label_nodes converted relative to tile.
  """
  
  # Output buildings that are in the tile
  labels_in_tile = {super_class: {} for super_class in label_coords}

  for super_class, sub_class_labels in label_coords.items():
    for sub_class, labels in sub_class_labels.items():
      
      labels_in_tile[super_class][sub_class] = []
      for label in labels:
        # All the x and y coordinates of the nodes in a building, separated
        x_coords, y_coords = [node[0] for node in label], [node[1] for node in label]

        min_x, max_x= min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        centre_x, centre_y = (min_x + max_x) / 2, (min_y + max_y) / 2

        if col_start <= centre_x < col_end and row_start <= centre_y < row_end:
          
          # Goes through each node in building, converts coords relative to entire image to 
          # coords relative to tile
          new_label_nodes = [(node[0] - col_start, node[1] - row_start) for node in label]
          labels_in_tile[super_class][sub_class].append(new_label_nodes)

  return labels_in_tile

      
def tile_image(label_coords, im_arr, im_size, data_info):
  """
  Tiles image array [im_arr] and saves tiles of size [tile_size x tile_size] 
  and corresponding bounding boxes in [DATA_PATH] as individual .jpeg and .json files

  Requires: 
  [tile_size] is a positive integer
  [label_coords] is 
    {building: 
      building_class1: [label_nodes_1, ...], ...
     roads: ...} 
    where each label_nodes_i = [label_node1, ...] and label_node is (pixel_x, pixel_y)
  [im_arr] is a numpy array of the entire queried image
  [im_size] is the shape of the numpy array
  """
  tile_size = data_info.tile_size
  overlap = data_info.overlap

  height, width, _ = im_size
  total_rows = height//(tile_size-overlap)
  total_cols = width//(tile_size-overlap)

  index = 0
  for row_start in range(0, height, tile_size-overlap):
    for col_start in range(0, width, tile_size-overlap):
      # row_start,row_end, col_start, col_end in pixels relative to entire img
      row_end, col_end = row_start+tile_size, col_start+tile_size
      tile = im_arr[row_start:row_end, col_start:col_end, :]

      # All the building bounding boxes in the tile range
      labels_in_tile = boxes_in_tile(label_coords, col_start, col_end, row_start,row_end)
      save_tile_and_bboxes(tile, labels_in_tile, index, data_info)
      
      index += 1


def passed_arguments():
  parser = argparse.ArgumentParser(
    description="Script to extract raw data from PAIRS and Open Street Map.")
  parser.add_argument("-d", "--data_path",
                      type=str, 
                      required=True, 
                      help="Path to directory where extracted data will be stored.")
  parser.add_argument("-t", "--tile_size", 
                      type=int, 
                      default=224,
                      help="Size of square tile (in pixels) into which to break large image.")
  parser.add_argument("-o", "--overlap", 
                      type=int, 
                      default=0,
                      help="Amount of overlapping pixels between adjacent tiles.")
  parser.add_argument("-q", "--query_path", 
                      type=str, 
                      default=os.path.join(".", "PAIRS_Queries", "Query_WhitePlains.json"),
                      help="Path to file containing json query for PAIRS data.")
  parser.add_argument("-c", "--classes", 
                      type=str, 
                      default=os.path.join(".", "classes.json"),
                      help="Path to json file determining OSM classes. Should not be changed.")
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = passed_arguments()
  data_info = DataInfo(
    args.data_path,
    args.tile_size,
    args.overlap,
    args.query_path,
    args.classes
  )

  # For now only IBM.
  create_dataset(data_info, source="IBM")     
