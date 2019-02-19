import os
import numpy as np
import scipy.misc
import math
from PIL import Image
import json
import pickle

# Visualising
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from shapely.geometry.polygon import Polygon


class Dataset:
  """
  The Dataset class provides an interface for working with a dataset consisting 
  of images and annotations. 
  """

  def __init__(self, data_path):
    """
    Initializes a dataset object using the data in path_to_data. 
    """
    self.data_path = data_path
    self.images_path = f'{data_path}/images'
    self.annotations_path = f'{data_path}/annotations'
    self.img_list = sorted(os.listdir(self.images_path))
    self.annotation_list = sorted(os.listdir(self.annotations_path))
    self.length = len(self.img_list)


  def get_img_size(self):
    """
    Gets the size of the images in the dataset
    """
    # Gets first image in dataset
    im = Image.open(f'{self.images_path}/{self.img_list[0]}')
    # Returns the shape of the image
    return np.array(im).shape
  
  def get_tile_and_label(self, index):
    """
    Gets the tile and label associated with data index.

    Returns:
    (tile_array, dictionary_of_buildings)
    """

    # Open the jpeg image and save as numpy array
    im = Image.open(f'{self.images_path}/{self.img_list[index]}')
    im_arr = np.array(im)

    # Open the json file and parse into dictionary of index -> buildings pairs
    buildings_in_tile = {}
    with open(f'{self.annotations_path}/{self.annotation_list[index]}', 'r') as filename:
      try: 
        buildings_in_tile = json.load(filename)
      except ValueError:
        buildings_in_tile = {}
    
    return (im_arr, buildings_in_tile)
    
  
  def get_batch(self, start_index, batch_size):
    """
    Gets batch of tiles and labels associated with data start_index.

    Returns:
    [(tile_array, list_of_buildings), ...]
    """
    batch = []
    for i in range(start_index, start_index + batch_size):
      batch.append(self.get_tile_and_label(i))
    
    return batch

  def remove_tiles(self, indices_to_remove):
    """
    Removes the tiles associated with the indices in indices_to_remove, and renames all files
    in self.images_path and self.annotations.path

    Requires: indices_to_remove is a set
    """
    file_index = 0
    for i in range(self.length):

      # Check if index is to be removed
      if i in indices_to_remove:
        os.remove(f'{self.images_path}/img_{i}.jpg')
        os.remove(f'{self.annotations_path}/annotation_{i}.json')
      else:

        # If not to be removed, then check if index of file is in line with new file_index
        if i != file_index:
          os.rename(f'{self.images_path}/img_{i}.jpg', f'{self.images_path}/img_{file_index}.jpg')
          os.rename(f'{self.annotations_path}/annotation_{i}.json', f'{self.annotations_path}/annotation_{file_index}.json')
        
        file_index += 1

    self.img_list = sorted(os.listdir(self.images_path))
    self.annotation_list = sorted(os.listdir(self.annotations_path))
    self.length = len(self.img_list)

  def visualize_tile(self, index):
    """
    Provides a visualization of the tile with filename tile and its 
    corresponding annotation. 
    """
    im = Image.open(f'{self.images_path}/{self.img_list[index]}')
    im_arr = np.array(im)

    plt.imshow(im_arr)

    # Open the json file and parse into dictionary of index -> buildings pairs
    buildings_in_tile = {}
    with open(f'{self.annotations_path}/{self.annotation_list[index]}', 'r') as filename:
      try: 
        buildings_in_tile = json.load(filename)
      except ValueError:
        buildings_in_tile = {}

    for building_coords in buildings_in_tile.values():
      poly = Polygon(building_coords)
      x, y = poly.exterior.xy
      plt.plot(x, y)

    # TODO: Visualize bounding boxes from json format.

    plt.show()

  def visualize_dataset(self):
    """
    Provides visualization of entire dataset image area, 
    including annotations.

    This uses the data stored in the RAW_DATA_PATH.
    """
    bboxes = []
    # Open pickle file with osm data
    with open(f"{self.data_path}/raw_data/buildings.pkl", "rb") as filename:
      bboxes = pickle.load(filename)

    im = Image.open(f"{self.data_path}/raw_data/Entire_Area.jpg")
    im_arr = np.array(im)

    plt.imshow(im_arr)
    for building_coords in bboxes:
      poly = Polygon(building_coords)
      x, y = poly.exterior.xy
      plt.plot(x, y)
    
    # # Add the grid
    # ax.grid(which='major', axis='both', linestyle='-')
    # ax.show()
    plt.grid()
    # plt.xticks(np.arange(0, 6000, 228), range(0, 23))
    # plt.yticks(np.arange(0, 6000, 228), range(0, 23))
    plt.show()
