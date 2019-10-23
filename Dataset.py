import os
import numpy as np
import scipy.misc
import math
from PIL import Image
import json
import pickle
import re
import argparse
import random

# Visualising
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import collections as mc
from shapely.geometry.polygon import Polygon


class Dataset:
  """
  The 'Dataset' class provides an interface for working with a dataset consisting 
  of images and annotations. 

  An object of this class will provide the following functionality:

  Attributes:
  1) The path to the dataset, images and annotations.
  2) A sorted list of image file names
  3) A sorted list of annotation/ building label file names
  4) The length of the dataset.

  Methods:
  1) Getting the size of each image in the dataset (assumed to be the same for all images)
  2) Getting an image and its associated building labels given an index
  3) Getting a batch of images and assoicated building labels given a start index and batch size
  4) Removing a set of images and assoicated building labels given a set of indices.
  5) Visualizing a single image in images_path with its assoicated building labels
  6) Visualizing a sequence of tiles (images) in images_path with associated building labels, given
     a start and end index.
  7) Visualizing the entire area with all bounding boxes (assuming such an image exists in the
      raw_data directory of the data_path)

  """

  def __init__(self, data_path):
    """
    Initializes a dataset object using the data in data_path. 
    """

    # The path to the entire data, the images and the annotations. Attributes 1)
    self.data_path = data_path
    self.images_path = f'{data_path}/images'
    self.annotations_path = f'{data_path}/annotations'
      
    # Attribute 2)
    self.img_list = sorted(os.listdir(self.images_path), key = self.sort_key)

    # Attritbute 3)
    self.annotation_list = sorted(os.listdir(self.annotations_path), key = self.sort_key)

    # Attribute 4)
    self.length = len(self.img_list)

  def sort_key(self, file_name):
    """
    Helper method only.
    Finds the integer present in the string file_name. If an integer cannot be found,
    returns the file_name itself. Used as key function in sorting list of file names.
    """
    d = re.search('[0-9]+', file_name)
    return int(file_name[d.start():d.end()]) if d else file_name

  def get_img_size(self):
    """
    Method 1)
    Gets the size of the images in the dataset (assumed to be uniform)
    """
    # Gets first image in dataset
    im = Image.open(f'{self.images_path}/{self.img_list[0]}')
    # Returns the shape of the image
    return np.array(im).shape
  
  def get_tile_and_label(self, index):
    """
    Method 2)
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
    Method 3)
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
    Method 4)
    Removes the tiles associated with the indices in indices_to_remove, and renames all files
    in self.images_path and self.annotations.path (as appropriate)

    Requires: indices_to_remove is a set
    """

    # file_index keeps track of the correct index for the images in the directory 
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
          os.rename(f'{self.annotations_path}/annotation_{i}.json', \
                    f'{self.annotations_path}/annotation_{file_index}.json')
        
        file_index += 1

    # Update attributes 1)
    self.img_list = sorted(os.listdir(self.images_path), key = self.sort_key)
    self.annotation_list = sorted(os.listdir(self.annotations_path), key = self.sort_key)
    self.length = len(self.img_list)

  def visualize_tile(self, index):
    """
    Method 5)
    Provides a visualization of the tile with the tile and its corresponding annotation/ label. 
    """
    im = Image.open(f'{self.images_path}/{self.img_list[index]}')
    im_arr = np.array(im)
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.imshow(im_arr)

    # Open the json file and parse into dictionary of index -> buildings pairs
    labels_in_tile = {}
    with open(f'{self.annotations_path}/{self.annotation_list[index]}', 'r') as filename:
      try: 
        labels_in_tile = json.load(filename)
      except ValueError:
        labels_in_tile = {}

    for super_class, sub_class_labels in labels_in_tile.items():
      for sub_class, labels in sub_class_labels.items():
        sub_class_colour = list(np.random.choice(range(256), size=3)/256)
        if super_class == 'building':
          for label in labels:
            poly = Polygon(label)
            x, y = poly.exterior.xy
            plt.plot(x, y, c=sub_class_colour)
        elif super_class == 'highway':
          lines = mc.LineCollection(labels, colors=sub_class_colour)
          plt.gca().add_collection(lines)

    # TODO: Visualize bounding boxes from json format.

    plt.show()
    
    
  def visualize_tiles(self, start_idx, end_idx):
    """
    Method 6)
    Provides a visualization of a sequence of tiles with associated annotations/labels
    between the start index and the end index (not including end index) of the tiles.
    """
    for i in range(start_idx, end_idx):
      print("Tile index: " + str(i))
      self.visualize_tile(i)
    

  def visualize_dataset(self):
    """
    Method 7)
    Provides visualization of entire dataset image area, 
    including annotations.

    This uses the data stored in the RAW_DATA_PATH.
    Requires:
    The entire image area with OSM data to be stored in a directory called raw_data.
    The OSM data should be in a pickle file, and the entire image area should be in 
    a jpeg file.
    """
    label_coords = {}
    # Open pickle file with osm data
    with open(f"{self.data_path}/raw_data/annotations.pkl", "rb") as filename:
      label_coords = pickle.load(filename)

    im = Image.open(f"{self.data_path}/raw_data/Entire_Area.jpg")
    im_arr = np.array(im)

    plt.imshow(im_arr)
    for super_class, sub_class_labels in label_coords.items():
      for sub_class, labels in sub_class_labels.items():
        sub_class_colour = list(np.random.choice(range(256), size=3)/256)
        if super_class == 'building':
          for label in labels:
            poly = Polygon(label)
            x, y = poly.exterior.xy
            plt.plot(x, y, c=sub_class_colour)
        elif super_class == 'highway':
          lines = mc.LineCollection(labels, colors=sub_class_colour)
          plt.gca().add_collection(lines)
    
    # # Add the grid
    # ax.grid(which='major', axis='both', linestyle='-')
    # ax.show()
    plt.grid()
    # plt.xticks(np.arange(0, 6000, 228), range(0, 23))
    # plt.yticks(np.arange(0, 6000, 228), range(0, 23))
    plt.show()


def passed_arguments():
  parser = argparse.ArgumentParser(description="Script to visualize labels on entire queried area.")
  parser.add_argument('--data_path',\
                      type=str,
                      required=True,
                      help='Path to directory where extracted dataset is stored.')
  parser.add_argument('--tile',\
                      action='store_true',
                      default=False,
                      help='Visualize a random sequence of 20 tiles in the dataset.')
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = passed_arguments()
  ds = Dataset(args.data_path)
  
  if args.tile:
    inds = random.sample(range(ds.length), 20)
    for i in inds:
      ds.visualize_tile(i)
  else:
    ds.visualize_dataset()
