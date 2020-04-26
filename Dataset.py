import os
import re
import json
import pickle
import random
import argparse
import numpy as np
from PIL import Image
from shutil import copyfile

# Visualising
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import collections as mc
from shapely.geometry.polygon import Polygon


class Dataset:
  """
  The 'Dataset' class provides an interface for working with a dataset consisting 
  of images and annotations. 

  An object of this class will provide the following functionality:\n

  Attributes:
  1) The path to the dataset, raw image area, tiled images and annotations.\n
  2) The dictionary of classes defined for the dataset.\n
  3) A sorted list of image file names\n
  4) A sorted list of annotation/ building label file names\n

  Static methods (invariant of object):\n
  1) Copy over data from already created datasets into a combined dataset\n

  Instance Methods:\n
  1) Getting the length of the dataset (the number of images/ image file names)\n
  2) Getting the size of each image in the dataset (assumed to be the same for all images).\n
  3) Getting an image and its associated building labels given an index.\n
  4) Getting a batch of images and assoicated building labels given a start index and batch size.\n
  5) Removing a set of images and assoicated building labels given a set of indices.\n
  6) Visualizing a single image in images_path with its assoicated building labels.\n
  7) Visualizing a sequence of tiles (images) in images_path with associated building labels, given
     a start and end index.\n
  8) Visualizing the entire area with all bounding boxes (assuming such an image exists in the
      raw_data directory of the data_path).\n
  """

  def __init__(self, data_path, classes_path='classes.json'):
    """
    Initializes a dataset object using the data in data_path. 
    """

    # The path to the entire data, the images and the annotations. Attributes 1)
    self.data_path = data_path
    self.raw_data_path = os.path.join(data_path, 'raw_data')
    self.images_path = os.path.join(data_path, 'images')
    self.annotations_path = os.path.join(data_path, 'annotations')
    Dataset._create_dirs(self.data_path, self.raw_data_path, 
                         self.images_path, self.annotations_path)

    # Attribute 2)
    with open(classes_path, 'r') as f:
      self.classes = json.load(f)
      
    # Attribute 3)
    self.img_list = Dataset.file_names(self.images_path, '.jpg','.jpeg', key=self.sort_key)
    
    # Attritbute 4)
    self.annotation_list = Dataset.file_names(self.annotations_path, '.json', key=self.sort_key)
  
  @staticmethod
  def _create_dirs(*dirs):
    """
    Creates directories based on paths passed in as arguments.
    """
    def f_mkdir(p):
      if not os.path.isdir(p):
        print(f"Creating directory {p}")
        os.makedirs(p)

    for p in dirs: f_mkdir(p)
  
  @staticmethod
  def file_names(path, *file_exts, key=None):
    """
    Returns a list of all files in `path` ending in `file_ext`,
    optionally sorted by `key`.
    Requires:
      dir: path/to/dir \n
      *file_ext: String file extension (eg: '.jpg') to match.\n
      key: Function to sort by file name\n
    """
    files = []
    for f in os.listdir(path):
      if file_exts:
        for ext in file_exts:
          if f.endswith(ext):
            files.append(f)
            break
      else:
        files.append(f)

    return sorted(files, key=key) if key else files


  def sort_key(self, file_name):
    """
    Helper method only.
    Finds the integer present in the string file_name. If an integer cannot be found,
    returns the file_name itself. Used as key function in sorting list of file names.
    """
    d = re.search('[0-9]+', file_name)
    return int(file_name[d.start():d.end()]) if d else file_name
  
  def __len__(self):
    """
    Method 1)
    Updates the img_list and annotation_list attributes and returns the number
    of images in the dataset.
    """
    self.img_list = Dataset.file_names(self.images_path, '.jpg', '.jpeg', key=self.sort_key)
    self.annotation_list = Dataset.file_names(self.annotations_path, '.json', key=self.sort_key)
    return len(self.img_list)

  def get_img_size(self):
    """
    Method 2)
    Gets the size of the images in the dataset (assumed to be uniform)
    """
    # Gets first image in dataset
    im = Image.open(os.path.join(self.images_path, self.img_list[0]))
    # Returns the shape of the image
    return np.array(im).shape
  
  def get_tile_and_label(self, index):
    """
    Method 3)
    Gets the tile and label associated with data index.

    Returns:
    (tile_array, dictionary_of_buildings)
    """

    # Open the jpeg image and save as numpy array
    im = Image.open(os.path.join(self.images_path, self.img_list[index]))
    im_arr = np.array(im)

    # Open the json file and parse into dictionary of index -> buildings pairs
    buildings_in_tile = {}
    ann_path = os.path.join(self.annotations_path, self.annotation_list[index])
    with open(ann_path, 'r') as filename:
      try: 
        buildings_in_tile = json.load(filename)
      except ValueError:
        buildings_in_tile = {}
    
    return (im_arr, buildings_in_tile)
    
  
  def get_batch(self, start_index, batch_size):
    """
    Method 4)
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
    Method 5)
    Removes the tiles associated with the indices in indices_to_remove, and renames all files
    in self.images_path and self.annotations.path (as appropriate)

    Requires: indices_to_remove is a set
    """

    # file_index keeps track of the correct index for the images in the directory 
    file_index = 0
    for i in range(len(self)):
      img_path = os.path.join(self.images_path, f'{i}.jpg')
      ann_path = os.path.join(self.annotations_path, f'{i}.json')

      # Check if index is to be removed
      if i in indices_to_remove:
        os.remove(img_path)
        os.remove(ann_path)
      else:

        # If not to be removed, then check if index of file is in line with new file_index
        if i != file_index:
          new_img_path = os.path.join(self.images_path, f'{file_index}.jpg')
          os.rename(img_path, new_img_path)

          new_ann_path = os.path.join(self.annotations_path, f'{file_index}.json')
          os.rename(ann_path, new_ann_path)
        
        file_index += 1

    # Update attributes 4)
    self.img_list = Dataset.file_names(self.images_path, '.jpg','.jpeg', key=self.sort_key)
    self.annotation_list = Dataset.file_names(self.annotations_path, '.json', key=self.sort_key)

  def visualize_tile(self, index):
    """
    Method 6)
    Provides a visualization of the tile with the tile and its corresponding annotation/ label. 
    """
    im = Image.open(os.path.join(self.images_path, self.img_list[index]))
    im_arr = np.array(im)
    mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    plt.imshow(im_arr)

    # Open the json file and parse into dictionary of index -> buildings pairs
    labels_in_tile = {}
    ann_path = os.path.join(self.annotations_path, self.annotation_list[index])
    with open(ann_path, 'r') as filename:
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
    Method 7)
    Provides a visualization of a sequence of tiles with associated annotations/labels
    between the start index and the end index (not including end index) of the tiles.
    """
    for i in range(start_idx, end_idx):
      print("Tile index: " + str(i))
      self.visualize_tile(i)
    

  def visualize_dataset(self):
    """
    Method 8)
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
    osm_data_path = os.path.join(self.data_path, 'raw_data', 'annotations.pkl')
    with open(osm_data_path, "rb") as filename:
      label_coords = pickle.load(filename)

    im = Image.open(os.path.join(self.data_path, 'raw_data', 'Entire_Area.jpg'))
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
  

  @staticmethod
  def _combine_datasets(new_data_path, classes_path='classes.json', *data_paths):
    """
    Create a combined dataset from already created Datasets. \n
    Copies over the `images` and `annotations` directories from given datasets.
    Requires:
      new_data_path: Path to directory where combined data will be stored.
    """
    print("Creating directories store images, annotations...")
    new_ds = Dataset(new_data_path, classes_path=classes_path)

    i = 0
    for data_path in data_paths:
      assert os.path.isdir(data_path), f"Can't use non-existent data path: {data_path}"
      ds = Dataset(data_path, classes_path=classes_path)
      assert len(ds) > 0, "Previous dataset must have data."

      # Iterate over each image, annotation, copying to new dataset
      for img_path, ann_path in zip(ds.img_list, ds.annotation_list):
        copyfile(os.path.join(ds.images_path, img_path), 
                 os.path.join(new_ds.images_path, f'{i}.jpg'))

        copyfile(os.path.join(ds.annotations_path, ann_path),
                 os.path.join(new_ds.annotations_path, f'{i}.json'))
        i += 1


def passed_arguments():
  parser = argparse.ArgumentParser(description="Script to visualize labels on entire queried area.")
  parser.add_argument('-d', '--data_path',
                      type=str,
                      default=True,
                      help='Path to directory where extracted dataset is stored.')
  parser.add_argument('-c', '--classes_path',
                      type=str,
                      default='classes.json',
                      help='Path to .json file denoting classes of labels used in dataset.')
  parser.add_argument('-t', '--tile',
                      action='store_true',
                      default=False,
                      help='Visualize a random sequence of 20 tiles in the dataset.')
  parser.add_argument('--combine',
                      nargs='+',
                      type=str,
                      default=None,
                      help='Sequence of data_paths to combine into one new dataset.')
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = passed_arguments()

  if args.combine:
    Dataset._combine_datasets(args.data_path, args.classes_path, *args.combine)

  ds = Dataset(args.data_path, args.classes_path)
  if args.tile:
    inds = random.sample(range(len(ds)), 20)
    for i in inds:
      ds.visualize_tile(i)
  else:
    if not args.combine:
      ds.visualize_dataset()
