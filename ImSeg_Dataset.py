from Dataset import Dataset
import os
import numpy as np
import math
from PIL import Image
import json
import random
from shutil import copyfile

# Visualising
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from shapely.geometry.polygon import Polygon
from matplotlib.path import Path


class ImSeg_Dataset(Dataset):
  """
  The ImSeg_Dataset class inherits from the parent 'Dataset' class and provides
  functionality to convert the images and annotations into the format required
  for semantic segmentation models.

  """

  def __init__(self, data_path, train_val_test=(0.8, 0.1, 0.1)):
    """
    Initialises a ImSeg_Dataset object by calling the superclass initialiser.

    The difference between an ImSeg_Dataset object and a Dataset object is the annotation.
    This object will therefore override the self.annotations_path and
    self.annotation_list attributes.
    """
    assert (train_val_test[0] + train_val_test[1] + train_val_test[2]
            ) == 1, 'Train, val and test percentages should add to 1'
    assert train_val_test[0] > 0 and train_val_test[1] > 0 and train_val_test[
        2] > 0, 'Train, val and test percentages should be non-negative'

    Dataset.__init__(self, data_path)

    self.train_val_test = train_val_test
    self.train_path = self.data_path + '/im_seg/train'
    self.val_path = self.data_path + '/im_seg/val'
    self.test_path = self.data_path + '/im_seg/test'
    self.data_sizes = [] # [train_size, val_size, test_size]

    if not os.path.isdir(self.data_path + '/im_seg'):
      print(f"Creating directory to store semantic segmentation formatted dataset.")
      os.mkdir(self.data_path + '/im_seg')

    # Create train, validation, test directories, each with an images and
    # annotations sub-directories
    for directory in [self.train_path, self.val_path, self.test_path]:
      if not os.path.isdir(directory):
        os.mkdir(directory)

      if not os.path.isdir(directory + '/images'):
        os.mkdir(directory + '/images')

      if not os.path.isdir(directory + '/annotations'):
        os.mkdir(directory + '/annotations')

      # Size of each training, val and test directories  
      num_samples = len([name for name in os.listdir(f'{directory}/images') if name.endswith('.jpg')])
      self.data_sizes.append(num_samples)


  def get_batch(self, indices, path):
    """
    Returns the batch of images and labels associated with the images,
    based on the list of indicies to look up.
    Requires:
      indices: list of indices with which to make a batch
    Format: (block of images, labels)
    """
    batch = []
    for i in indices:
      image = Image.open(f'{path}/images/{i}.jpg')
      image = np.array(image)

      with open(f'{path}/annotations/{i}.json', 'r') as ann:
        annotation = np.array(json.load(ann)['annotation'])
        
      batch.append((image, annotation))

    return batch

  def build_dataset(self):
    """
    Helper method only called in build_dataset that splits data into test
    train and validation sets.
    """
    data = list(zip(self.img_list, self.annotation_list))
    random.shuffle(data)
    shuffled_img, shuffled_annotations = zip(*data)

    train, val, test = self.train_val_test

    # index counter i
    i = 0
    while i < len(shuffled_img):
      if i < math.floor(train*len(shuffled_img)):
        # Add to train folder
        copyfile(
          f"{self.images_path}/{shuffled_img[i]}", f"{self.train_path}/images/{i}.jpg")
        self.format_json(
          f"{self.annotations_path}/{shuffled_annotations[i]}", f"{self.train_path}/annotations/{i}.json", f"{i}.jpg")
        
        self.data_sizes[0] += 1

      elif i < math.floor((train+val)*len(shuffled_img)):
        # Add to val folder
        ind = i - math.floor(train*len(shuffled_img))
        copyfile(
          f"{self.images_path}/{shuffled_img[i]}", f"{self.val_path}/images/{ind}.jpg")
        self.format_json(
          f"{self.annotations_path}/{shuffled_annotations[i]}", f"{self.val_path}/annotations/{ind}.json", f"{ind}.jpg")
        
        self.data_sizes[1] += 1
        
      else:
        # Add to test folder
        ind = i - math.floor((train+val)*len(shuffled_img))
        copyfile(
          f"{self.images_path}/{shuffled_img[i]}", f"{self.test_path}/images/{ind}.jpg")
        self.format_json(
          f"{self.annotations_path}/{shuffled_annotations[i]}", f"{self.test_path}/annotations/{ind}.json", f"{ind}.jpg")
        
        self.data_sizes[2] += 1
      # increment index counter
      i += 1

  def format_json(self, path_to_file, path_to_dest, img_name):
    """
    Helper method only called in split_data that takes a json file at
    path_to_file and writes a corresponding json at path_to_dest.
    {
      "annotation" : [pixel-wise encoding],
      "img" : img_name
    }
    """

    # Im_size: [width, height, depth] should be squares
    with open(path_to_file) as f:
      try:
        buildings_dict = json.load(f)
      except:
        buildings_dict = {}

    buildings_dict = self.one_hot_encoding(buildings_dict)

    # Add corresponding image name to annotaion
    buildings_dict["img"] = img_name

    # save annotation in file
    with open(path_to_dest, 'w') as dest:
      json.dump(buildings_dict, dest)

  def one_hot_encoding(self, buildings_dict):
    """
    Helper method only called in convert_json that takes a dictionary of
    building coordinates and creates a one-hot encoding for each pixel 
    for an image { "annotation" : [pixel-wise encoding] }
    Assumes: only one class.
    Reference: https://stackoverflow.com/questions/21339448/how-to-get-list-of-points-inside-a-polygon-in-python
    """

    w, h, _ = self.get_img_size()  # Right order?

    # make a canvas with pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()
    # A list of all pixels in terms of indices
    all_pix = np.vstack((x, y)).T

    # Single, w*h 1d array of all pixels in image initialised to 0s. This will accumulate
    # annotation markings for each building. 
    pixel_annotations = np.zeros((w*h), dtype=bool)
    for building, nodes in buildings_dict.items():
      p = Path(nodes)
      one_building_pixels = p.contains_points(all_pix)

      pixel_annotations = np.logical_or(pixel_annotations, one_building_pixels)

    return {"annotation": pixel_annotations.tolist()}

  def visualize_tile(self, index, directory="train"):
    """
    Provides a visualization of the tile and its corresponding annotation/ label in one
    of the train/test/val directories as specified. 
    """

    path = self.train_path
    if directory == "test":
      path = self.test_path
    elif directory == "val":
      path = self.val_path

    # Image visualization
    im = Image.open(f'{path}/images/{index}.jpg')
    im_arr = np.array(im)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(im_arr)

    with open(f'{path}/annotations/{index}.json') as f:
      try:
        annotation = json.load(f)
      except:
        annotation = {}

    w, h, _ = self.get_img_size()  # Right?

    pixel_annotation = np.array(annotation["annotation"]).reshape(w, h)  # Right?

    # Check our results
    ax.imshow(pixel_annotation, alpha=0.3)

    plt.show()
