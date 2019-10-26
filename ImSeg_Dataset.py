import os
import math
import json
import random
import argparse
import numpy as np
from PIL import Image
from shutil import copyfile
from Dataset import Dataset

# Visualising
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.colors as colors
from shapely.geometry.polygon import Polygon
from matplotlib.path import Path


class ImSeg_Dataset(Dataset):
  """
  The ImSeg_Dataset class inherits from the parent 'Dataset' class and provides
  functionality to convert the images and annotations into the format required
  for semantic segmentation models.

  """

  def __init__(self, data_path, classes_path, train_val_test=(0.8, 0.1, 0.1), image_resize=None):
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

    Dataset.__init__(self, data_path, classes_path=classes_path)

    self.image_size = image_resize if image_resize else self.get_img_size()
    self.seg_classes = self.sorted_classes(self.classes)
    self.class_colors = [colors.ListedColormap(np.random.rand(256,3)) for _ in self.seg_classes]

    self.train_val_test = train_val_test
    self.train_path = os.path.join(self.data_path, 'im_seg', 'train')
    self.val_path = os.path.join(self.data_path, 'im_seg', 'val')
    self.test_path = os.path.join(self.data_path, 'im_seg', 'test')
    self.out_path = os.path.join(self.data_path, 'im_seg', 'out')
    self.data_sizes = [] # [train_size, val_size, test_size, out_size]

    if not os.path.isdir(os.path.join(self.data_path, 'im_seg')):
      print(f"Creating directory to store semantic segmentation formatted dataset.")
      os.mkdir(os.path.join(self.data_path, 'im_seg'))

    # Create train, validation, test directories, each with an images and
    # annotations sub-directories
    for directory in [self.train_path, self.val_path, self.test_path, self.out_path]:
      if not os.path.isdir(directory):
        os.mkdir(directory)

      if not os.path.isdir(os.path.join(directory, 'images')):
        os.mkdir(os.path.join(directory, 'images'))

      if not os.path.isdir(os.path.join(directory, 'annotations')):
        os.mkdir(os.path.join(directory, 'annotations'))

      # Size of each training, val and test directories  
      num_samples = len([name for name in os.listdir(os.path.join(directory, 'images')) if name.endswith('.jpg')])
      self.data_sizes.append(num_samples)

  
  def get_seg_class_name(self, super_class_name, sub_class_name, delim=':'):
    """
    Maps a given super class name and sub-class name to the class name stored
    in the ordered list of segmentation classes.
    """
    return super_class_name + delim + sub_class_name


  def sorted_classes(self, classes_dict):
    """
    Helper method to return a sorted list of all the sub_classes.
    The list of classes should be same every time a new ImSeg_Dataset object is created.
    Returns: 
      ['superclass0:subclass1', 'superclass0:subclass1', ...]
    """
    seg_classes = []
    for super_class, sub_classes in classes_dict.items():
      seg_classes.extend([self.get_seg_class_name(super_class, sub_class) for sub_class in sub_classes])
    return sorted(seg_classes)


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

        self.format_image(os.path.join(self.images_path, shuffled_img[i]), \
                          os.path.join(self.train_path, "images", f"{i}.jpg"))

        self.format_json(os.path.join(self.annotations_path, shuffled_annotations[i]), \
                         os.path.join(self.train_path, "annotations", f"{i}.json"), f"{i}.jpg")
        
        self.data_sizes[0] += 1

      elif i < math.floor((train+val)*len(shuffled_img)):
        # Add to val folder
        ind = i - math.floor(train*len(shuffled_img))

        self.format_image(os.path.join(self.images_path, shuffled_img[i]), \
                          os.path.join(self.val_path, "images", f"{ind}.jpg"))

        self.format_json(os.path.join(self.annotations_path, shuffled_annotations[i]), \
                         os.path.join(self.val_path, "annotations", f"{ind}.json"), f"{ind}.jpg")
        
        self.data_sizes[1] += 1
        
      else:
        # Add to test folder
        ind = i - math.floor((train+val)*len(shuffled_img))

        self.format_image(os.path.join(self.images_path, shuffled_img[i]), \
                          os.path.join(self.test_path, "images", f"{ind}.jpg"))

        self.format_json(os.path.join(self.annotations_path, shuffled_annotations[i]), \
                         os.path.join(self.test_path, "annotations", f"{ind}.json"), f"{ind}.jpg")
        
        self.data_sizes[2] += 1
      # increment index counter
      i += 1


  def format_image(self, path_to_file, path_to_dest):
    """
    Helper method called in build_dataset that copies the file from 
    path_to_file, resizes it to IMAGE_SIZE x IMAGE_SIZE x 3, and saves
    it in the destination folder. 
    """
    # copyfile(path_to_file, path_to_dest)
    im = Image.open(path_to_file)
    if (im.size[1], im.size[0], len(im.getbands())) != self.image_size:
      im = im.resize(self.image_size, resample=Image.BILINEAR)
    im.save(path_to_dest)


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
        labels_in_tile = json.load(f)
      except:
        labels_in_tile = {}

    labels_in_tile = self.create_mask(labels_in_tile)

    # Add corresponding image name to annotaion
    labels_in_tile["img"] = img_name

    # save annotation in file
    with open(path_to_dest, 'w') as dest:
      json.dump(labels_in_tile, dest)


  def create_mask(self, labels_in_tile):
    """
    Helper method only called in convert_json that takes a dictionary of
      building coordinates and creates a one-hot encoding for each pixel 
      for an image { "annotation" : [pixel-wise encoding] }
    Also resizes the mask to IMAGE_SIZE x IMAGE_SIZE before returning the one-hot encoding.
    Assumes: only one class.
    Reference: https://stackoverflow.com/questions/21339448/how-to-get-list-of-points-inside-a-polygon-in-python
    """
    # Image size of tiles.
    h, w, _ = self.image_size
    C = len(self.seg_classes)

    # make a canvas with pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()
    # A list of all pixels in terms of indices
    all_pix = np.vstack((x, y)).T

    # Single, (C, w*h) 1d array of empty mask for all pixels in image, for each class, initialised
    # to 0s. This will accumulate annotation markings for each building. 
    pixel_annotations = np.zeros((C, w*h), dtype=bool)
    for super_class, sub_class_labels in labels_in_tile.items():
      for sub_class, labels in sub_class_labels.items():
        for label_nodes in labels:
          p = Path(label_nodes)
          one_building_pixels = p.contains_points(all_pix)

          # Index of label's class name in list of ordered seg_classes
          seg_class = self.seg_classes.index(self.get_seg_class_name(super_class, sub_class))
          pixel_annotations[seg_class] = np.logical_or(pixel_annotations[seg_class], one_building_pixels)
          # pixel_annotations = np.array(pixel_annotations)

    pixel_annotations = pixel_annotations.astype(np.uint8).reshape((C, h, w))

    return {"annotation": pixel_annotations.tolist()}


  def get_batch(self, indices, train_val_test):
    """
    Returns the batch of images and labels associated with the images,
    based on the list of indicies to look up.
    Requires:
      indices: list of indices with which to make a batch
    Format: (block of images, block of labels)
    """

    # Initialise path based on argument
    path = self.train_path
    if train_val_test == "val":
      path = self.val_path
    elif train_val_test == "test":
      path = self.test_path
    elif train_val_test == "out":
      path = self.out_path

    # Accumulators for images and annotations in batch
    images = []
    annotations = []
    C = len(self.seg_classes)
    for i in indices:
      image = Image.open(os.path.join(path, 'images', f'{i}.jpg'))
      image = np.array(image)

      # Reshape to (h,w,C) dimensions
      with open(os.path.join(path, 'annotations', f'{i}.json'), 'r') as ann:
        annotation = np.array(json.load(ann)['annotation']).T

      images.append(image)
      annotations.append(annotation)

    # Return tuple by stacking them into blocks
    return (np.stack(images), np.stack(annotations))


  def save_preds(self, image_indices, preds, image_dir="val"):
    """
    Saves the images specified by image_indices (accessed from image_dir) 
    and the model's predictions (as json files) in the output directory.
    Requires:
      image_indices: A list of indices corresponding to images stored in directory
                     image_dir/images
      preds: A list of np arrays (usually size 224x224) corresponding to the model 
             predictions for each image in image_indices.
      image_dir: The directory that image_indices corresponds to. (Usually validation)
    """
    # Path from where images will be copied
    path_to_im = self.val_path
    if image_dir == "train":
      path_to_im = self.train_path
    elif image_dir == "test":
      path_to_im = self.test_path
    
    # Output directory
    if not os.path.isdir(self.out_path):
      os.mkdir(self.out_path) 
      os.mkdir(self.out_path + '/images')
      os.mkdir(self.out_path + '/annotations')
    
    # First copy the images in image_indices
    for i in image_indices:
      copyfile(
          f"{path_to_im}/images/{i}.jpg", f"{self.out_path}/images/{i}.jpg")

    # Save prediction in json format and dump
    for i in range(len(preds)): 
      preds_json = {"img": str(image_indices[i]) + ".jpg"}
      # take annotation
      preds_json["annotation"] = preds[i].tolist()

      # save annotation in file
      with open(f"{self.out_path}/annotations/{image_indices[i]}.json", 'w') as dest:
        json.dump(preds_json, dest)
    

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
    elif directory == "out":
      path = self.out_path

    # Image visualization
    im = Image.open(os.path.join(path, 'images', f'{index}.jpg'))
    im_arr = np.array(im)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(im_arr)
  
    with open(os.path.join(path, 'annotations', f'{index}.json')) as f:
      try:
        annotation = json.load(f)
      except:
        annotation = {}

    h, w, _ = self.image_size
    C = len(self.seg_classes)

    class_masks = np.array(annotation["annotation"])#.reshape(C, h, w)

    # Check our results
    for i, mask in enumerate(class_masks):
      ax.imshow(mask, alpha=0.15, cmap=self.class_colors[i])

    plt.show()


def passed_arguments():
  parser = argparse.ArgumentParser(description="Script to create Image Segmentation dataset from raw dataset.")
  parser.add_argument('--data_path',\
                      type=str,
                      required=True,
                      help='Path to directory where extracted dataset is stored.')
  parser.add_argument('--classes_path',\
                      type=str,
                      default='./classes.json',
                      help='Path to directory where extracted dataset is stored.')
  parser.add_argument('--tile',\
                      action='store_true',
                      default=False,
                      help='Visualize a random sequence of 20 tiles in the dataset.')
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = passed_arguments()
  
  ds = ImSeg_Dataset(args.data_path, args.classes_path)

  # Create dataset.
  if not os.path.isdir(os.path.join(args.data_path, 'im_seg')):
    ds.build_dataset()

  # Visualize tiles.
  if args.tile:
    inds = random.sample(range(ds.data_sizes[0]), 20)
    for i in inds:
      ds.visualize_tile(i, directory='train')

