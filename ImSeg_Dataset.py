from Dataset import Dataset
import os
import numpy as np
import math
from PIL import Image
import json
import random
from shutil import copyfile
import scipy.misc 

# Visualising
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from shapely.geometry.polygon import Polygon
from matplotlib.path import Path

## Size of image that the old tiles will be resized to when building the data set.
IMAGE_SIZE = 224

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
    im = np.array(im)
    im = scipy.misc.imresize(im, (IMAGE_SIZE, IMAGE_SIZE, 3), interp="bilinear")
    scipy.misc.imsave(path_to_dest, im)


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
    Also resizes the mask to IMAGE_SIZE x IMAGE_SIZE before returning the one-hot encoding.
    Assumes: only one class.
    Reference: https://stackoverflow.com/questions/21339448/how-to-get-list-of-points-inside-a-polygon-in-python
    """
    # Image size of tiles.
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
      pixel_annotations = np.array(pixel_annotations)

    pixel_annotations = np.array(pixel_annotations, dtype=np.uint8).reshape((w,h))
    pixel_annotations = scipy.misc.imresize(pixel_annotations, (IMAGE_SIZE, IMAGE_SIZE) )
    pixel_annotations = np.array(pixel_annotations, dtype=bool)

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
    for i in indices:
      image = Image.open(f'{path}/images/{i}.jpg')
      image = np.array(image)

      with open(f'{path}/annotations/{i}.json', 'r') as ann:
        annotation = np.array(json.load(ann)['annotation'])
        
      # Reshape to the width/height dimensions of image, with depth=num_classes (here, it is 1)
      annotation = np.reshape(annotation, (image.shape[0], image.shape[1], 1))

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
    im = Image.open(f'{path}/images/{index}.jpg')
    im_arr = np.array(im)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(im_arr)

    with open(f'{path}/annotations/{index}.json') as f:
      try:
        annotation = json.load(f)
      except:
        annotation = {}

    # w, h, _ = self.get_img_size()  # Right?
    w, h = IMAGE_SIZE, IMAGE_SIZE

    pixel_annotation = np.array(annotation["annotation"]).reshape(w, h)  # Right?

    # Check our results
    ax.imshow(pixel_annotation, alpha=0.3)

    plt.show()
