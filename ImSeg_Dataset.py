import os
import math
import json
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw
from shutil import copyfile
from Dataset import Dataset
from ImSeg.preprocess import augment_data

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

  def __init__(self, data_path, classes_path=os.path.join('.', 'classes.json'), 
              train_val_test=(0.8, 0.1, 0.1), image_resize=None, augment_kwargs={}):
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
    self.class_colors = []
    # power set of colors across RBG for visualizing
    for i in range(2**3):
      c = [((i >> s) % 2) * 255 for s in range(2, -1, -1)]
      self.class_colors.append(tuple(c))

    # Set up data file paths
    self.train_val_test = train_val_test
    self.train_path = os.path.join(self.data_path, 'im_seg', 'train')
    self.val_path = os.path.join(self.data_path, 'im_seg', 'val')
    self.test_path = os.path.join(self.data_path, 'im_seg', 'test')
    self.out_path = os.path.join(self.data_path, 'im_seg', 'out')

    self.data_sizes = [0] * 4
    self.init_directories()

    # Set up data augmentor if need be
    self.augment = self.get_data_gen(**augment_kwargs) if augment_kwargs else None
  

  def init_directories(self):
    """
    Creates the 'im_seg' directory in the data_path. Also creates the train/val/test/out
    directories with the images/ and annotations/ directory for each.
    If the directories already exist, then initialises the data_sizes based on existing 
    directories.
    """
    if not os.path.isdir(os.path.join(self.data_path, 'im_seg')):
      print(f"Creating directory to store semantic segmentation formatted dataset.")
      os.mkdir(os.path.join(self.data_path, 'im_seg'))

    # Create train, validation, test directories, each with an images and
    # annotations sub-directories
    for i, directory in enumerate([self.train_path, self.val_path, self.test_path]):
      if not os.path.isdir(directory):
        os.mkdir(directory)

      if not os.path.isdir(os.path.join(directory, 'images')):
        os.mkdir(os.path.join(directory, 'images'))

      if not os.path.isdir(os.path.join(directory, 'annotations')):
        os.mkdir(os.path.join(directory, 'annotations'))

      # Size of each training, val and test directories  
      num_samples = len([name for name in os.listdir(os.path.join(directory, 'images'))\
                         if name.endswith('.jpg')])
      self.data_sizes[i] = num_samples
  

  def create_model_out_dir(self, model_name):
    """
    Creates directories for metrics, ouput images and annotations for a
    given model during training.
    """
    try:
      getattr(self, "model_path")
      raise AttributeError("Attribute model_path already created")
    except AttributeError as e:
      pass

    if not os.path.isdir(self.out_path):
      os.mkdir(self.out_path)
    
    self.model_path = os.path.join(self.out_path, model_name)
    if not os.path.isdir(self.model_path):
      os.mkdir(self.model_path)

    self.checkpoint_path = os.path.join(self.model_path, 'checkpoints')
    if not os.path.isdir(self.checkpoint_path):
      os.mkdir(self.checkpoint_path)
    
    self.metrics_path = os.path.join(self.model_path, 'metrics')
    if not os.path.isdir(self.metrics_path):
      os.mkdir(self.metrics_path)
    
    if not os.path.isdir(os.path.join(self.model_path, 'preds')):
      os.mkdir(os.path.join(self.model_path, 'preds'))

  
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
    # to 0s. This will accumulate annotation markings for each building by ORing the pixels.
    pixel_annotations = np.zeros((C, h*w), dtype=bool)
    for super_class, sub_class_labels in labels_in_tile.items():
      for sub_class, labels in sub_class_labels.items():
        for label_nodes in labels:
          
          one_label_pixels = []
          if super_class == "highway":
            # Draw the road as a thick line on a blank mask.
            label_nodes = [tuple(point) for point in label_nodes]
            road_mask = Image.fromarray(np.zeros((h, w)).astype(np.uint8))
            drawer = ImageDraw.Draw(road_mask)
            drawer.line(label_nodes, fill=1, width=3)

            one_label_pixels = np.array(road_mask).astype(np.bool).flatten()

          elif super_class == "building":
            # Create a path enclosing the nodes, and then fill in a blank mask for building's mask
            p = Path(label_nodes)
            one_label_pixels = p.contains_points(all_pix)
          else:
            raise NotImplementedError("Only support roads and buildings currently.")

          # Index of label's class name in list of ordered seg_classes
          seg_class = self.seg_classes.index(self.get_seg_class_name(super_class, sub_class))
          pixel_annotations[seg_class] = np.logical_or(pixel_annotations[seg_class], one_label_pixels)

    pixel_annotations = pixel_annotations.astype(np.uint8).reshape((C, h, w))

    return {"annotation": pixel_annotations.tolist()}

  
  def get_data_gen(self, rotate_range=0, flip=False, 
                   channel_shift_range=1e-10, multiplier=0, seed=0):
    """
    Creates data augmenting generators for both images and annotations.
    Locally imports tensorflow, because don't want to import if not training.
    Requires:
      rotate_range: 0-90 degrees, range of rotations to rotate image/label
      flip: whether to randomly veritcally/horizontally flip samples
      channel_shift_range: Add colour changes to images in range [0..255]
      multiplier: Return x multiplier of the data
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    data_gen_X = ImageDataGenerator(rotation_range=rotate_range,
                                    horizontal_flip=flip,
                                    vertical_flip=flip,
                                    channel_shift_range=channel_shift_range,
                                    fill_mode='constant',
                                    cval=0)
    data_gen_Y = ImageDataGenerator(rotation_range=rotate_range,
                                    horizontal_flip=flip,
                                    vertical_flip=flip,
                                    channel_shift_range=1e-10,
                                    fill_mode='constant', 
                                    cval=0)
    return data_gen_X, data_gen_Y, multiplier, seed


  def get_batch(self, indices, train_val_test, classes_of_interset=[]):
    """
    Returns the batch of images and labels associated with the images,
    based on the list of indicies to look up.
    Requires:
      indices: list of indices with which to make a batch
      classes_of_interest: list of class names to get annotations for

    Format: (block of images, block of labels)
    """

    # Initialise path based on argument
    path = self.train_path
    if train_val_test == "val":
      path = self.val_path
    elif train_val_test == "test":
      path = self.test_path

    # Filter label classes by classes_of_interest
    indices_of_interest = []
    for class_ in classes_of_interset:
      try:
        index = self.seg_classes.index(class_)
      except ValueError:
        raise ValueError("Invalid class name in classes_of_interest.")
      indices_of_interest.append(index)

    # interested in all classes if no classes specified
    if indices_of_interest == []:
      indices_of_interest = list(range(len(self.seg_classes)))

    # Accumulators for images and annotations in batch
    images, annotations = [], []
    for i in indices:
      image = Image.open(os.path.join(path, 'images', f'{i}.jpg'))
      image = np.array(image)

      # Filter out classes we don't want then reshape to (h,w,C) dimensions
      with open(os.path.join(path, 'annotations', f'{i}.json'), 'r') as ann:
        annotation = np.array(json.load(ann)['annotation'])
        annotation = np.moveaxis(annotation[indices_of_interest], 0, -1)

      images.append(image)
      annotations.append(annotation)

    # Return tuple by stacking them into blocks
    images, annotations = np.stack(images), np.stack(annotations)

    # call function from data_augmentation.pyplot
    if self.augment:
      images, annotations = augment_data(images, annotations, *self.augment)

    # TODO: Normalize images
    return images, annotations


  def draw_mask_on_im(self, im_path, masks):
    """
    Helper method that opens an image, draws the segmentation masks in `masks`
    as bitmaps, and then returns the masked image.
    Requires:
      im_path: Path to .jpg image
      masks: Array shaped as: #C x h x h
    """
    # Open the image and set up an ImageDraw object
    im = Image.open(im_path).convert('RGB')
    im_draw = ImageDraw.Draw(im)

    # Draw the bitmap for each class
    for i, mask in enumerate(masks):
      mask_im = Image.fromarray(mask.astype(np.uint8) * 64, mode='L')
      im_draw.bitmap((0,0), mask_im, fill=self.class_colors[i % len(self.class_colors)])
    
    return im


  def save_preds(self, image_indices, batch_preds, metrics, set_type="val"):
    """
    Saves the model's predictions as annotated images in `../model_path/images/`. 
    Corresponding images from image_indices are accessed from image_dir.
    Also copies the ground truth annotation to the `../model_path/annotations/` dir.\n
    Requires:\n
      image_indices: A list of indices corresponding to images stored in directory
                     image_dir/images \n
      batch_preds: The model predictions, shaped: (n x h x w x #C), each pixel between 0,1.\n
      metrics: List of metric dict containing image iou/prec/... (per class, average etc.)\n
      set_type: The directory that image_indices corresponds to. (Usually val)\n
    """
    # Path from where images will be copied
    path = self.val_path
    if set_type == "train":
      path = self.train_path
    elif set_type == "test":
      path = self.test_path
    
    # Save the images annotated with their predicted labels
    for i, image_ind in enumerate(image_indices):
      im_path = os.path.join(path, 'images', f'{image_ind}.jpg')

      # Reshape to #C x h x w dimensions
      pred_masks = batch_preds[i]
      pred_masks = np.moveaxis(pred_masks, -1, 0)
      
      # Draw pred masks on image, save prediction
      pred_im = self.draw_mask_on_im(im_path, pred_masks)
      pred_im.save(os.path.join(self.model_path, 'preds', f'{set_type}_pred_{image_ind}.jpg'))

      # Save metrics for prediction
      metric_path = os.path.join(self.model_path, 'preds', f'{set_type}_metrics_{image_ind}.json')
      with open(metric_path, 'w') as f:
        json.dump(metrics[i], f, indent=2)

      # Save associated image annotated with ground truth masks
      with open(os.path.join(path, 'annotations', f'{image_ind}.json')) as f:
        try: annotation = json.load(f)
        except: annotation = {}
      gt_masks = np.array(annotation["annotation"])
      gt_im = self.draw_mask_on_im(im_path, gt_masks)
      gt_im.save(os.path.join(self.model_path, 'preds', f'{set_type}_gt_{image_ind}.jpg'))
      

  def visualize_tile(self, index, directory="train"):
    """
    Provides a visualization of the tile and its corresponding annotation/ label in one
    of the train/test/val directories as specified. 
    Requires:
      index: A valid index in one of train/test/val
    """
    if directory == "train":
      path = self.train_path
    elif directory == "test":
      path = self.test_path
    elif directory == "val":
      path = self.val_path
    else: raise ValueError("Can only visualize annotations from train/val/test.")

    # Image visualization
    fig, ax = plt.subplots(nrows=1, ncols=1)

    with open(os.path.join(path, 'annotations', f'{index}.json')) as f:
      try: annotation = json.load(f)
      except: annotation = {}

    class_masks = np.array(annotation["annotation"])#.reshape(C, h, w)

    # Draw masks on image
    im_path = os.path.join(path, 'images', f'{index}.jpg')
    masked_im = self.draw_mask_on_im(im_path, class_masks)
    ax.imshow(masked_im)
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
  if ds.data_sizes[0] == 0:
    ds.build_dataset()
  print(ds.seg_classes)

  # Visualize tiles.
  if args.tile:
    inds = random.sample(range(ds.data_sizes[0]), 20)
    for i in inds:
      ds.visualize_tile(i, directory='train')

