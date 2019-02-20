import Dataset
import os
import numpy as np
import scipy.misc
import math
from PIL import Image
import json
from lxml import etree
import random

# Visualising
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from shapely.geometry.polygon import Polygon

class YOLO_Dataset(Dataset):
  """
  The 'YOLO_Dataset' class inherits from the parent 'Dataset' class and provides
  functionality to convert the images and annotations into the format required 
  by the YOLO architecture (implemented as in Darkflow link: https://github.com/thtrieu/darkflow)

  An object of 'YOLO_Dataset' wil provide the following functionality:



  """
  
  def __init__(self, data_path, train_val_test=(0.8,0.1,0.1)):
    """
    Initialises a 'YOLO_Dataset' object by calling the superclass initialiser.

    The difference between a YOLO_Dataset object and a Dataset object is the annotation.
    The YOLO_Dataset object will therefore override the self.annotations_path and 
    self.annotation_list attributes such that the building labels are in XML format.
    """
    assert (train_val_test[0] + train_val_test[1] + train_val_test[2]) == 1, 'Train, val and test percentages should add to 1'
    assert train_val_test[0] > 0 and train_val_test[1] > 0 and train_val_test[2] > 0, 'Train, val and test percentages should be non-negative'

    Dataset.__init__(data_path)

    self.train_path = self.data_path + '/yolo/train'
    self.val_path = self.data_path + '/yolo/val'
    self.test_path = self.data_path + '/yolo/test'

    if not os.path.isdir(self.data_path + '/yolo'):
      print(f"Creating directory to store YOLO formatted dataset.")
      os.mkdir(self.data_path + '/yolo')

    # Create train, validation, test directories, each with an images and annotations
    # sub-directory
    for directory in [self.train_path, self.val_path, self.test_path]:
      if not os.path.isdir(directory):
        os.mkdir(directory)

      if not os.path.isdir(directory + '/images'):
        os.mkdir(directory + '/images')
      
      if not os.path.isdir(directory + '/annotations'):
        os.mkdir(directory + '/annotations')
    
    data = list(zip(self.img_list, self.annotation_list))
    random.shuffle(data)
    shuffled_img, shuffled_annotations = zip(*data)

  



    
