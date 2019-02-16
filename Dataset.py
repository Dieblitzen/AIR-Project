import os
import pandas as pd
import numpy as np
import geojson
import requests
import zipfile
import io
from osgeo import gdal
from time import sleep
import overpy
import pickle
import scipy.misc
import math
from PIL import Image

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

  def get_length(self):
    """
    Gets the length of the dataset (number of annotated images)
    """
    # All images in dataset
    im_list = [name for name in os.listdir(
        f"./{self.data_path}/img") if os.path.isfile(name)]

    return len(im_list)

  def get_img_size(self):
    """
    Gets the size of the images in the dataset
    """
    # Gets first image in dataset
    im = Image.open(os.listdir(f"./{self.data_path}/img")[0])
    # Returns the shape of the image
    return np.array(im).shape

  def visualize_tile_with_annotation(self, tile):
    """
    Provides a visualization of the tile with filename tile and its 
    corresponding annotation. 
    """
    im = Image.open(os.listdir(f"./{self.data_path}/{tile}"))
    plt.imshow(im)

    # TODO: Visualize bounding boxes from json format.

    plt.show()

  def visualize_dataset(self):
    """
    Provides visualization of entire dataset image area, 
    including annotations
    """
    pass
