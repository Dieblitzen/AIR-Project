from Dataset import Dataset
from minimum_bounding_box import MinimumBoundingBox
import os
import numpy as np
import scipy.misc
import math
from PIL import Image
import json
from lxml import etree
import random
from shutil import copyfile
from functools import reduce
from scipy.spatial import Delaunay

# Visualising
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from shapely.geometry.polygon import Polygon

import logging
logging.basicConfig(level=logging.INFO, filename="tile_logfile", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

INDICES_TO_REMOVE = list(range(4,24)) + list(range(3+23,24+23)) + [x+(23*2) for x in [2,3,4,5,6,7,8,12,13,14,15,16,17,17,22]] +[x+(23*3) for x in [0,10,11,12,13,14,15,16,22]] +[x+(23*4) for x in [10,11,13,14,15]] +[x+(23*5) for x in [0,8,9,10,11,14,15,16,17,18]] +[x+(23*6) for x in [6,7,8,9,10,15,16,17,18]] +[x+(23*7) for x in [10,11,17,18,19,22]] +[x+(23*8) for x in [18,19,20,21]] +[x+(23*9) for x in [20,21,22]] +[x+(23*10) for x in [4,5,6,20,21,22]] +[x+(23*11) for x in [5,6,7,20,21,22]] +[x+(23*12) for x in [1,2,20,21,22]] + [x+(23*13) for x in [1,2]] + [x+(23*16) for x in [0,1]] + [x+(23*17) for x in [0,1]] + [x+(23*18) for x in [0,1,2]] + [x+(23*19) for x in [0,1,2,3,17,18]] + [x+(23*20) for x in [0,1,2,3,4,17,18]] + [x+(23*21) for x in [0,1,2,3,4,5]] + [x+(23*22) for x in [0,1,2,3,4,5]]

class PIXOR_Dataset(Dataset):
  """
  The 'PIXOR_Dataset' class inherits from the parent 'Dataset' class and provides
  functionality to convert the images and annotations into the format required
  by the PIXOR architecture 
  """

  def __init__(self, data_path, train_val_test=(0.8, 0.1, 0.1)):
    """
    Initialises a 'PIXOR_Dataset' object by calling the superclass initialiser.

    The difference between a PIXOR_Dataset object and a Dataset object is the annotation.
    The PIXOR_Dataset object will therefore override the self.annotations_path and
    self.annotation_list attributes such that the building labels are in XML format.
    """
    assert (train_val_test[0] + train_val_test[1] + train_val_test[2]
            ) == 1, 'Train, val and test percentages should add to 1'
    assert train_val_test[0] > 0 and train_val_test[1] > 0 and train_val_test[
        2] > 0, 'Train, val and test percentages should be non-negative'

    Dataset.__init__(self, data_path)

    self.train_val_test = train_val_test
    self.train_path = self.data_path + '/pixor/train'
    self.val_path = self.data_path + '/pixor/val'
    self.test_path = self.data_path + '/pixor/test'

    if not os.path.isdir(self.data_path + '/pixor'):
      print(f"Creating directory to store PIXOR formatted dataset.")
      os.mkdir(self.data_path + '/pixor')

    # Create train, validation, test directories, each with an images and annotations
    # sub-directory
    for directory in [self.train_path, self.val_path, self.test_path]:
      if not os.path.isdir(directory):
          os.mkdir(directory)

      if not os.path.isdir(directory + '/images'):
          os.mkdir(directory + '/images')

      if not os.path.isdir(directory + '/annotations'):
          os.mkdir(directory + '/annotations')
  
  def build_dataset(self):
    """
    IMPORTANT: remove unwanted tiles before running this.

    OSM node sets have already been matched with the tiles they are in.
    Each node set is matched with a maximum of one tile, which constrains the
    number of boxes to check.

    Run get_bounding_boxes to convert each set of nodes to a box with 4 corners.
    Run some method to get each box in terms of center, width, height, heading.
    For each tile, call boxes_in_tile_pixor in a constrained setting.
    
    """

    data = list(zip(self.img_list, self.annotation_list))
    random.shuffle(data)
    shuffled_img, shuffled_annotations = zip(*data)

    train, val, test = self.train_val_test

    logging.info("number of tiles: " + str(len(shuffled_img)))
    # for each tile
    for i in range(0, len(shuffled_img)):
      logging.info("tile " + str(i))
      img_path = self.images_path + "/" + shuffled_img[i]
      labels_path = self.annotations_path + "/" + shuffled_annotations[i]

      # load data from json format
      with open(labels_path) as f:
        try:
          buildings_dict = json.load(f)
          buildings_list = list(buildings_dict.values())
        except:
          buildings_list = []

      # convert each node set to a (bbox as 4 corners)
      corner_boxes = self.get_rects(buildings_list)
      # convert each (bbox as 4 corners) to a PIXOR box
      pixor_boxes = self.create_pixor_labels(corner_boxes)
      # assign to pixels
      box_labels, class_labels = self.boxes_in_pixels(pixor_boxes, corner_boxes, (228, 228))

      if i < math.floor(train*len(shuffled_img)):
        # Copy image to train folder
        copyfile(
            f"{self.images_path}/{shuffled_img[i]}", f"{self.train_path}/images/{i}.jpg")
        # Create annotation matrices
        np.save(self.train_path + "/box_annotations/" + str(i), box_labels)
        np.save(self.train_path + "/class_annotations/" + str(i), class_labels)
      
      elif i < math.floor((train+val)*len(shuffled_img)):
        # Add to val folder
        ind = i - math.floor((train)*len(shuffled_img))
        copyfile(
            f"{self.images_path}/{shuffled_img[i]}", f"{self.val_path}/images/{ind}.jpg")
        # Create annotation matrices
        np.save(self.val_path + "/box_annotations/" + str(ind), box_labels)
        np.save(self.val_path + "/class_annotations/" + str(ind), class_labels)

      else:
        # Add to test folder
        ind = i - math.floor((train+val)*len(shuffled_img))
        copyfile(
            f"{self.images_path}/{shuffled_img[i]}", f"{self.test_path}/images/{ind}.jpg")
        # Create annotation matrices
        np.save(self.test_path + "/box_annotations/" + str(ind), box_labels)
        np.save(self.test_path + "/class_annotations/" + str(ind), class_labels)
      
  
  def create_pixor_labels(self, corner_labels):
    """ Input: Set of bounding boxes, where each box is repped as 4 corners.
        Output: Set of bounding boxes, where each box is repped PIXOR-style. """
    bb_pixels = []
    for corner_label in corner_labels:
      centreX, centreY = self.get_pixor_center(corner_label)
      heading, width, length = self.get_pixor_box_dimensions(corner_label)
      dimensions = [centreX, centreY, heading, width, length]
      bb_pixels.append(dimensions)
    return bb_pixels

  def boxes_in_pixels(self, bboxes, corner_boxes, tile_shape):
    
    logging.info("len of boxes_within_tile: " + str(len(bboxes)))
    
    pixel_box_labels = np.zeros((228, 228, 6))
    pixel_class_labels = np.zeros((228, 228, 1))

    counter = 0
    sec_counter = 0
    
    logging.info("boxes within tile: ")
    logging.info(bboxes)
    
    for r in range(0, tile_shape[0]):
        for c in range(0, tile_shape[1]):
            dx = 228
            dy = 228
            heading = 0
            width = 0
            length = 0
            in_a_box = 0
            for bbox_index in range(0,len(bboxes)):

                pixel_xyform = (c, r)
                
                if self.inside_box(pixel_xyform, corner_boxes[bbox_index]):
                    new_dx = -1*(pixel_xyform[0]) - bboxes[bbox_index][0]
                    new_dy = -1*(pixel_xyform[1]) - bboxes[bbox_index][1]
                    counter+=1
                        
                    if(np.sqrt(new_dx**2 + new_dy**2) <= np.sqrt(dx**2 + dy**2)):
                        sec_counter+=1
                        dx = new_dx
                        dy = new_dy
                        heading, width, length = bboxes[bbox_index][2:]
                        in_a_box = 1
                
            pixel_box_labels[r, c, :] = [int(dx), int(dy), np.sin(heading), np.cos(heading), int(width), int(length)]
            pixel_class_labels[r, c] = in_a_box

    # logging.info("Things that got inside first if: " + str(counter))
    # logging.info("Things that got inside second if: " + str(sec_counter))
    unique_boxes = self.extract_positive_labels(pixel_box_labels)
    
    # print("len of unique_boxes: " + str(len(unique_boxes)))
    logging.info("len of unique_boxes: " + str(len(unique_boxes)))
    logging.info("unique boxes: ")
    logging.info(unique_boxes)
    # print("end of unique boxes \n\n")
    logging.info("end of unique boxes")
    
    return pixel_box_labels, pixel_class_labels

  def get_rects(self, buildings):
    """ Returns each bounding box in terms of its 4 corners. """
    corner_boxes = []
    for building in buildings:
      bounding_box = list(MinimumBoundingBox(building).corner_points)
      corner_boxes.append(bounding_box)
    return corner_boxes
  
  def get_pixor_center(self, bbox):
    sorted_by_lat = sorted(bbox, key= lambda pair: pair[0])
    sorted_by_lon = sorted(bbox, key= lambda pair: pair[1])
    center_y = math.floor((sorted_by_lon[0][1] + sorted_by_lon[3][1])/2)
    center_x = math.floor((sorted_by_lat[0][0] + sorted_by_lat[3][0])/2)
    return center_x, center_y
  
  #returns the angle of the heading. bbox is coordinates of the four corners
  def get_pixor_box_dimensions(self, bbox):
    corner, closest, second_closest, sorted_distances = self.get_two_closest_points(bbox)
    vector = np.array(np.subtract(second_closest,corner)) if corner[1] < second_closest[1] else np.array(np.subtract(corner,second_closest))
    unit_vector = vector / np.linalg.norm(vector)
    width, length = math.floor(sorted_distances[1][1]), math.floor(sorted_distances[2][1])
    return np.arccos(np.clip(np.dot(unit_vector, (1,0)), -1.0, 1.0)), width, length
  
  def inside_box(self, p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """

    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0
  
  #new procedure:
  # 1. take random three corner
  # 2. find biggest angle, assign that as your corner
  # 3. of the two remaining points, the closer one is the width vector, the farthest is length
  def get_two_closest_points(self, bbox):
      points = np.array(bbox[0:4])
      # angles = [(i, calculate_angle(point, [p for p in points if not np.array_equal(p,point)])) for i, point in enumerate(points)]
      # sorted_angles = sorted(angles, key= lambda pair: pair[1])
      corner = points[0]

      distances = [(i,np.linalg.norm(corner-c)) for i, c in enumerate(points)]
      sorted_distances = sorted(distances, key= lambda pair: pair[1])

      closest_index, second_closest_index = sorted_distances[1][0], sorted_distances[2][0]
      closest, second_closest = np.array(points[closest_index]), np.array(points[second_closest_index])
      return corner, closest, second_closest, sorted_distances
  
  def extract_positive_labels(self, bboxes):
    unique_boxes_set = set()
    unique_boxes = {}
    for r in range(0, bboxes.shape[0]):
        for c in range(0, bboxes.shape[1]):
            if bboxes[r,c][-1] != 0. and tuple(bboxes[r,c][2:]) not in unique_boxes_set:
                unique_boxes_set.add(tuple(bboxes[r,c][2:]))
                unique_boxes[str(r) + "_" + str(c)] = bboxes[r,c]

    return unique_boxes