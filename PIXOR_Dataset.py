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

INDICES_TO_REMOVE = {3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37, 39, 40, 42, 43, 45, 46, 49, 50, 51, 52, 53, 54, 55, 56, 57, 63, 64, 65, 72, 73, 74, 75, 76, 77, 79, 81, 86, 96, 97, 98, 100, 101, 103, 108, 110, 116, 117, 118, 119, 120, 121, 124, 125, 126, 138, 139, 140, 142, 147, 148, 149, 150, 151, 155, 160, 173, 174, 175, 176, 178, 197, 198, 199, 200, 201, 222, 223, 229, 230, 245, 247, 268, 269, 270, 291, 292, 293, 299, 370, 409, 410, 414, 432, 433, 434, 449, 455, 456, 457, 458, 471, 472, 478, 479, 480, 481, 482, 501, 503, 504, 505, 506}
IMAGE_SIZE = 224
#create mapping for classes to class label
class_map = {}
cnt = 1

with open('classes.json', 'r') as filename:
      try: 
        classes = json.load(filename)
      except ValueError:
        classes = {}

for super_class, sub_class_labels in classes.items():
  for sub_class in sub_class_labels:
    class_map[sub_class] = cnt
    cnt+=1
    


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

      if not os.path.isdir(directory + '/class_annotations'):
          os.mkdir(directory + '/class_annotations')

      if not os.path.isdir(directory + '/box_annotations'):
          os.mkdir(directory + '/box_annotations')
  
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
      labels_in_tile = {}
      with open(labels_path, 'r') as filename:
        try: 
          labels_in_tile = json.load(filename)
        except ValueError:
          labels_in_tile = {}
          
      buildings_list = []

      for super_class, sub_class_labels in labels_in_tile.items():
        for sub_class, labels in sub_class_labels.items(): #labels is all building
          if super_class == 'building':
            for label in labels: 
              #each label is one set of points for a particular building
              buildings_list.append((sub_class, label))
              #print(m)
    

      # convert each node set to a (bbox as 4 corners)
      # corner_boxes contains class label
      corner_boxes = self.get_rects(buildings_list)

      # convert each (bbox as 4 corners) to a PIXOR box
      pixor_boxes = self.create_pixor_labels(corner_boxes)
      # assign to pixels
      box_labels, class_labels = self.boxes_in_pixels_2(pixor_boxes, corner_boxes, (IMAGE_SIZE, IMAGE_SIZE))

      #ADD PLOTING 
      im = Image.open(img_path)
      im_arr = np.array(im)
      f = plt.figure()
      f.add_subplot(1, 2, 1)
      plt.imshow(im_arr)
      f.add_subplot(1, 2, 2)
      plt.imshow(class_labels)
      plt.show(block=True)


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
    for _,corner_label in corner_labels:
      centreX, centreY = self.get_pixor_center(corner_label)
      heading, width, length = self.get_pixor_box_dimensions(corner_label)
      dimensions = [centreX, centreY, heading, width, length]
      bb_pixels.append(dimensions)
    return bb_pixels

 
# boxes_in_pixels_2: 
# for every boundingbox with corners (x1, y1), (x2, y2), (x3, y3), (x4, y4), take the pixels (x, y) inbetween these four points
# check if this pixel (x, y) using `inside_box`, if yes, then calculate pixel_box_labels and class label etc. If not, leave it alone.
# set the pixel_box_labels of the pixels not in a box to some default value
  def boxes_in_pixels_2(self, bboxes, corner_boxes, tile_shape):
    logging.info("len of boxes_within_tile: " + str(len(bboxes)))
    
    pixel_box_labels = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 6))
    pixel_class_labels = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

    counter = 0
    sec_counter = 0
    
    logging.info("boxes within tile: ")
    logging.info(bboxes)

    print("Bboxes:", len(bboxes))
    for bbox_index in range(0, len(bboxes)):
      dx = IMAGE_SIZE
      dy = IMAGE_SIZE
      heading = 0
      width = 0
      length = 0
      in_a_box = 0
     
      heading, width, length = bboxes[bbox_index][2:]
      corner_points = corner_boxes[bbox_index][-1]
      x_coords, y_coords = zip(*corner_points)
      print('x_coords', x_coords)
      print('y_coords', y_coords)
      

      for c in range(max(int(min(x_coords)), 0), min(int(max(x_coords)), 224)):
        for r in range(max(int(min(y_coords)), 0), min(int(max(y_coords)), 224)):
          pixel_xyform = (c, r)
          in_a_box = 0
          if self.inside_box(pixel_xyform, corner_boxes[bbox_index][1]):
            new_dx = -1*(pixel_xyform[0] - bboxes[bbox_index][0])
            new_dy = -1*(pixel_xyform[1] - bboxes[bbox_index][1])
            counter+=1
            sec_counter+=1
            dx = new_dx
            dy = new_dy
            heading, width, length = bboxes[bbox_index][2:]
            in_a_box = class_map[corner_boxes[bbox_index][0]]
                
          pixel_box_labels[r, c, :] = [int(dx), int(dy), np.sin(heading), np.cos(heading), int(width), int(length)]
          pixel_class_labels[r, c] = in_a_box
    unique_boxes = self.extract_positive_labels(pixel_box_labels)
    
    logging.info("len of unique_boxes: " + str(len(unique_boxes)))
    logging.info("unique boxes: ")
    logging.info(unique_boxes)
    logging.info("end of unique boxes")
    
    return pixel_box_labels, pixel_class_labels

  def get_rects(self, buildings):
    """ Returns each bounding box in terms of its 4 corners. """
    corner_boxes = []
    for building in buildings:
      bounding_box = list(MinimumBoundingBox(building[1]).corner_points)
      corner_boxes.append((building[0],bounding_box))
    return corner_boxes
  
  def get_pixor_center(self, bbox):
    sorted_by_lat = sorted(bbox, key= lambda pair: pair[0])
    sorted_by_lon = sorted(bbox, key= lambda pair: pair[1])
    center_y = math.floor((sorted_by_lon[0][1] + sorted_by_lon[3][1])/2)
    center_x = math.floor((sorted_by_lat[0][0] + sorted_by_lat[3][0])/2)
    return center_x, center_y
  
  #returns the angle of the heading. bbox is coordinates of the four corners
  # angle of head
  def get_pixor_box_dimensions(self, bbox):
    corner, closest, second_closest, sorted_distances = self.get_two_closest_points(bbox)
    vector = np.array(np.subtract(second_closest,corner)) if corner[1] < second_closest[1] else np.array(np.subtract(corner,second_closest)) #want a upward pointing vector
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

  def get_batch(self, start_index, batch_size, base_path):
    """
    Method 3)
    Gets batch of tiles and labels associated with data start_index.

    Returns:
    [(tile_array, list_of_buildings), ...]
    """
    batch = np.zeros((batch_size, 3))
    for i in range(start_index, start_index + batch_size):
      batch[i] = self.get_tile_and_label(i, base_path)
    
    return batch

  def get_tile_and_label(self, index, base_path):
    """
    Method 2)
    Gets the tile and label associated with data index.

    Returns:
    (tile_array, dictionary_of_buildings)
    """

    # Open the jpeg image and save as numpy array
    im = Image.open(base_path + '/images/' + str(index) + '.jpg')
    im_arr = np.array(im)

    # Open the json file and parse into dictionary of index -> buildings pairs
    box_annotation = np.load(base_path + '/box_annotations/' + str(index) + '.npy')
    class_annotation = np.load(base_path + '/class_annotations/' + str(index) + '.npy')
    
    return np.array([im_arr, box_annotation, class_annotation])