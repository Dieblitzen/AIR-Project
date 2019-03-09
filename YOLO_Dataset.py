from Dataset import Dataset
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

# Visualising
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from shapely.geometry.polygon import Polygon
import xml.etree.ElementTree as ET


class YOLO_Dataset(Dataset):
    """
    The 'YOLO_Dataset' class inherits from the parent 'Dataset' class and provides
    functionality to convert the images and annotations into the format required
    by the YOLO architecture (implemented as in Darkflow link: https://github.com/thtrieu/darkflow)

    An object of 'YOLO_Dataset' wil provide the following functionality:



    """

    def __init__(self, data_path, train_val_test=(0.8, 0.1, 0.1)):
        """
        Initialises a 'YOLO_Dataset' object by calling the superclass initialiser.

        The difference between a YOLO_Dataset object and a Dataset object is the annotation.
        The YOLO_Dataset object will therefore override the self.annotations_path and
        self.annotation_list attributes such that the building labels are in XML format.
        """
        assert (train_val_test[0] + train_val_test[1] + train_val_test[2]
                ) == 1, 'Train, val and test percentages should add to 1'
        assert train_val_test[0] > 0 and train_val_test[1] > 0 and train_val_test[
            2] > 0, 'Train, val and test percentages should be non-negative'

        Dataset.__init__(self, data_path)

        self.train_val_test = train_val_test
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
                self.json_to_xml(
                    f"{self.annotations_path}/{shuffled_annotations[i]}", f"{self.train_path}/annotations/{i}.xml", f"{i}.jpg")
            elif i < math.floor((train+val)*len(shuffled_img)):
                # Add to val folder
                ind = i - math.floor((train)*len(shuffled_img))
                copyfile(
                    f"{self.images_path}/{shuffled_img[i]}", f"{self.val_path}/images/{ind}.jpg")
                self.json_to_xml(
                    f"{self.annotations_path}/{shuffled_annotations[i]}", f"{self.val_path}/annotations/{ind}.xml", f"{ind}.jpg")
            else:
                # Add to test folder
                ind = i - math.floor((train+val)*len(shuffled_img))
                copyfile(
                    f"{self.images_path}/{shuffled_img[i]}", f"{self.test_path}/images/{ind}.jpg")
                self.json_to_xml(
                    f"{self.annotations_path}/{shuffled_annotations[i]}", f"{self.test_path}/annotations/{ind}.xml", f"{ind}.jpg")
            # increment index counter
            i += 1

    def json_to_xml(self, path_to_file, path_to_dest, img_name):
        """
        Helper method only called in split_data that takes a json file at
        path_to_file and writes a corresponding xml at path_to_dest.
        """
        # Im_size: [width, height, depth] ??? should be squares anyways
        with open(path_to_file) as f:
            try:
                buildings_dict = self.format_coords(json.load(f))
            except:
                buildings_dict = {}

        # begin creating annotation
        annotation = etree.Element('annotation')

        # Add to xml etree
        filename = etree.Element('filename')
        filename.text = img_name

        # Image size
        size = etree.Element('size')
        im_size = self.get_img_size()
        # nested elements in size
        width = etree.Element('width')
        height = etree.Element('height')
        depth = etree.Element('depth')
        width.text = str(im_size[1])
        height.text = str(im_size[0])
        depth.text = str(im_size[2])
        # append nested elements to size element
        size.append(width)
        size.append(height)
        size.append(depth)

        # append filename and size to main xml etree
        annotation.append(filename)
        annotation.append(size)

        for bbox in buildings_dict.values():
            # object for each bounding box
            obj = etree.Element('object')

            # We only have one class for now. (Note: name is label)
            name = etree.Element('name')
            name.text = "building"

            # Bounding box preocessing. We assume that bboxes are in
            # [centerX, centerY, width, height] format, and convert it to
            # x_min, x_max, y_min, y_max
            bndbox = etree.Element('bndbox')
            xmin = etree.Element('xmin')
            xmin.text = str(bbox[0] - (bbox[2]/2))

            ymin = etree.Element('ymin')
            ymin.text = str(bbox[1] - (bbox[3]/2))

            xmax = etree.Element('xmax')
            xmax.text = str(bbox[0] + (bbox[2]/2))

            ymax = etree.Element('ymax')
            ymax.text = str(bbox[1] + (bbox[3]/2))

            # Append xmin, xmax, ymin, ymax to bounding box object
            bndbox.append(xmin)
            bndbox.append(ymin)
            bndbox.append(xmax)
            bndbox.append(ymax)

            # append nested elements in obj
            obj.append(name)
            obj.append(bndbox)

            # Append the whole obj to the annotation.
            annotation.append(obj)

        # Full annotation, ready to be written
        xml_annotation = etree.ElementTree(annotation)

        # save annotation in file
        with open(path_to_dest, 'wb') as dest:
            xml_annotation.write(dest)

    def format_coords(self, buildings):
        """
        Helper method only called in json_to_xml that takes a dictionary of
        building coordinates and converts them to YOLO format, i.e.
        (centerX, centerY, width, height) for each building
        """

        for k, v in buildings.items():
            minX = reduce(lambda acc, elt: min(elt[0], acc), v, np.inf)
            minY = reduce(lambda acc, elt: min(elt[1], acc), v, np.inf)
            maxX = reduce(lambda acc, elt: max(elt[0], acc), v, -np.inf)
            maxY = reduce(lambda acc, elt: max(elt[1], acc), v, -np.inf)
            width = maxX - minX
            height = maxY - minY
            centerX = minX + width/2.0
            centerY = minY + height/2.0
            buildings[k] = [centerX, centerY, width, height]

        return buildings

    def visualize_tile(self, index, directory="train"):
        """
        Method 5)
        Provides a visualization of the tile with the tile and its corresponding annotation/ label. 
        """

        path = self.train_path
        if directory == "test":
            path = self.test_path
        elif directory == "val":
            path = self.val_path

        # Image visualization
        im = Image.open(f'{path}/images/{index}.jpg')
        im_arr = np.array(im)
        plt.imshow(im_arr)

        ann = ET.parse(f'{path}/annotations/{index}.xml').getroot()
        
        buildings_in_tile = {}
        i = 0
        for building in ann.iter("bndbox"):
          xmin = float(building.find('xmin').text)
          ymin = float(building.find('ymin').text)
          xmax = float(building.find('xmax').text)
          ymax = float(building.find('ymax').text)
          buildings_in_tile[i] = [(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin)]
          i += 1
        
        for building_coords in buildings_in_tile.values():
            poly = Polygon(building_coords)
            x, y = poly.exterior.xy
            plt.plot(x, y)

        plt.show()
