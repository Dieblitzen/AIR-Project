import Dataset
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

        Dataset.__init__(data_path)

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
        pass

    def split_data(self):
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
            if i < train*len(shuffled_img):
                # Add to train folder
                copyfile(
                    f"{self.images_path}/{shuffled_img[i]}", f"{self.train_path}/images/{i}")
                self.json_to_xml(
                    f"{self.annotations_path}/{shuffled_annotations[i]}", f"{self.train_path}/annotations/{i}")
            elif i < (train+val)*len(shuffled_img):
                # Add to val folder
                ind = i - (train)*len(shuffled_img)
                copyfile(
                    f"{self.images_path}/{shuffled_img[i]}", f"{self.val_path}/images/{ind}")
                self.json_to_xml(
                    f"{self.annotations_path}/{shuffled_annotations[i]}", f"{self.val_path}/annotations/{ind}")
            else:
                # Add to test folder
                ind = i - (train+val)*len(shuffled_img)
                copyfile(
                    f"{self.images_path}/{shuffled_img[i]}", f"{self.test_path}/images/{ind}")
                self.json_to_xml(
                    f"{self.annotations_path}/{shuffled_annotations[i]}", f"{self.test_path}/annotations/{ind}")
            # increment index counter
            i += 1

    def json_to_xml(self, path_to_file, path_to_dest):
        """
        Helper method only called in split_data that takes a json file at
        path_to_file and writes a corresponding xml at path_to_dest.
        """

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
