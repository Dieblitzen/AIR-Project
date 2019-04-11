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
from matplotlib.path import Path
import xml.etree.ElementTree as ET


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

        if not os.path.isdir(self.data_path + '/im_seg'):
            print(f"Creating directory to store segmentation formatted dataset.")
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
                self.convert_json(
                    f"{self.annotations_path}/{shuffled_annotations[i]}", f"{self.train_path}/annotations/{i}.json", f"{i}.jpg")
            elif i < math.floor((train+val)*len(shuffled_img)):
                # Add to val folder
                ind = i - math.floor((train)*len(shuffled_img))
                copyfile(
                    f"{self.images_path}/{shuffled_img[i]}", f"{self.val_path}/images/{ind}.jpg")
                self.convert_json(
                    f"{self.annotations_path}/{shuffled_annotations[i]}", f"{self.val_path}/annotations/{ind}.json", f"{ind}.jpg")
            else:
                # Add to test folder
                ind = i - math.floor((train+val)*len(shuffled_img))
                copyfile(
                    f"{self.images_path}/{shuffled_img[i]}", f"{self.test_path}/images/{ind}.jpg")
                self.convert_json(
                    f"{self.annotations_path}/{shuffled_annotations[i]}", f"{self.test_path}/annotations/{ind}.json", f"{ind}.jpg")
            # increment index counter
            i += 1

    def convert_json(self, path_to_file, path_to_dest, img_name):
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
                buildings_dict = self.format_annotation(json.load(f))
            except Exception as e:
                print(e)
                buildings_dict = {}

        # Add corresponding image name to annotaion
        buildings_dict["img"] = img_name

        # save annotation in file
        with open(path_to_dest, 'w') as dest:
            json.dump(buildings_dict, dest)

    def format_annotation(self, buildings_dict):
        """
        Helper method only called in convert_json that takes a dictionary of
        building coordinates and creates a one-hot encoding for each pixel 
        for an image { "annotation" : [pixel-wise encoding] }
        """

        w, h, _ = self.get_img_size()  # Right order?

        # make a canvas with pixel coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x, y = x.flatten(), y.flatten()
        # A list of all pixels in terms of indices
        all_pix = np.vstack((x, y)).T

        acc = np.zeros((all_pix.shape[0],), dtype=bool)
        for _, v in buildings_dict.items():
            p = Path(v)
            one_building_pixels = p.contains_points(all_pix)

            acc = np.logical_or(
                acc, one_building_pixels)

        return {"annotation": acc.flatten().tolist()}

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
                ann = json.load(f)
            except:
                ann = {}

        w, h, _ = self.get_img_size()  # Right?

        pixel_annotation = np.array(ann["annotation"]).reshape(w, h)  # Right?

        # Check our results
        ax.imshow(pixel_annotation, alpha=0.3)

        plt.show()
