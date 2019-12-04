import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import matplotlib.ticker as plticker
from shapely.geometry.polygon import Polygon
import scipy.misc
import math
from PIL import Image

ENTIRE_IMG_SIZE = (3648, 5280, 3)

# TODO: add option to graph on image at certain path (or maybe just give np array?)

def extract_positive_labels(bboxes):
    unique_boxes_set = set()
    unique_boxes = []
    for r in range(0, bboxes.shape[0]):
        for c in range(0, bboxes.shape[1]):
            if bboxes[r,c][-1] != 0. and tuple(bboxes[r,c][2:]) not in unique_boxes_set:
                unique_boxes_set.add(tuple(bboxes[r,c][2:]))
                unique_boxes.append(bboxes[r,c])

    return unique_boxes

#applies translation, rotation, then un-translation of four points
def rotate_point(point, center_x, center_y, cos_angle, sin_angle):
    ox, oy = center_x, center_y
    px, py = point
    angle = np.arccos(cos_angle)

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    
    return qx, qy

def tf_rotate_point(corners, center_x, center_y, cos_angle, sin_angle):
    qx = center_x + cos_angle * (corners[:, :, :, :, 0] - center_x) - sin_angle * (corners[:, :, :, :, 1] - center_y)
    qy = center_y + sin_angle * (corners[:, :, :, :, 0] - center_x) + cos_angle * (corners[:, :, :, :, 1] - center_y)
    
    rotated_points = tf.stack([qx, qy], axis=-1)
    
    return rotated_points

#Converts pixor description of a box into four coordinates.
def pixor_to_corners(box):
    center_x, center_y, sin_angle, cos_angle, width, length = box
    four_corners = [(center_x+length//2, center_y+width//2),
        (center_x+length//2, center_y-width//2),
        (center_x-length//2, center_y-width//2),
        (center_x-length//2, center_y+width//2)]

    rotated_corners = [rotate_point(corner, center_x, center_y, cos_angle, sin_angle) for corner in four_corners]
    return rotated_corners

#Converts pixor description of a box into four coordinates.
def tf_pixor_to_corners(box):
    center_x = box[:, :, :, 0:1]
    center_y = box[:, :, :, 1:2]
    sin_angle = box[:, :, :, 2:3]
    cos_angle = box[:, :, :, 3:4]
    width = box[:, :, :, 4:5]
    length = box[:, :, :, 5:6]
    
    corner_1_x = tf.divide(tf.add(center_x, length), 2)
    corner_1_y = tf.divide(tf.add(center_y, width), 2)
    corner_1 = tf.concat([corner_1_x, corner_1_y], 3)
    
    corner_2_x = tf.divide(tf.add(center_x, length), 2)
    corner_2_y = tf.divide(tf.subtract(center_y, width), 2)
    corner_2 = tf.concat([corner_2_x, corner_2_y], 3)
    
    corner_3_x = tf.divide(tf.subtract(center_x, length), 2)
    corner_3_y = tf.divide(tf.subtract(center_y, width), 2)
    corner_3 = tf.concat([corner_3_x, corner_3_y], 3)
    
    corner_4_x = tf.divide(tf.subtract(center_x, length), 2)
    corner_4_y = tf.divide(tf.add(center_y, width), 2)
    corner_4 = tf.concat([corner_4_x, corner_4_y], 3)
    
    corners = tf.stack([corner_1, corner_2, corner_3, corner_4], axis=3)

    rotated_corners = tf_rotate_point(corners, center_x, center_y, cos_angle, sin_angle)
    return rotated_corners

def visualize_pixels(image_array, bboxes):
    plt.imshow(image_array)

    for box in bboxes:
        x, y = box
        plt.plot(x, y, 'o', markersize = 20, markerfacecolor='red')

    plt.show()

def visualize_bounding_boxes(image_array, bboxes, save, counter, save_path, box_color):
    # Visualize bounding boxes on an image with bb_pixels either as horizontal boxes
    if box_color == 'blue':
        plt.clf()
    plt.imshow(image_array)
    print("len of bboxes", len(bboxes))
    for box in bboxes:
        coordinates = pixor_to_corners(box)
        if not math.isnan(coordinates[0][0]):
            poly = Polygon(coordinates)

            x, y = poly.exterior.xy
            plt.plot(x,y, color = box_color)

    if save:
        plt.savefig(save_path+ '/' + str(counter)+ ".png")
    else:
        plt.show()



if __name__ == "__main__":
    
    data_path = '../WhitePlains_data'
    truth_counter = 0
    for i in range(300):
        image = Image.open(data_path + "/pixor/train/images/" + str(i) + ".jpg")
        image = np.array(image)
        bboxes = np.load(data_path + "/pixor/train/box_annotations/"+ str(i) + ".npy")
        # bboxes = np.load("./data_path/pixor/val/box_annotations/"+ str(i) + ".npy")
        class_labels = np.load(data_path + "/pixor/train/class_annotations/"+ str(i) + ".npy")
        unique_boxes_set = set()
        pixels_to_color = []
        boxes_in_image = []
        counter = 0
        for r in range(0, image.shape[0]):
            for c in range(0, image.shape[1]):
                if class_labels[r,c][0] > .8:
                    center_x = (c) + (int(bboxes[r,c][0]))
                    truth_counter+=1
                    center_y = (r) + (int(bboxes[r,c][1]))
                    center = np.array([center_x, center_y])
                    box = np.concatenate([center, bboxes[r,c][2:]])
                    pixels_to_color.append((r,c))
                    if tuple(bboxes[r,c][2:]) not in unique_boxes_set:
                        unique_boxes_set.add(tuple(bboxes[r,c][2:]))
                        boxes_in_image.append(box)
                        print(box)
                      
                        counter+=1

        visualize_bounding_boxes(image, boxes_in_image, True, i, 'label_visualized', 'blue')