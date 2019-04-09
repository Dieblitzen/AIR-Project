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

    cos_angle = np.cos(np.arccos(cos_angle))
    sin_angle = np.sin(np.arcsin(sin_angle))
    temp_x = point[0] - center_x
    temp_y = point[1] - center_y
    cos_angle = np.cos(2*math.pi - np.arccos(cos_angle))
    sin_angle = np.sin(2*math.pi - np.arcsin(sin_angle))

    rotated_x = temp_x*cos_angle - temp_y*sin_angle
    rotated_y = temp_x*sin_angle + temp_y*cos_angle
    x = rotated_x + center_x
    y = rotated_y + center_y
    return x, y

#Converts pixor description of a box into four coordinates.
def pixor_to_corners(box):
    center_x, center_y, cos_angle, sin_angle, width, length = box
    four_corners = [(center_x+width//2, center_y+length//2),
        (center_x+width//2, center_y-length//2),
        (center_x-width//2, center_y-length//2),
        (center_x-width//2, center_y+length//2)]

    rotated_corners = [rotate_point(corner, center_x, center_y, cos_angle, sin_angle) for corner in four_corners]
    return rotated_corners

def visualize_pixels(image_array, bboxes):
    plt.imshow(image_array)

    for box in bboxes:
        x, y = box
        plt.plot(x, y, 'o', markersize = 20, markerfacecolor='red')

    plt.show()

def visualize_bounding_boxes(image_array, bboxes, save, counter, save_path):
    # Visualize bounding boxes on an image with bb_pixels either as horizontal boxes
    plt.clf()
    plt.imshow(image_array)
    print("len of bboxes")
    print(len(bboxes))
    for box in bboxes:
#         print("visualizing box:")
#         print(box)
        coordinates = pixor_to_corners(box)
        if not math.isnan(coordinates[0][0]):
            poly = Polygon(coordinates)
#             print("here are coordinates: " + str(coordinates))
#             print(box)
            x, y = poly.exterior.xy
            plt.plot(x,y)
        # print(box)
    # if counter % 5 == 0 or counter >375:
    #     plt.savefig('tile_images/tile'+str(counter)+".png")
    if save:
        plt.savefig(save_path+ '/' + str(counter)+ ".png")
    else:
        plt.show()



if __name__ == "__main__":
    
    data_path = '../WhitePlains_data'
    
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
                # print(bboxes[r,c])
                if class_labels[r,c][0] > .8:
                # if bboxes[r,c][-1] != 0:
                    center_x = (c) + (int(bboxes[r,c][0]))
                      
                    center_y = (r) + (int(bboxes[r,c][1]))
                    center = np.array([center_x, center_y])
                    box = np.concatenate([center, bboxes[r,c][2:]])
                    pixels_to_color.append((r,c))
                    if tuple(bboxes[r,c][2:]) not in unique_boxes_set:
                        unique_boxes_set.add(tuple(bboxes[r,c][2:]))
                        boxes_in_image.append(box)
                        print(box)
                        # if counter == 915:
                        #     print("counter is: ")
                        #     print(counter)
                        #     print("label is: ")
                        #     print(bboxes[r,c])
                        #     print("box is: ")
                        #     print(box)
                        #     print("pixels are: ")
                        #     print((r,c))
                        counter+=1

        # visualize_pixels(image, pixels_to_color)
        visualize_bounding_boxes(image, boxes_in_image, True, i, 'label_visualized')
