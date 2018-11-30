from minimum_bounding_box import MinimumBoundingBox
import overpy
import numpy as np
import math
import matplotlib.pyplot as plt
import statistics as st
from get_bounding_boxes import get_two_closest_points
import data_extract

def inside_box(point, bbox):
    corner, closest, second_closest = get_two_closest_points(bbox)
    short_side = np.array(np.subtract(corner,closest))
    long_side = np.array(np.subtract(second_closest,corner))
    closest_to_point = np.array(np.subtract(point,closest))
    second_closest_to_point = np.array(np.subtract(point, second_closest))
    inside = ((0 <= np.dot(short_side, closest_to_point) <= np.dot(short_side, short_side)) and (0 <= np.dot(long_side,second_closest_to_point) <= np.dot(long_side,long_side)))
    return inside

#if a pixel is within a box, assign it the boxes properties. assuming pixel is (x,y)
def assign_pixels(image_array, bboxes):
    pixel_labels = np.zeros(image_array.shape[:2])
    for r in range(0, image_array.shape[0]):
        for c in range(0, image_array.shape[1]):
            dx = 228
            dy = 228
            heading = 0
            width = 0
            length = 0
            in_a_box = 0
            for bbox in bboxes:
                pixel = (r, c)
                if inside_box(pixel, bbox):
                    new_dx = abs(pixel[0] - bbox[0])
                    new_dy = abs(pixel[1] - bbox[1])
                    if(np.sqrt(new_dx**2 + new_dy**2) <= np.sqrt(dx**2 + dy**2)):
                        dx = new_dx
                        dy = new_dy
                        heading, width, length = bbox[2:]
                        in_a_box = 1
            pixel_labels[r,c] = [dx, dy, heading, width, length, in_a_box]
    return pixel_labels

if __name__ == "__main__":
    # get the image as an array
    im_array = data_extract.image_to_np_array("./ibm")

    # get bounding boxes from OSM
    bboxes = get_bounding_boxes.get_bounding_boxes(YOLO = False)
    # print(len(bboxes)) # We have 2990 buildings. Seems like more than enough
    pixels = get_bounding_boxes.OSM_to_pixels(
        im_array.shape[:2], bboxes, YOLO=False, PIXOR = True)

    # visualize_bounding_boxes(im_array, pixels, YOLO=True)
    indices_to_remove = list(range(4,24)) + list(range(3+23,24+23)) + [x+(23*2) for x in [2,3,4,5,6,7,8,12,13,14,15,16,17,17,22]] +[x+(23*3) for x in [0,10,11,12,13,14,15,16,22]] +[x+(23*4) for x in [10,11,13,14,15]] +[x+(23*5) for x in [0,8,9,10,11,14,15,16,17,18]] +[x+(23*6) for x in [6,7,8,9,10,15,16,17,18]] +[x+(23*7) for x in [10,11,17,18,19,22]] +[x+(23*8) for x in [18,19,20,21]] +[x+(23*9) for x in [20,21,22]] +[x+(23*10) for x in [4,5,6,20,21,22]] +[x+(23*11) for x in [5,6,7,20,21,22]] +[x+(23*12) for x in [1,2,20,21,22]] + [x+(23*13) for x in [1,2]] + [x+(23*16) for x in [0,1]] + [x+(23*17) for x in [0,1]] + [x+(23*18) for x in [0,1,2]] + [x+(23*19) for x in [0,1,2,3,17,18]] + [x+(23*20) for x in [0,1,2,3,4,17,18]] + [x+(23*21) for x in [0,1,2,3,4,5]] + [x+(23*22) for x in [0,1,2,3,4,5]]

    # # tile the image, which returns a list
    # tile_list = tile.tile_image(im_array, pixels, 228, indices_to_remove)
    # save_data(tile_list)
