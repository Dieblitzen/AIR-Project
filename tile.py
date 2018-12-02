# Data processising

import numpy as np
import get_bounding_boxes
from sklearn.feature_extraction import image
from associate_pixels import inside_box

# Gets the bounding boxes per tile, with centre relative to tile coordinates.
def boxes_in_tile(bboxes, row_start, row_end, col_start, col_end):

    bboxes_in_tile = []
    selected_indices = []

    for i in range(len(bboxes)):
        centreX = bboxes[i][0]
        centreY = bboxes[i][1]

        if (col_start <= centreX < col_end) and (row_start <= centreY < row_end):

            # Changing bbox centre to be relative to tile
            newX = bboxes[i][0] - col_start
            newY = bboxes[i][1] - row_start

            # Don't forget to carry remaining info that was unmutated
            bboxes_in_tile.append([newX, newY] + bboxes[i][2:])
            
            selected_indices.append(i)

    return bboxes_in_tile, selected_indices


def boxes_in_tile_pixor(bboxes, corner_boxes, row_start, row_end, col_start, col_end, entire_img_shape):
    # gets the boxes within this tile, with coordinates relative to tile
    boxes_within_tile, selected_indices = boxes_in_tile(bboxes, row_start, row_end, col_start, col_end)

    pixel_box_labels = np.zeros((228, 228, 6))
    pixel_class_labels = np.zeros((228, 228, 1))

    print("looking at new tile")
    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            dx = 228
            dy = 228
            heading = 0
            width = 0
            length = 0
            in_a_box = 0
            for bbox_index in range(0,len(boxes_within_tile)):

                pixel = (r, c)

                if inside_box(pixel, corner_boxes[selected_indices[bbox_index]], entire_img_shape):
                    new_dx = abs(pixel[0] - boxes_within_tile[bbox_index][0])
                    new_dy = abs(pixel[1] - boxes_within_tile[bbox_index][1])
                    
                    if(np.sqrt(new_dx**2 + new_dy**2) <= np.sqrt(dx**2 + dy**2)):
                        dx = new_dx
                        dy = new_dy
                        heading, width, length = boxes_within_tile[bbox_index][2:]
                        in_a_box = 1

            new_r = r - row_start
            new_c = c - col_start
            pixel_box_labels[new_r, new_c,:] = [int(dx), int(dy), np.cos(heading), np.sin(heading), int(width), int(length)]
            pixel_class_labels[new_r, new_c] = in_a_box
    
    return pixel_box_labels, pixel_class_labels
            

# Takes array representing entire queried image and bounding boxes (with pixel coordinates) relative to
# entire image, and outputs a list of tuples where the first element is the tiled image and the second
# element is the list of bounding boxes with coordinates relative to the tile.
def tile_image(entire_img, b_boxes, corner_boxes, tile_size, indices_to_remove, grid=False):

    num_rows, num_cols, depth = entire_img.shape

    output_images = []
    output_boxes = []
    output_classes = []

    cell = 0  # used for gridding
    # For num tiles/grid cells
    total_rows = num_rows//tile_size
    total_cols = num_cols//tile_size
    print("total rows: " + str(total_rows))
    print("total columns: " + str(total_cols))
    for row in range(num_rows//tile_size):
        print("Progress: " + str(row/total_rows))
        for col in range(num_cols//tile_size):

            row_start = row*tile_size
            row_end = (row+1)*tile_size

            col_start = col*tile_size
            col_end = (col+1)*tile_size

            # row_end and col_end is not included in indexing, because array indexing is not end inclusive.
            if grid:
                tile = cell  # append number as label for grid cell used
            else:
                tile = entire_img[row_start:row_end, col_start:col_end, :]

            cell += 1

            # get bboxes in the tile in both cases (grid or not)
            pixel_box_labels, pixel_class_labels = boxes_in_tile_pixor(
                b_boxes, corner_boxes, row_start, row_end, col_start, col_end, entire_img.shape)
            
            # Add results to output
            output_images.append(tile)
            output_boxes.append(pixel_box_labels)
            output_classes.append(pixel_class_labels)

    new_output_images = []
    new_output_boxes = []
    new_output_classes = []

    for ind in range(len(output_images)):
        if ind not in indices_to_remove:
            new_output_images.append(output_images[ind])
            new_output_boxes.append(output_boxes[ind])
            new_output_classes.append(output_classes[ind])

    return new_output_images, new_output_boxes, new_output_classes

    # tiled_images = image.extract_patches_2d(entire_image, (tile_size, tile_size))
    # return tiled_images


def is_equal(input1, input2):

    if len(input1) != len(input2):
        return False

    for i in range(len(input1)):
        tile1, bboxes1 = input1[i]
        tile2, bboxes2 = input2[i]
        if not (np.all(tile1 == tile2) and bboxes1 == bboxes2):
            return False

    return True
