# Data processising

import numpy as np
import get_bounding_boxes
from sklearn.feature_extraction import image

# Gets the bounding boxes per tile, with centre relative to tile coordinates.


def boxes_in_tile(bboxes, row_start, row_end, col_start, col_end):

    bboxes_in_tile = []

    for i in range(len(bboxes)):
        centreX = bboxes[i][0]
        centreY = bboxes[i][1]

        if (col_start <= centreX < col_end) and (row_start <= centreY < row_end):

            # Changing bbox centre to be relative to tile
            newX = bboxes[i][0] - col_start
            newY = bboxes[i][1] - row_start

            # Mutating bboxes to reduce loop time after getting each set of bboxes per tile.
            bboxes_in_tile.append([newX, newY, bboxes[i][2], bboxes[i][3]])

    return bboxes_in_tile


# Takes array representing entire queried image and bounding boxes (with pixel coordinates) relative to
# entire image, and outputs a list of tuples where the first element is the tiled image and the second
# element is the list of bounding boxes with coordinates relative to the tile.
def tile_image(entire_img, b_boxes, tile_size, indices_to_remove, grid=False):

    num_rows, num_cols, depth = entire_img.shape

    output = []
    cell = 0  # used for gridding
    # For num tiles/grid cells
    for row in range(num_rows//tile_size):
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
            bboxes_in_tile = boxes_in_tile(
                b_boxes, row_start, row_end, col_start, col_end)
            
            # Add results to output
            output.append((tile, bboxes_in_tile))

    new_output = []
    for ind in range(len(output)):
        if ind not in indices_to_remove:
            new_output.append(output[ind])
    counter = 0
    for x in new_output:
        counter += len(x[1])
    print(counter)
    return new_output

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
