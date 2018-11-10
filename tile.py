## Data processising

import numpy as np
import get_bounding_boxes
import visualize
from sklearn.feature_extraction import image


def tile_image(entire_image, tile_size):

    tiled_images = image.extract_patches_2d(entire_image, (tile_size, tile_size))
    return tiled_images
