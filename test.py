## Test cases.

import get_bounding_boxes
import tile
import unittest
import numpy as np


#####################################################
# Tests for tile.py
#####################################################

class Tests(unittest.TestCase):

    def test_tile(self):
        im0 = np.zeros((10,12,3))
        bboxes0 = [[3,2,3,3],
                  [9,0,2,2],
                  [10,2,2,2],
                  [4,8,2,2]]
        tile_size0 = 4

        tile0 = np.zeros((4,4))
        exp0 = [(tile0, [[3,2,3,3]]),
                (tile0,[]),
                (tile0,[[1,0,2,2],[2,2,2,2]]),
                (tile0,[]),
                (tile0,[]),
                (tile0,[])]

        result0 = tile.tile_image(im0, bboxes0, tile_size0)
        print(result0)
        self.assertEqual(exp0, result0)

    
    # Also add tests for other functions in this class. 
    



if __name__ == '__main__':
    unittest.main()