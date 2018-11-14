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
        im0 = np.zeros((10,12,3), dtype=int)
        bboxes0 = [[2,3,3,3],
                  [0,9,2,2],
                  [2,10,2,2],
                  [8,4,2,2]]
        tile_size0 = 4

        tile0 = np.zeros((4,4,3), dtype=int)
        exp0 = [(tile0, [[2,3,3,3]]),
                (tile0,[]),
                (tile0,[[0,1,2,2],[2,2,2,2]]),
                (tile0,[]),
                (tile0,[]),
                (tile0,[])]

        result0 = tile.tile_image(im0, bboxes0, tile_size0)

        # Need to define own equality function 

        self.assertTrue(tile.is_equal(result0,exp0))

        # self.assertTrue(np.all((exp0 == result0)))

    def test_grid(self): 
        # im0 = np.zeros((10,12,3), dtype=int)
        # bboxes0 = [[2,3,3,3],
        #           [0,9,2,2],
        #           [2,10,2,2],
        #           [8,4,2,2]]
        # tile_size0 = 4

        tile0 = np.zeros((4,4,3), dtype=int)
        tiles = [(tile0, [[2,3,3,3]]),
                (tile0,[]),
                (tile0,[[0,1,2,2],[2,2,2,2]]),
                (tile0,[]),
                (tile0,[]),
                (tile0,[])]
        
        expected0 = [(0, []),
                    (1, []),
                    (2, []),
                    (3, [[0,1,3,3]])]
        expected1 = [(0, []),
                    (1, []),
                    (2, []),
                    (3, [])]
        expected2 = [(0, [[0,1,2,2]]),
                    (1, []),
                    (2, []),
                    (3, [[0,0,2,2]])]
        result0 = tile.tile_image(tiles[0][0],tiles[0][1], 2, grid=True)
        result1 = tile.tile_image(tiles[1][0], tiles[1][1], 2, grid=True)
        result2 = tile.tile_image(tiles[2][0], tiles[2][1], 2, grid=True)
        
        self.assertListEqual([expected0,expected1,expected2],[result0,result1, result2])
    
    # Also add tests for other functions in this class. 
    



if __name__ == '__main__':
    unittest.main()