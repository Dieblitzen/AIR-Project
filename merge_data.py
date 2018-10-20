from get_bounding_boxes import get_bounding_boxes 
import numpy as np
import math

## Given: 
## lats and lons of query area
## image 
## building nodes 

## bb_coord = [lat_min, lon_min, lat_max, lon_max]

def OSM_to_pixels(bb_coord, image_size, buildings):
    ## bb_area: [lat_min, lon_min, lat_max, lon_max]
    ## image_size: image.size
    ## buildings: OSM fetched data 

    lat_min, lon_min, lat_max, lon_max = bb_coord[0], bb_coord[1], bb_coord[2], bb_coord[3]
    
    ## Define total lat and lon in area 
    width = lon_max-lon_min
    height = lat_max-lat_min
    
    ## All bounding box pixels
    bb_pixels = []
    for building in buildings: 
        ## Pixels for a single building
        pixels = []
        for vertex in building: 
            ## vertex is a set (lat, lon)
            lat = vertex[0]
            lon = vertex[1]
            
            ## want each element of pixels to be (pixel x, pixel y)
            pixel_x = math.floor(((lon-lon_min)/width)*image_size[0])
            pixel_y = math.floor(((lat-lat_min)/height)*image_size[1])
            pixels.append((pixel_x, pixel_y))
            
            # End for loop
            
        bb_pixels.append(pixels)
        
        # End for loop 
        
    return bb_pixels

white_plain_buildings = get_bounding_boxes(41.014456, -73.769573, 41.018465,-73.765043)
OSM_to_pixels([41.014456, -73.769573, 41.018465,-73.765043],[100,100],white_plain_buildings)

## TODO: Convert Access_Pairs to a function, Filter OSM data for data outside the box.