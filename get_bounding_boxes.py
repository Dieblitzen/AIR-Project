
from minimum_bounding_box import MinimumBoundingBox
import overpy
import numpy as np
import math
import matplotlib.pyplot as plt
# from shapely.geometry.polygon import LinearRing, Polygon
import statistics as st

api = overpy.Overpass()


def get_rect(nodes, YOLO=True, pad=1):
    # Gets either horizontally aligned (horizontal=True) or minimum bounding box for a set of nodes (horizontal=False)
    points = [(float(str(n.lat)), float(str(n.lon))) for n in nodes]
    if YOLO:
        lats = [node[0] for node in points]
        lons = [node[1] for node in points]
        lat_min = min(lats)
        lon_min = min(lons)
        lat_max = max(lats)
        lon_max = max(lons)

        width = (lon_max - lon_min)*pad
        height = (lat_max - lat_min)*pad
        centreX = lon_min + width/2
        centreY = lat_min + height/2
        bounding_box = [centreX, centreY, width, height]

        # bounding_box = [(lat_min,lon_min),(lat_max,lon_min),(lat_max,lon_max),(lat_min,lon_max)]

    else:
        bounding_box = MinimumBoundingBox(points).corner_points

    return bounding_box


# Bounding box of decided area in White Plains, NY
LAT_MIN, LON_MIN, LAT_MAX, LON_MAX = 41.009, - \
    73.7792749922, 41.03, -73.7582464736
LAT_HEIGHT = LAT_MAX - LAT_MIN
LON_WIDTH = LON_MAX - LON_MIN
LAT_STEP = 1/23 * LAT_HEIGHT
LON_STEP = 1/16 * LON_WIDTH

# Returns: a list of the bounding box location (4 lat/long coordinates)
#    for each building within the area defined by coordinate parameters.
#    lat_min, lat_max, long_min, long_max: represents the edges of the area
#    that we're interested in. Horizontal=True gets horizontal bounding boxes while
#    horizontal=False gets minimum bounding boxes


def get_bounding_boxes(YOLO=True):

    # query overpass for all buildings
    # date format will come above the way, as [date: "2016-01-01T00:00:00Z"];
    result = api.query(("""
        way
            ({}, {}, {}, {}) ["building"];
        (._;>;);
        out body;
        """).format(LAT_MIN, LON_MIN, LAT_MAX, LON_MAX))

    buildings = result.ways
    building_coordinates = []
    # use the imported package to find minimum bounding box
    for building in buildings:
        building_coordinates.append(
            list(get_rect(building.nodes, YOLO, pad=1.25)))
    return building_coordinates

#convert a pair of (lon, lat) coordinates into pixel pair
def convert_coord_to_pixel(coord, image_size, width, height, lat_max, lon_min):
   x = math.floor(((coord[1]-lon_min)/width)*image_size[1])
   y = math.floor(((lat_max-coord[0])/height)*image_size[0]) - 10
   return x, y

def get_two_closest_points(bbox):
    corner = np.array(bbox[0])
    distances = [(i,np.linalg.norm(corner-np.array(c))) for i, c in enumerate(bbox)]
    sorted_distances = sorted(distances, key= lambda pair: pair[1])
    closest_index, second_closest_index = sorted_distances[1][0], sorted_distances[2][0]
    closest, second_closest = np.array(bbox[closest_index]), np.array(bbox[second_closest_index])
    return corner, closest, second_closest, sorted_distances

#returns the angle of the heading. bbox is coordinates of the four corners
def get_pixor_box_dimensions(bbox):
    corner, closest, second_closest, sorted_distances = get_two_closest_points(bbox)
    vector = np.array(np.subtract(second_closest,corner)) if corner[0] > second_closest[0] else np.array(np.subtract(corner,second_closest))
    unit_vector = vector / np.linalg.norm(vector)
    width, length = sorted_distances[1][1], sorted_distances[2][1]
    return np.arccos(np.clip(np.dot(unit_vector, (1,0)), -1.0, 1.0)), width, length

def get_pixor_center(bbox):
    sorted_by_lat = sorted(bbox, key= lambda pair: pair[0])
    sorted_by_lon = sorted(bbox, key= lambda pair: pair[1])
    center_x = (sorted_by_lon[0][1] + sorted_by_lon[3][1])/2
    center_y = (sorted_by_lat[0][0] + sorted_by_lat[3][0])/2
    return center_x, center_y

def corner_boxes_in_pixels(image_size, buildings):
    fixed = []
    for building in buildings:
        # Pixels for a single building
        pixels = []
        box_pixels = [convert_coord_to_pixel(c, image_size, LON_WIDTH, LAT_HEIGHT, LAT_MAX, LON_MIN) for c in building]
        fixed.append(box_pixels)
    
    return fixed


def OSM_to_pixels(image_size, buildings, YOLO=True, PIXOR=False):
    ## bb_area: [lat_min, lon_min, lat_max, lon_max]
    ## image_size: image.shape[:2]
    # buildings: OSM fetched data

    # Define total lat and lon in area
    lat_max = LAT_MAX
    lat_min = LAT_MIN
    lon_min = LON_MIN
    width = LON_WIDTH
    height = LAT_HEIGHT
    bb_pixels = []
    # print(f"WIDTH: {width}, HEIGHT: {height}")

    if YOLO:
        # width_min = []
        # height_min = []
        for building in buildings:
            # Pixels for a single building
            pixels = []
            lonX = building[0]
            latY = building[1]

            centreX = math.floor(((lonX-lon_min)/width)*image_size[1])
            centreY = math.floor(((lat_max-latY)/height)*image_size[0]) - 10

            # This will break visualize.py
            # centreX = (centreX%38)/38.0
            # centreY = (centreY%38)/38.0

            widthPixel = math.floor((building[2]/width)*image_size[1])
            heightPixel = math.floor((building[3]/height)*image_size[0])

            # width_min.append(widthPixel)
            # height_min.append(heightPixel)

            pixels = [centreX, centreY, widthPixel, heightPixel]
            bb_pixels.append(pixels)

        # print('min width is {} and min height is {}'.format(min(width_min), min(height_min)))
        # print('median width is {} and median height is {}'.format(st.median(width_min), st.median(height_min)))

    elif PIXOR:
        for building in buildings:
            # Pixels for a single building
            pixels = []
            box_pixels = [convert_coord_to_pixel(c, image_size, LON_WIDTH, LAT_HEIGHT, lat_max, lon_min) for c in building]

            centreX, centreY = get_pixor_center(box_pixels)
            heading, width, length = get_pixor_box_dimensions(box_pixels)

            dimensions = [centreX, centreY, heading, width, length]
            bb_pixels.append(dimensions)

        # print('min width is {} and min height is {}'.format(min(width_min), min(height_min)))
        # print('median width is {} and median height is {}'.format(st.median(width_min), st.median(height_min)))


    else:
        # All bounding box pixels
        for building in buildings:
            # Pixels for a single building
            pixels = []
            for vertex in building:
                # vertex is a set (lat, lon)
                lat = vertex[0]
                lon = vertex[1]

                # want each element of pixels to be (pixel x, pixel y)
                pixel_x = math.floor(((lon-lon_min)/width)*image_size[1])
                pixel_y = math.floor(((lat_max-lat)/height)*image_size[0]) - 10
                pixels.append((pixel_x, pixel_y))

                # End for loop

                bb_pixels.append(pixels)

                # End for loop
    return bb_pixels

# white_plain_buildings = get_bounding_boxes(41.014456, -73.769573, 41.018465,-73.765043)
# OSM_to_pixels([41.014456, -73.769573, 41.018465,-73.765043],[100,100],white_plain_buildings)

# TODO: Convert Access_Pairs to a function, Filter OSM data for data outside the box.
